"""Freeze policy and trunk, then verify WDL learning against fresh outcomes."""
import argparse
import copy
import fcntl
import json
import os
import shutil
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import rules121 as rules
from az121 import action_id, encode, search
from checkpoint121_save import REGISTRY, digest
from collect121_order_pairs import build_history
from curriculum121 import atomic, replay
from train121_wdl_loop import decisive_checks, decisive_metrics, prior_for_training
from wdl121 import WDLNet121
from wdl121_data import build_samples, identity, make_schedule, run_selfplay
from wdl121_metrics import metrics_from_probabilities, wdl_metrics

PROTOCOL = Path('docs/experiments/wdl121_head_only_20261107/protocol.md')
PARENT_SHA = '0e9d850853771c0b0d0e58d6bcddc7e972378548ea9c8bb4461b6a01cae8c79d'


def pool_by_game(samples):
    groups = defaultdict(list)
    for index, sample in enumerate(samples):
        if sample['wdl_class'] is not None:
            groups[sample['group']].append(index)
    return dict(sorted(groups.items()))


def draw_indices(rng, pools, n):
    keys = list(pools)
    return [int(rng.choice(pools[keys[int(i)]])) for i in rng.integers(len(keys), size=n)]


def encoded(game, model):
    return torch.from_numpy(encode(game.board, game.player, game.ply,
        game.repetitions, canonical=model.canonical)).unsqueeze(0)


@torch.no_grad()
def frozen_features(model, samples):
    features = []
    hook = model.wdl.register_forward_pre_hook(lambda module, args: features.append(args[0].clone()))
    try:
        for start in range(0, len(samples), 64):
            model.forward_wdl(torch.from_numpy(np.stack([s['x'] for s in samples[start:start+64]])))
    finally:
        hook.remove()
    return torch.cat(features)


def late(sample):
    return sample['terminal_ply'] - sample['ply'] <= 16


def trajectory_orbits(game):
    g = replay(game['opening'])
    yield identity(g)
    for path in game['paths']:
        g.play(path)
        yield identity(g)


@torch.no_grad()
def choose_probe(model, game, blocked):
    g = replay(game['opening'])
    prefix = list(game['opening'])
    for path in game['paths']:
        in_goal = [sum(g.board[i] == side for i in rules.GOAL[side]) for side in (1, 2)]
        if g.ply >= 50 and max(in_goal) >= 8 and identity(g) not in blocked:
            moves = [p for _, p in rules.get_all_moves(g.board, g.player)]
            logits, _ = model(encoded(g, model))
            ranked = sorted(moves, key=lambda p: -float(logits[0, action_id(p[0], p[-1], g.player, model.canonical)]))
            options = []
            used = blocked | {identity(g)}
            for move in ranked:
                child = copy.deepcopy(g)
                child.play(move)
                oid = identity(child)
                if child.result()[0] is None and oid not in used:
                    options.append(dict(path=move, orbit=oid))
                    used.add(oid)
                    if len(options) == 2:
                        return dict(group=game['group'], opening=prefix, ply=g.ply, player=g.player,
                                    orbit=identity(g), in_goal=in_goal, options=options)
        g.play(path)
        prefix.append(path)
    return None


@torch.no_grad()
def probe_predictions(model, root):
    g = replay(root['opening'])
    _, info = search(model, g.board, g.player, g.ply, g.repetitions,
                     simulations=128, wdl_stats=True)
    children = []
    for item in root['options']:
        child = copy.deepcopy(g)
        child.play(item['path'])
        p = model.predict_wdl(encoded(child, model))[0].tolist()
        index = info['paths'].index(item['path'])
        children.append(dict(raw_wdl=p[::-1], search_wdl=info['child_wdl'][index],
                             visits=int(info['visits'][index])))
    return dict(raw_wdl=model.predict_wdl(encoded(g, model))[0].tolist(),
                search_wdl=info['root_wdl'], children=children,
                chosen=info['paths'][int(np.argmax(info['visits']))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    args = parser.parse_args()
    if digest(args.parent) != PARENT_SHA:
        raise ValueError('父模型SHA不符')
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    lock = (out/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    began = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(113120)
    parent = WDLNet121.load(args.parent).requires_grad_(False)
    source = [Path(n) for n in ('train121_wdl_head_only.py','train121_wdl_loop.py',
        'wdl121.py','wdl121_data.py','wdl121_metrics.py','az121.py','rules121.py',
        'curriculum121.py','collect121_order_pairs.py','expert121_pilot48.py',
        'history121_exclusions.py','opening121_split_audit.py')] + [PROTOCOL]
    hashes = {str(n):digest(n) for n in source}
    registry_sha = digest(REGISTRY)
    for path in source:
        dest = out/'source'/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    atomic(out/'config.json', dict(roadmap_step=3,profile=rules.PROFILE,
        parent=str(args.parent.resolve()),parent_sha=PARENT_SHA,registry_sha=registry_sha,
        source_sha256=hashes,groups=80,train=48,development=24,probe=8,
        schedule_seed=113121,selfplay_seed=113122,sampling_seed=113123,
        probe_seed=113124,steps=1200,lr=.0003,simulations=32,probe_simulations=16,
        repeats=8,max_seconds=5400))

    def status(stage, **extra):
        seconds = time.monotonic()-began
        atomic(out/'status.json', dict(stage=stage,pid=os.getpid(),seconds=seconds,**extra))
        if seconds > 5400:
            raise TimeoutError('固定预算耗尽')

    def append(name, row):
        with (out/name).open('a') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')

    try:
        status('历史覆盖预检')
        history = build_history(out/'history',exclude_parent=True)
        atomic(out/'history_exclusions.json',history)
        historical = set(history['orbits'])
        schedule = make_schedule(parent,historical,count=80,train=48,development=24,
                                 seed=113121,status=status)
        atomic(out/'schedule.json',schedule)
        schedule_sha = digest(out/'schedule.json')
        atomic(out/'schedule_freeze.json',dict(sha256=schedule_sha,groups=80))
        games = []
        for row in schedule:
            game = run_selfplay(parent,row,seed=113122+row['group'],status=status)
            append('games.jsonl',game)
            games.append(game)
        pools, data = build_samples(games[:72],schedule,historical)
        atomic(out/'data_audit.json',data)
        train, dev = pools['train'], pools['development']
        torch.save({part:[{**s,'x':torch.from_numpy(s['x'])} for s in rows]
                    for part,rows in pools.items()},out/'samples.pt')
        atomic(out/'dataset_freeze.json',dict(schedule_sha=schedule_sha,
            games_sha=digest(out/'games.jsonl'),samples_sha=digest(out/'samples.pt')))
        all_pool = pool_by_game(train)
        late_pool = {k:[i for i in v if late(train[i])] for k,v in all_pool.items()}
        late_pool = {k:v for k,v in late_pool.items() if v}
        if len(all_pool)<24 or len(late_pool)<16:
            atomic(out/'result.json',dict(training_started=False,complete_games=len(all_pool),
                late_games=len(late_pool),reason='训练覆盖不足'))
            status('complete')
            return
        blocked = historical | set().union(*(set(trajectory_orbits(g)) for g in games[:72]))
        probes = []
        for game in games[72:]:
            root = choose_probe(parent,game,blocked)
            if root is not None:
                probes.append(root)
                blocked.update([root['orbit'],*(o['orbit'] for o in root['options'])])
        atomic(out/'probe_schedule.json',probes)
        atomic(out/'probe_freeze.json',dict(sha256=digest(out/'probe_schedule.json'),roots=len(probes)))
        model = copy.deepcopy(parent)
        model.wdl.requires_grad_(True)
        features = frozen_features(parent,train)
        labels = torch.tensor([s['wdl_class'] if s['wdl_class'] is not None else 0 for s in train])
        optimizer = torch.optim.AdamW(model.wdl.parameters(),lr=.0003,weight_decay=.0001)
        rng = np.random.default_rng(113123)
        for step in range(1200):
            if step%50==0:
                status('只训练胜率头',step=step,total=1200)
            indices = draw_indices(rng,all_pool,8)+draw_indices(rng,late_pool,8)
            loss = F.cross_entropy(model.wdl(features[indices]),labels[indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.wdl.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            append('sample_trace.jsonl',dict(step=step,indices=indices,loss=float(loss.detach())))
        model.eval().requires_grad_(False)
        model.save(out/'candidate_step1200.pt',dict(iteration=3,training='WDL head only',parent_wdl_sha256=PARENT_SHA))
        frozen = all(torch.equal(value,model.state_dict()[key])
                     for key,value in parent.state_dict().items() if not key.startswith('wdl.'))
        logits_equal = True
        with torch.no_grad():
            for start in range(0,len(dev),32):
                x = torch.from_numpy(np.stack([s['x'] for s in dev[start:start+32]]))
                logits_equal &= torch.equal(parent(x)[0],model(x)[0])
        assessments = {}
        for name, select in [('late',late),('middle',lambda s:s['ply']>=30 and not late(s))]:
            ds = [s for s in dev if select(s)]
            prior = prior_for_training([s for s in train if select(s)])
            parent_metrics = wdl_metrics(parent,ds)
            candidate_metrics = wdl_metrics(model,ds)
            prior_metrics = metrics_from_probabilities(ds,[prior[s['player']] for s in ds])
            assessments[name] = dict(parent=parent_metrics,candidate=candidate_metrics,prior=prior_metrics,
                decisive={k:decisive_metrics(m) for k,m in [('parent',parent_metrics),
                    ('candidate',candidate_metrics),('prior',prior_metrics)]})
        lm = assessments['late']
        checks = decisive_checks(lm['candidate'],lm['parent'],lm['prior'])
        mid = assessments['middle']['decisive']
        m,p = mid['candidate']['overall'],mid['parent']['overall']
        checks['middle_nonworse']=m['n_games']>=8 and m['game_macro_brier']<=p['game_macro_brier']+.01
        checks.update(frozen_parameters=frozen,policy_logits_equal=logits_equal)
        atomic(out/'development.json',assessments)
        for root in probes:
            append('probe_predictions.jsonl',dict(group=root['group'],
                parent=probe_predictions(parent,root),candidate=probe_predictions(model,root)))
            for arm, forced in enumerate([None,*[o['path'] for o in root['options']]]):
                opening = root['opening'] + ([forced] if forced else [])
                g = replay(opening)
                row = dict(group=root['group'],split='probe',opening=opening,board=g.board,
                           player=g.player,ply=g.ply)
                for repeat in range(8):
                    game=run_selfplay(parent,row,simulations=16,seed=113124+root['group']*100+repeat,status=status)
                    append('probe_games.jsonl',dict(**game,arm=arm,repeat=repeat))
        assert digest(args.parent)==PARENT_SHA and digest(REGISTRY)==registry_sha
        assert digest(out/'schedule.json')==schedule_sha
        assert all(digest(n)==sha for n,sha in hashes.items())
        atomic(out/'result.json',dict(training_started=True,checks=checks,development_pass=all(checks.values()),
            probe_roots=len(probes),probe_games=len(probes)*24,steps=1200,seconds=time.monotonic()-began,
            parent_sha=PARENT_SHA,candidate_sha=digest(out/'candidate_step1200.pt')))
        status('complete')
    except BaseException:
        atomic(out/'status.json',dict(stage='failed',pid=os.getpid(),error=traceback.format_exc(),
                                      seconds=time.monotonic()-began))
        raise


if __name__=='__main__':
    main()
