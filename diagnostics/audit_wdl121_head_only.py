"""Replay head-only experiment and independently summarize the fixed continuations."""
import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import rules121 as rules
from az121 import encode
from checkpoint121_save import REGISTRY, digest
from curriculum121 import atomic, replay
from diagnostics.audit_wdl121_loop import audit_games, identity
from wdl121 import WDLNet121


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify_game(game, simulations):
    g = replay(game['opening'])
    assert g.ply == game['start_ply']
    assert len(game['paths']) == len(game['search_info'])
    rng = np.random.default_rng(game['seed'])
    states = {identity(g)}
    for i,(path,record) in enumerate(zip(game['paths'],game['search_info'])):
        assert g.result()[0] is None
        assert (g.ply,g.player,path)==(record['ply'],record['player'],record['path'])
        legal = [p for _,p in rules.get_all_moves(g.board,g.player)]
        info = record['info']
        assert legal==info['paths']
        assert info['root_visits']==info['child_visits']==sum(info['visits'])==simulations
        p = np.asarray(info['visits'],dtype=float)/simulations
        assert np.allclose(p,record['visit_target'])
        rng.dirichlet(np.full(len(legal),.3))
        expected = rng.choice(len(legal),p=p) if i<12 else p.argmax()
        assert path==legal[expected]
        g.play(path)
        states.add(identity(g))
    assert g.ply==game['plies'] and g.result()==(game['winner'],game['reason'])
    return states


def binary(records, side=None):
    groups = defaultdict(list)
    for row in records:
        if row['label']==1 or (side is not None and row['player']!=side):
            continue
        win,_,loss = row['probabilities']
        groups[row['game_id']].append((win/(win+loss)-(row['label']==0))**2)
    return float(np.mean([np.mean(rows) for rows in groups.values()])) if groups else None


def points(wdl):
    return wdl[0]+.5*wdl[1]


def summarize_probes(roots, predictions, games):
    records = []
    for root in roots:
        pred = predictions[root['group']]
        arms = []
        for arm in range(3):
            ss = [g for g in games if g['group']==root['group'] and g['arm']==arm]
            assert {g['repeat'] for g in ss}==set(range(8)) and len(ss)==8
            counts = Counter('截断' if g['reason']=='max_ply' else '和' if g['winner']==0
                             else '胜' if g['winner']==root['player'] else '负' for g in ss)
            n = 8-counts['截断']
            arms.append(dict(counts=dict(counts),n=n,empirical_win=counts['胜']/n if n else None,
                             empirical_points=(counts['胜']+.5*counts['和'])/n if n else None))
        a,b = arms[1:]
        gap = a['empirical_points']-b['empirical_points'] if min(a['n'],b['n'])>=7 else None
        label = (1 if gap>0 else -1) if gap is not None and abs(gap)>=.25 else 0
        records.append(dict(group=root['group'],arms=arms,empirical_gap=gap,label=label,
                            predictions=pred))
    score = {}
    for name in ('parent','candidate'):
        for mode in ('raw_wdl','search_wdl'):
            groups = [r for r in records if r['arms'][0]['n'] and r['predictions'][name][mode] is not None]
            errors = []
            bins = defaultdict(list)
            for row in groups:
                p = row['predictions'][name][mode][0]
                arm = row['arms'][0]
                errors.append((p*p*(arm['n']-arm['counts'].get('胜',0))+(1-p)**2*arm['counts'].get('胜',0))/arm['n'])
                index = int(np.searchsorted([.4,.5,.6],p,side='right'))
                bins[index].append((p,arm['empirical_win']))
            pair_correct = pair_n = 0
            for row in records:
                a,b = [c[mode] for c in row['predictions'][name]['children']]
                if row['label'] and a is not None and b is not None:
                    pair_n += 1
                    pair_correct += (points(a)-points(b))*row['label']>0
            score[name+'/'+mode] = dict(win_brier=float(np.mean(errors)) if errors else None,
                root_count=len(errors),pair_correct=int(pair_correct),pair_n=pair_n,
                bins={str(k):dict(n=len(v),predicted=float(np.mean([x[0] for x in v])),
                                  empirical=float(np.mean([x[1] for x in v]))) for k,v in bins.items()})
    return dict(roots=records,scores=score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment',type=Path,required=True)
    e = ap.parse_args().experiment
    load = lambda n:json.loads((e/n).read_text())
    cfg, result = load('config.json'),load('result.json')
    torch.set_num_threads(1)
    for name,sha in cfg['source_sha256'].items():
        assert digest(e/'source'/name)==sha and digest(name)==sha,name
    assert digest(cfg['parent'])==cfg['parent_sha'] and digest(REGISTRY)==cfg['registry_sha']
    freeze = load('dataset_freeze.json')
    for file,key in [('schedule.json','schedule_sha'),('samples.pt','samples_sha'),('games.jsonl','games_sha')]:
        assert digest(e/file)==freeze[key]
    schedule, games = load('schedule.json'), jsonl(e/'games.jsonl')
    history=set(load('history_exclusions.json')['orbits'])
    replay_audit = audit_games(games[:72],schedule,history,{**cfg,'iteration':3},cfg['parent'])
    assert len(games)==80
    for row, game in zip(schedule[72:],games[72:]):
        assert game['group']==row['group'] and game['opening']==row['opening']
        assert game['seed']==cfg['selfplay_seed']+row['group']
        verify_game(game,32)
    samples=torch.load(e/'samples.pt',weights_only=True)
    assert {k:len(v) for k,v in samples.items()}==replay_audit['retained']
    assert not ({s['orbit'] for s in samples['train']} & {s['orbit'] for s in samples['development']})
    lookup={(s['group'],s['ply']):s for pool in samples.values() for s in pool}
    assert len(lookup)==sum(map(len,samples.values()))
    data_orbits=set()
    for game in games[:72]:
        g=replay(game['opening'])
        for path in game['paths']:
            oid=identity(g)
            data_orbits.add(oid)
            s=lookup.get((game['group'],g.ply))
            if s is not None:
                expected=None if game['reason']=='max_ply' else 1 if game['winner']==0 else 0 if game['winner']==g.player else 2
                assert s['wdl_class']==expected and s['terminal_ply']==game['plies']
                assert s['player']==g.player and s['orbit']==oid and oid not in history
                assert np.array_equal(s['x'].numpy(),encode(g.board,g.player,g.ply,g.repetitions,canonical=True))
            g.play(path)
        data_orbits.add(identity(g))
    result_audit=dict(replay=replay_audit,source_games=8,hashes_verified=len(cfg['source_sha256']))
    if not result['training_started']:
        atomic(e/'independent_audit.json',result_audit)
        return
    parent=WDLNet121.load(cfg['parent']).requires_grad_(False)
    candidate_path=e/'candidate_step1200.pt'
    assert digest(candidate_path)==result['candidate_sha']
    model=WDLNet121.load(candidate_path).requires_grad_(False)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in parent.state_dict().items() if not k.startswith('wdl.'))
    full,late=defaultdict(list),defaultdict(list)
    for i,s in enumerate(samples['train']):
        if s['wdl_class'] is not None:
            full[s['group']].append(i)
            if s['terminal_ply']-s['ply']<=16:
                late[s['group']].append(i)
    rng=np.random.default_rng(113123)
    trace=jsonl(e/'sample_trace.jsonl')
    assert len(trace)==1200
    for step,row in enumerate(trace):
        indices=[]
        for pool in (full,late):
            keys=sorted(pool)
            indices.extend(int(rng.choice(pool[keys[i]])) for i in rng.integers(len(keys),size=8))
        assert row['step']==step and row['indices']==indices and np.isfinite(row['loss'])
    d=load('development.json')
    metrics = {}
    for stage in ('late','middle'):
        def selected(s):
            remaining = s['terminal_ply']-s['ply']
            return remaining<=16 if stage=='late' else s['ply']>=30 and remaining>16

        ss = [s for s in samples['development'] if s['wdl_class'] is not None and selected(s)]
        for name, net in [('parent',parent),('candidate',model)]:
            rows = d[stage][name]['records']
            assert len(rows)==len(ss)
            for start in range(0,len(ss),32):
                batch = ss[start:start+32]
                with torch.no_grad():
                    probabilities = net.predict_wdl(torch.stack([s['x'] for s in batch])).numpy()
                expected = rows[start:start+32]
                assert np.allclose(probabilities,[row['probabilities'] for row in expected],atol=2e-6)
                for s,row in zip(batch,expected):
                    assert (s['ply'],str(s['group']),s['player'],s['wdl_class']) == (row['ply'],row['game_id'],row['player'],row['label'])
        metrics[stage] = {name:binary(d[stage][name]['records']) for name in ('parent','prior','candidate')}
        for name in metrics[stage]:
            assert np.isclose(metrics[stage][name],d[stage]['decisive'][name]['overall']['game_macro_brier'])
    winners={1:set(),2:set()}
    for row in d['late']['candidate']['records']:
        if row['label']!=1:
            winners[row['player'] if row['label']==0 else 3-row['player']].add(row['game_id'])
    l=metrics['late'];m=metrics['middle'];coverage=min(map(len,winners.values()))>=2
    checks=dict(decisive_coverage=coverage,decisive_parent_gain005=coverage and l['candidate']<=l['parent']-.005,
        decisive_prior_gain005=coverage and l['candidate']<=l['prior']-.005,
        middle_nonworse=d['middle']['decisive']['candidate']['overall']['n_games']>=8 and m['candidate']<=m['parent']+.01,
        frozen_parameters=True,policy_logits_equal=True)
    for side in (1,2):
        c,p=[binary(d['late'][name]['records'],side) for name in ('candidate','parent')]
        subset=d['late']['decisive']['candidate']['by_player'][str(side)]
        checks['player'+str(side)+'_decisive_nonworse']=bool(subset['wins'] and subset['losses'] and c<=p)
    assert checks==result['checks'] and all(checks.values())==result['development_pass']
    roots=load('probe_schedule.json')
    assert digest(e/'probe_schedule.json')==load('probe_freeze.json')['sha256']
    predictions={r['group']:r for r in jsonl(e/'probe_predictions.jsonl')} if roots else {}
    probe_games=jsonl(e/'probe_games.jsonl') if roots else []
    assert len(probe_games)==len(roots)*24
    root_map={r['group']:r for r in roots}
    blocked=history|data_orbits
    for root in roots:
        g=replay(root['opening'])
        assert g.result()[0] is None and (g.player,g.ply,identity(g))==(root['player'],root['ply'],root['orbit'])
        assert g.ply>=50 and max(sum(g.board[i]==side for i in rules.GOAL[side]) for side in (1,2))>=8
        assert root['orbit'] not in blocked
        blocked.add(root['orbit'])
        for option in root['options']:
            child=copy.deepcopy(g)
            child.play(option['path'])
            assert child.result()[0] is None and identity(child)==option['orbit'] and option['orbit'] not in blocked
            blocked.add(option['orbit'])
    owners=defaultdict(set)
    for game in probe_games:
        root=root_map[game['group']]
        suffix=[] if game['arm']==0 else [root['options'][game['arm']-1]['path']]
        assert game['opening']==root['opening']+suffix
        assert game['seed']==113124+game['group']*100+game['repeat']
        for oid in verify_game(game,16):
            owners[oid].add(game['group'])
    probe = summarize_probes(roots,predictions,probe_games)
    probe['cross_root_trajectory_orbits']=sum(len(v)>1 for v in owners.values())
    probe['trajectory_orbits_in_train_or_dev']=len(set(owners)&data_orbits)
    atomic(e/'probe_summary.json',probe)
    result_audit.update(trace_steps=len(trace),frozen_parameters=True,development_checks=checks,
                        decisive_brier=metrics,probe_games=len(probe_games),probe_roots=len(roots))
    atomic(e/'independent_audit.json',result_audit)


if __name__=='__main__':
    main()
