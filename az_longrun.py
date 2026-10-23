"""Resumable, logged self-play research run. Never overwrites the GUI champion.
CPU-only, one Torch thread; actual terminal outcomes only. See SELFPLAY_LONGRUN.md.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import time
import traceback

import numpy as np
import torch
from alphazero_nn import expand_game_records
import Chinese_Checker as cc
from checkpoint_paths import ROOT, artifact_path
from nn_resnet import NetResNet, _make_batch, train
from self_play_resnet_mcts import mcts_think_resnet


def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    temp.replace(path)


def atomic_pickle(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as f:
        pickle.dump(obj, f)
    temp.replace(path)


def samples_for(records, discount):
    result = []
    for rec in records:
        if rec['winner'] not in (1, 2):
            continue
        samples = expand_game_records([rec])
        for i, sample in enumerate(samples):
            sample['z'] *= discount ** (len(samples) - i)
        result.extend(samples)
    return result


@torch.no_grad()
def metrics(model, samples):
    model.eval()
    if not samples:
        return None
    totals = np.zeros(4)
    for i in range(0, len(samples), 32):
        batch = samples[i:i+32]
        xs, src, dst, pi, mask, z, vw = _make_batch(batch, 'cpu')
        logits, value = model(xs)
        logits = (logits.gather(1, src) + logits.gather(1, dst)).masked_fill(mask == 0, -1e30)
        ce = -(pi * torch.log_softmax(logits, 1)).sum(1)
        totals += [float(ce.sum()), float(((value-z)**2).sum()),
                   int((value.abs() > .99).sum()), float(value.abs().sum())]
    return dict(policy_ce=totals[0]/len(samples), value_mse=totals[1]/len(samples),
                saturated_fraction=totals[2]/len(samples), mean_abs_value=totals[3]/len(samples),
                samples=len(samples))


def validate(rec):
    board = cc.init_board()
    seen = set()
    repeats = 0
    for ply, step in enumerate(rec['steps']):
        player = 1 + ply % 2
        assert player == step['player']
        key = (tuple(board), player)
        repeats += key in seen
        seen.add(key)
        legal = {(s, tuple(p)) for s, p in cc.get_all_moves(board, player)}
        assert (step['move_path'][0], tuple(step['move_path'])) in legal
        assert len(step['moves']) == len(step['pi'])
        assert all((s, tuple(p)) in legal and d == p[-1] for s, d, p in step['moves'])
        pi = np.asarray(step['pi'])
        assert np.isfinite(pi).all() and np.all(pi >= 0) and abs(pi.sum()-1) < 1e-5
        board = cc.apply_move(board, step['move_path'])
    assert cc.is_game_over(board) == rec['winner']
    return dict(plies=len(rec['steps']), repeated_positions=repeats, errors=0)


def play(models, seed, simulations, discounts, explore=False, opening=0, max_moves=320, progress=None):
    budgets = [simulations, simulations] if isinstance(simulations, int) else list(simulations)
    if len(budgets) != 2 or any(not isinstance(n, int) or n < 1 for n in budgets):
        raise ValueError("simulations must be a positive int or two positive ints")
    search_seconds = [0.0, 0.0]
    search_moves = [0, 0]
    rng = np.random.default_rng(seed)
    board = cc.init_board()
    seen, steps = {}, []
    reason = 'move_limit'
    for ply in range(max_moves):
        player = 1 + ply % 2
        key = (tuple(board), player)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 3:
            reason = 'repetition'
            break
        if ply < opening:
            moves = [(s, p[-1], p) for s, p in cc.get_all_moves(board, player)]
            pi = np.full(len(moves), 1/max(1, len(moves)))
        else:
            search_started = time.monotonic()
            (moves, pi), _ = mcts_think_resnet(
                models[player-1], board, player, budgets[player-1],
                add_root_noise=explore, rng=rng, history=seen, discount=discounts[player-1])
            search_seconds[player-1] += time.monotonic() - search_started
            search_moves[player-1] += 1
        if not moves:
            reason = 'no_moves'
            break
        probs = np.asarray(pi, dtype=np.float64)
        probs /= probs.sum()
        # Only actual MCTS visitation targets enter training, never invented wins.
        choice = rng.choice(len(moves), p=probs) if (explore and ply < 24) or ply < opening else np.argmax(probs)
        path = moves[int(choice)][2]
        steps.append(dict(player=player, move_path=list(path), moves=moves, pi=probs.tolist()))
        board = cc.apply_move(board, path)
        if cc.is_game_over(board) is not None:
            reason = 'terminal'
            break
        if progress and (ply+1) % 40 == 0:
            progress(ply+1)
    rec = dict(winner=cc.is_game_over(board), reason=reason, steps=steps,
               seed=seed, explore=explore, simulations=simulations, discounts=discounts, budgets=budgets, search_seconds=search_seconds,
               search_moves=search_moves)
    rec['validation'] = validate(rec)
    return rec


class Run:
    def __init__(self, args):
        self.args = args
        self.root = args.out
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root/'worker.lock').open('w')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.lock.write(str(os.getpid())); self.lock.flush()
        self.state = dict(pid=os.getpid(), started=stamp(), deadline=args.deadline,
                          status='running', phase='setup', round=0)
        self.event('started', deadline=args.deadline, pid=os.getpid())

    def event(self, event, **details):
        row = dict(time=stamp(), event=event, **details)
        with (self.root/'events.jsonl').open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def status(self, **changes):
        self.state.update(changes, updated=stamp())
        atomic_json(self.root/'status.json', self.state)

    def set_games(self, directory, label, count, make_game):
        path = directory/(label+'.pkl')
        records = pickle.loads(path.read_bytes()) if path.exists() else []
        for i in range(len(records), count):
            self.status(phase=label, game=i+1, games=count, ply=0)
            started = time.monotonic()
            rec = make_game(i, lambda ply:self.status(ply=ply))
            records.append(rec)
            atomic_pickle(path, records)
            self.event('game', round=self.state['round'], phase=label, game=i+1,
                       winner=rec['winner'], reason=rec['reason'], plies=len(rec['steps']),
                       seconds=round(time.monotonic()-started, 2))
        return records

    def execute(self):
        torch.set_num_threads(1)
        torch.manual_seed(20261022)
        config = dict(vars(self.args)); config['out'] = str(config['out'])
        config.update(value_scale=.1, lr=2e-5, batch_size=24, discount=.995,
                      model_promotion='internal only: >=3 of 4 incumbent wins and >=3 completed; no GUI promotion',
                      source='fresh selfplay only; warm-start existing champion; no book or teacher labels')
        config_path=self.root/'config.json'
        if not config_path.exists():
            atomic_json(config_path, config)
        snapshot=self.root/'source';snapshot.mkdir(exist_ok=True)
        for name in ['az_longrun.py','mcts_history.py','self_play_resnet_mcts.py','alphazero_nn.py',
                     'nn_resnet.py','nn_torch.py','Chinese_Checker.py','checkpoint_paths.py']:
            if not (snapshot/name).exists():shutil.copy(name,snapshot/name)
        if not (self.root/'frozen_baseline.pt').exists():
            source=artifact_path(ROOT/'params_resnet_champ.pt')
            shutil.copy(source,self.root/'frozen_baseline.pt')
            self.event('baseline',source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        frozen=NetResNet.load(self.root/'frozen_baseline.pt').eval()
        if not (self.root/'bootstrap.pt').exists():
            warm=copy.deepcopy(frozen)
            with torch.no_grad():
                warm.v_fc2.weight.mul_(.1);warm.v_fc2.bias.mul_(.1)
            warm.save(self.root/'bootstrap.pt')
            # Diagnose only: previous new selfplay isn't reused as training or holdout.
            prior=ROOT/'experiments/selfplay_20261022_b/selfplay.pkl'
            diagnostic_samples=samples_for(pickle.loads(prior.read_bytes()),.995)
            atomic_json(self.root/'value_diagnostic.json',dict(
                baseline=metrics(frozen,diagnostic_samples),scaled=metrics(warm,diagnostic_samples)))
        current=self.root/'bootstrap.pt'
        train_records=[]
        round_id=0
        while time.time() < self.args.deadline or round_id < 2:
            round_id+=1
            directory=self.root/f'round_{round_id:02d}';directory.mkdir(exist_ok=True)
            self.status(round=round_id,current_checkpoint=str(current),phase='round_setup')
            incumbent=NetResNet.load(current).eval()
            training=self.set_games(directory,'selfplay',self.args.games,
                lambda i,progress:play([incumbent,incumbent], 300000+round_id*100+i,
                    self.args.simulations,[.995,.995],explore=True,progress=progress))
            complete=[r for r in training if r['winner'] in (1,2)]
            # Split whole games. Validation games never enter the persistent training pool.
            heldout=complete[-2:] if len(complete)>=4 else complete[-1:]
            train_records.extend(complete[:-len(heldout)] if heldout else complete)
            train_records=train_records[-48:]
            train_samples=samples_for(train_records,.995)
            val_samples=samples_for(heldout,.995)
            if not train_samples:
                self.event('no_training_data',round=round_id,complete=len(complete))
                atomic_json(directory/'result.json',dict(skipped=True,completed=len(complete)))
                continue
            result_path=directory/'result.json'
            if result_path.exists():
                result=json.loads(result_path.read_text())
                if result.get('accepted'):current=directory/'candidate.pt'
                continue
            candidate=copy.deepcopy(incumbent)
            before=metrics(candidate,val_samples)
            checkpoint=directory/'candidate.pt'
            if checkpoint.exists():
                candidate=NetResNet.load(checkpoint).eval()
            else:
                self.status(phase='training',samples=len(train_samples),validation_samples=len(val_samples))
                loss=train(train_samples,candidate,steps=self.args.steps,lr=2e-5,bs=24,
                    seed=1000+round_id,log_every=100,
                    ckpt_cb=lambda step,model:self.status(train_step=step) if step%50==0 else None)
                candidate.eval().save(checkpoint)
                self.event('trained',round=round_id,samples=len(train_samples),loss=loss)
            after=metrics(candidate,val_samples)
            atomic_json(directory/'validation.json',dict(before=before,after=after,
                training_games=len(train_records),heldout_games=len(heldout),
                train_seeds=[r['seed'] for r in train_records],val_seeds=[r['seed'] for r in heldout]))
            # Disjoint opening seeds across rounds AND between selection and frozen testing.
            def match(i,progress,opponent,seedbase,opponent_discount):
                side=1+i%2
                models=[candidate,opponent] if side==1 else [opponent,candidate]
                discounts=[.995,opponent_discount] if side==1 else [opponent_discount,.995]
                rec=play(models,seedbase+i//2,self.args.simulations,discounts,opening=4,progress=progress)
                rec['candidate_side']=side
                return rec
            selection=self.set_games(directory,'incumbent_eval',4,
                lambda i,p:match(i,p,incumbent,500000+round_id*100,.995))
            testing=self.set_games(directory,'frozen_eval',4,
                lambda i,p:match(i,p,frozen,800000+round_id*100,1.0))
            def score(records):
                return dict(wins=sum(r['winner']==r['candidate_side'] for r in records),
                            losses=sum(r['winner']==3-r['candidate_side'] for r in records),
                            unfinished=sum(r['winner'] is None for r in records),
                            mean_plies=sum(len(r['steps']) for r in records)/len(records))
            gate=score(selection)
            accepted=gate['wins']>=3 and gate['unfinished']<=1
            result=dict(round=round_id,selfplay_complete=len(complete),selfplay_games=len(training),
                training_samples=len(train_samples),validation_before=before,validation_after=after,
                incumbent=gate,frozen=score(testing),accepted=accepted,gui_promoted=False)
            atomic_json(result_path,result)
            self.event('round_complete',**result)
            if accepted:current=checkpoint
            self.status(phase='round_complete',current_checkpoint=str(current))
        self.status(status='complete',phase='complete',current_checkpoint=str(current))
        self.event('complete',rounds=round_id,current_checkpoint=str(current))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=ROOT/'experiments/longrun_20261022')
    ap.add_argument('--deadline',type=float,required=True)
    ap.add_argument('--games',type=int,default=8)
    ap.add_argument('--simulations',type=int,default=64)
    ap.add_argument('--steps',type=int,default=500)
    args=ap.parse_args()
    run=Run(args)
    try:run.execute()
    except BaseException:
        run.status(status='failed',phase='failed',error=traceback.format_exc())
        run.event('failed',error=traceback.format_exc())
        raise

if __name__=='__main__':main()
