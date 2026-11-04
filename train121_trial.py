"""Bounded 121-cell bootstrap -> MCTS selfplay -> train -> paired evaluation.
No automatic champion promotion. Every stage and genuine rule draw is explicit.
"""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import rules121 as r
from engine_c121 import CSearch121
from az121 import Net121,Neural121,encode,train,search,action_id,FORMAT,ENCODING


def play(engines,seed,explore=False,opening=0,source='evaluation'):
    rng=np.random.default_rng(seed);g=r.Game();records=[];samples=[]
    while g.result()[0] is None:
        p=g.player;engine=engines[p];moves=r.get_all_moves(g.board,p)
        if g.ply<opening:
            i=int(rng.integers(len(moves)));path=moves[i][1];policy=None
        elif isinstance(engine,Neural121):
            path,info=search(engine.model,g.board,p,g.ply,g.repetitions,engine.simulations,rng,noise=explore,value_scale=engine.value_scale,root_safety=engine.root_safety,heuristic_mix=engine.heuristic_mix,fpu_mode=engine.fpu_mode,policy_temperature=engine.policy_temperature)
            paths=info['paths'];policy=info['policy']
            if explore and g.ply<24:
                path=paths[int(rng.choice(len(paths),p=policy))]
        else:
            path,_=engine.think(g.board,p,g.ply,g.repetitions)
            paths=[path for _,path in moves]
            policy=[float(move[0]==path[0] and move[-1]==path[-1]) for move in paths]
        if path is None or not r.valid_path(g.board,p,path): raise RuntimeError('invalid engine move')
        records.append(dict(board=g.board.copy(),player=p,ply=g.ply,path=path))
        if policy is not None:
            canonical=isinstance(engine,Neural121) and engine.model.canonical
            encoding=engine.model.encoding if isinstance(engine,Neural121) else ENCODING
            samples.append(dict(x=encode(g.board,p,g.ply,g.repetitions,canonical=canonical),player=p,encoding=encoding,
                                actions=[action_id(path[0],path[-1],p,canonical) for path in paths],policy=policy))
        g.play(path)
    winner,reason=g.result()
    for sample in samples:
        sample['z']=0. if winner==0 else 1. if winner==sample['player'] else -1.
        sample['value_weight']=0. if reason=='max_ply' else 1.
    return dict(seed=seed,source=source,winner=winner,reason=reason,plies=g.ply,steps=records),samples


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--bootstrap-games',type=int,default=4)
    ap.add_argument('--selfplay-games',type=int,default=2)
    ap.add_argument('--simulations',type=int,default=8)
    ap.add_argument('--steps',type=int,default=60)
    ap.add_argument('--seed',type=int,default=20261029)
    args=ap.parse_args()
    if args.bootstrap_games<2 or args.selfplay_games<1 or args.simulations<1 or args.steps<1: ap.error('invalid trial limits')
    args.out.mkdir(parents=True,exist_ok=False)
    lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    start=time.monotonic();status={}
    def update(stage,**extra):
        status.update(stage=stage,elapsed=time.monotonic()-start,**extra)
        temp=args.out/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(args.out/'status.json')
    config=dict(format=FORMAT,encoding=ENCODING,profile=r.PROFILE,max_ply=240,
                seed=args.seed,bootstrap_games=args.bootstrap_games,selfplay_games=args.selfplay_games,
                simulations=args.simulations,steps=args.steps,bootstrap_seconds=.03,
                eval_seed=args.seed+10000,eval_pairs=2,automatic_promotion=False,
                rule_draw_labels='true terminal labels; max_ply value_weight=0; no invented winners',
                source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
                               for name in ['az121.py','train121_trial.py','rules121.py','engine_c121.py','ccore121.c']})
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    all_games=[]
    def save_game(record):
        all_games.append(record)
        with (args.out/'games.jsonl').open('a') as file: file.write(json.dumps(record)+'\n')
        print(json.dumps({k:record[k] for k in ['source','seed','winner','reason','plies']}),flush=True)
    try:
        update('bootstrap');model=Net121();model.save(args.out/'initial_random.pt')
        teacher=CSearch121(seconds=.03,max_depth=8);bootstrap=[];heldout=[]
        for i in range(args.bootstrap_games):
            record,samples=play({1:teacher,2:teacher},args.seed+i,opening=4,source='C-bootstrap')
            save_game(record)
            if i==args.bootstrap_games-1: heldout=samples
            else: bootstrap.extend(samples)
            update('bootstrap',completed=i+1)
        update('bootstrap_train',samples=len(bootstrap),heldout_samples=len(heldout))
        def heldout_loss(model):
            with torch.no_grad():
                losses=[];policy_losses=[];value_losses=[]
                for s in heldout:
                    logits,value=model(torch.from_numpy(s['x']).unsqueeze(0))
                    ce=-(logits[0,s['actions']].log_softmax(0)*torch.tensor(s['policy'])).sum()
                    mse=float((value[0]-s['z'])**2)
                    policy_losses.append(float(ce));value_losses.append(mse)
                    losses.append(float(ce)+mse*s['value_weight'])
                return dict(total=float(np.mean(losses)),policy_ce=float(np.mean(policy_losses)),value_mse=float(np.mean(value_losses)))
        model.eval();before=heldout_loss(model)
        bootstrap_loss=train(model,bootstrap,args.steps,args.seed)
        after=heldout_loss(model);model.save(args.out/'bootstrap.pt')
        baseline=copy.deepcopy(model);samples=[]
        for i in range(args.selfplay_games):
            engine=Neural121(baseline,args.simulations)
            record,game_samples=play({1:engine,2:engine},args.seed+1000+i,explore=True,source='MCTS-selfplay')
            save_game(record);samples.extend(game_samples);update('selfplay',completed=i+1)
        torch.save(dict(format=FORMAT,encoding=ENCODING,profile=r.PROFILE,
                        samples=[{**s,'x':torch.from_numpy(s['x'])} for s in samples]),args.out/'selfplay_samples.pt')
        update('selfplay_train',samples=len(samples));candidate=copy.deepcopy(baseline)
        selfplay_loss=train(candidate,samples,args.steps,args.seed+1);candidate.save(args.out/'candidate.pt')
        evaluation=[]
        for pair in range(2):
            for side in (1,2):
                engines={side:Neural121(candidate,args.simulations),3-side:Neural121(baseline,args.simulations)}
                record,_=play(engines,args.seed+10000+pair,opening=4,source='paired-evaluation')
                record['candidate_side']=side;save_game(record);evaluation.append(record)
                update('evaluation',completed=len(evaluation))
        result=dict(bootstrap_games=args.bootstrap_games,bootstrap_samples=len(bootstrap),
                    heldout_samples=len(heldout),heldout_loss_before=before,heldout_loss_after=after,
                    bootstrap_train=bootstrap_loss,selfplay_train=selfplay_loss,selfplay_samples=len(samples),
                    selfplay_value_supervised=sum(s['value_weight']>0 for s in samples),
                    selfplay_outcomes=[{k:g[k] for k in ['winner','reason','plies']} for g in all_games if g['source']=='MCTS-selfplay'],
                    candidate_wins=sum(g['winner']==g['candidate_side'] for g in evaluation),
                    baseline_wins=sum(g['winner']==3-g['candidate_side'] for g in evaluation),
                    draws=sum(g['winner']==0 for g in evaluation),promoted=False,
                    conclusion='Pipeline smoke test only; four games cannot establish improvement.',seconds=time.monotonic()-start)
        (args.out/'result.json').write_text(json.dumps(result,indent=2));update('complete',result=result)
        print(json.dumps(result),flush=True)
    except BaseException:
        update('failed',error=traceback.format_exc());raise

if __name__=='__main__': main()
