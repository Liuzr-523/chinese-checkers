"""Continue a frozen 121-cell model with pure MCTS selfplay and fixed tests."""
import argparse
from collections import Counter
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import torch
from az121 import Net121,Neural121,train,FORMAT,ENCODING
from curriculum121 import atomic
from engine_c121 import CSearch121
from train121_trial import play
import rules121 as r


def score(rows):
    wins=sum(x['winner']==x['model_side'] for x in rows)
    losses=sum(x['winner']==3-x['model_side'] for x in rows)
    return dict(games=len(rows),wins=wins,losses=losses,draws=len(rows)-wins-losses,
                reasons=dict(Counter(x['reason'] for x in rows)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True);ap.add_argument('--games',type=int,default=100)
    ap.add_argument('--steps',type=int,default=600);ap.add_argument('--simulations',type=int,default=32)
    ap.add_argument('--eval-simulations',type=int,default=16);ap.add_argument('--eval-games',type=int,default=100)
    ap.add_argument('--c-games',type=int,default=20);ap.add_argument('--seed',type=int,default=291030)
    args=ap.parse_args()
    if min(args.games,args.steps,args.simulations,args.eval_simulations)<1 or args.eval_games<2 or args.eval_games%2 or args.c_games<2 or args.c_games%2:
        ap.error('invalid budgets; evaluation counts must be positive even numbers')
    # Refuse invalid weights before creating an output directory.
    torch.set_num_threads(1);torch.manual_seed(args.seed);base=Net121.load(args.model)
    args.out.mkdir(parents=True,exist_ok=False)
    lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic()
    def update(stage,**extra):
        atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-started,**extra))
    config={**vars(args),'out':str(args.out),'model':str(args.model.resolve()),'format':FORMAT,'encoding':base.encoding,
            'profile':r.PROFILE,'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'training_source':'new neural MCTS selfplay only, no teacher replay',
            'max_ply_value_weight':0,'c_seconds':.03,'automatic_promotion':False,
            'selfplay_seeds':[args.seed+i for i in range(args.games)],
            'evaluation_seeds':[args.seed+100000+i for i in range(args.eval_games//2)],
            'c_evaluation_seeds':[args.seed+200000+i for i in range(args.c_games//2)],
            'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['selfplay121_round.py','train121_trial.py','az121.py','curriculum121.py','rules121.py','ccore121.c']}}
    atomic(args.out/'config.json',config)
    folder=args.out/'source';folder.mkdir()
    for name in config['source_sha256']:(folder/name).write_bytes(Path(name).read_bytes())
    folder=args.out/'selfplay';folder.mkdir();base.save(args.out/'baseline.pt')
    def save(name,record):
        with (args.out/(name+'.jsonl')).open('a') as f:f.write(json.dumps(record)+'\n')
    def against_c(model,name):
        rows=[]
        for pair,seed in enumerate(config['c_evaluation_seeds']):
            for side in (1,2):
                update(name,completed=len(rows),total=args.c_games,score=score(rows))
                rec,_=play({side:Neural121(model,args.eval_simulations),3-side:CSearch121(seconds=.03,max_depth=10)},seed,opening=4,source=name)
                rec['model_side']=side;rec['pair']=pair;save(name,rec);rows.append(rec)
                print(json.dumps(dict(stage=name,completed=len(rows),**score(rows))),flush=True)
        atomic(args.out/(name+'.json'),score(rows));return score(rows)
    try:
        before=against_c(base,'baseline_vs_c')
        samples=[];outcomes=[]
        for i,seed in enumerate(config['selfplay_seeds']):
            update('selfplay',completed=i,total=args.games,reasons=dict(Counter(x['reason'] for x in outcomes)))
            engine=Neural121(base,args.simulations)
            rec,part=play({1:engine,2:engine},seed,explore=True,opening=4,source='pure-neural-selfplay')
            save('selfplay_games',rec);outcomes.append({k:rec[k] for k in ['seed','winner','reason']})
            torch.save(dict(format=FORMAT,encoding=base.encoding,profile=r.PROFILE,seed=seed,
                            samples=[{**s,'x':torch.from_numpy(s['x'])} for s in part]),folder/f'game_{i:03d}.pt')
            samples.extend(part)
            print(json.dumps(dict(stage='selfplay',completed=i+1,winner=rec['winner'],reason=rec['reason'],plies=len(rec['steps']))),flush=True)
        update('training',samples=len(samples),steps=args.steps)
        if not samples:raise ValueError('No policy samples; training skipped')
        candidate=copy.deepcopy(base);loss=train(candidate,samples,args.steps,args.seed)
        candidate.save(args.out/'candidate.pt');atomic(args.out/'training.json',loss)
        evaluation=[]
        for pair,seed in enumerate(config['evaluation_seeds']):
            for side in (1,2):
                update('candidate_vs_baseline',completed=len(evaluation),total=args.eval_games,score=score(evaluation))
                rec,_=play({side:Neural121(candidate,args.eval_simulations),3-side:Neural121(base,args.eval_simulations)},seed,opening=4,source='candidate-vs-baseline')
                rec.update(model_side=side,pair=pair);save('evaluation',rec);evaluation.append(rec)
                print(json.dumps(dict(stage='candidate_vs_baseline',completed=len(evaluation),**score(evaluation))),flush=True)
        after=against_c(candidate,'candidate_vs_c')
        result=dict(selfplay_games=len(outcomes),selfplay_reasons=dict(Counter(x['reason'] for x in outcomes)),
                    selfplay_decisive=sum(x['winner'] in (1,2) for x in outcomes),samples=len(samples),
                    value_supervised=sum(x['value_weight']>0 for x in samples),training=loss,
                    candidate_vs_baseline=score(evaluation),baseline_vs_c=before,candidate_vs_c=after,
                    promoted=False,seconds=time.monotonic()-started,
                    note='C is a fixed time-budget reference, neural-vs-neural uses equal simulation counts; no automatic promotion.')
        atomic(args.out/'result.json',result);update('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        update('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
