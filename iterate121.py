"""Bounded independent search ablations and confirmation; never blind retraining."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import torch
from az121 import Net121, Neural121
from curriculum121 import atomic,opening,replay,finish
from diagnostics.diagnose121 import policy
from engine_c121 import CSearch121
import rules121 as r

ARMS={'policy':dict(simulations=0,value_scale=1.),'mcts16':dict(simulations=16,value_scale=1.),
      'mcts64':dict(simulations=64,value_scale=1.),'mcts64_zero':dict(simulations=64,value_scale=0.)}
class PolicyEngine:
    def __init__(self,model):self.model=model
    def think(self,board,player,ply=0,repetitions=None):
        g=r.Game(board=board,player=player,ply=ply);g.repetitions=Counter(repetitions or {(tuple(board),player):1})
        path,value=policy(self.model,g);return path,dict(value=value)

def engine(model,name):
    return PolicyEngine(model) if name=='policy' else Neural121(model,**ARMS[name])

def summary(rows):
    w=sum(x['winner']==x['side'] for x in rows);l=sum(x['winner']==3-x['side'] for x in rows)
    return dict(games=len(rows),wins=w,losses=l,draws=len(rows)-w-l,
                points=w+.5*(len(rows)-w-l),reasons=dict(Counter(x['reason'] for x in rows)))

def paired_gate(rows):
    groups={}
    for x in rows:groups.setdefault(x['pair'],[]).append(x)
    positive=negative=ties=0
    for pair in groups.values():
        if len(pair)!=2:raise ValueError('incomplete evaluation pair')
        points=sum(1 if x['winner']==x['side'] else .5 if x['winner']==0 else 0 for x in pair)
        positive+=points>1;negative+=points<1;ties+=points==1
    n=positive+negative
    p=sum(math.comb(n,k) for k in range(positive,n+1))/2**n if n else 1.
    return dict(positive_pairs=positive,negative_pairs=negative,tied_pairs=ties,one_sided_p=p,
                passed=positive>negative and p<=.05)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True);ap.add_argument('--seed',type=int,default=301030)
    args=ap.parse_args();torch.set_num_threads(1);model=Net121.load(args.model)
    args.out.mkdir(parents=True,exist_ok=False)
    lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    began=time.monotonic()
    def update(stage,**data):atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-began,**data))
    config=dict(model=str(args.model.resolve()),model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),seed=args.seed,
                arms=ARMS,profile=r.PROFILE,scout_games_per_arm=12,confirmation_games=40,c_games=20,c_seconds=.03,
                rule='Select on scouting only; confirm once on fresh opening groups. No GUI overwrite.',
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['iterate121.py','az121.py','rules121.py','curriculum121.py','engine_c121.py']})
    atomic(args.out/'config.json',config)
    snapshot=args.out/'source';snapshot.mkdir()
    for name in config['source_sha256']:(snapshot/name).write_bytes(Path(name).read_bytes())
    def record(stage,row):
        with (args.out/(stage+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
    def matches(stage,arm,pairs,offset,opponent='c'):
        rows=[]
        for pair in range(pairs):
            prefix=opening(args.seed+offset+pair,0 if pair==0 else 4+(pair%4)*2)
            for side in (1,2):
                update(stage,arm=arm,completed=len(rows),total=2*pairs)
                other=CSearch121(seconds=.03,max_depth=10) if opponent=='c' else engine(model,opponent)
                start=time.monotonic();row=finish(replay(prefix),{side:engine(model,arm),3-side:other})
                row.update(arm=arm,pair=pair,side=side,opening=prefix,seconds=time.monotonic()-start)
                rows.append(row);record(stage,row)
                print(json.dumps(dict(stage=stage,arm=arm,**summary(rows))),flush=True)
        return rows
    try:
        scouting={}
        for arm in ARMS:
            rows=matches('scouting',arm,6,0);scouting[arm]=summary(rows)
            atomic(args.out/'scouting.json',scouting)
        # Freeze the hypothesis before touching confirmation games.
        challenger=max((a for a in ARMS if a!='mcts16'),key=lambda a:scouting[a]['points'])
        atomic(args.out/'selection.json',dict(challenger=challenger,scouting=scouting))
        confirmation=matches('confirmation',challenger,20,10000,opponent='mcts16')
        gate=paired_gate(confirmation)
        control_c=matches('control_c','mcts16',10,20000)
        challenger_c=matches('challenger_c',challenger,10,20000)
        control=summary(control_c);challenger_score=summary(challenger_c)
        result=dict(scouting=scouting,challenger=challenger,confirmation=summary(confirmation),paired_gate=gate,
                    control_vs_c=control,challenger_vs_c=challenger_score,
                    qualified=gate['passed'] and challenger_score['points']>=control['points'],
                    seconds=time.monotonic()-began,promoted=False,
                    note='Only search settings varied; weights unchanged. Unequal simulation/time budgets are explicit.')
        atomic(args.out/'result.json',result);update('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        update('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
