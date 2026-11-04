"""Frozen policy + explicit hand-progress leaf guidance; not pure learned value.
Development scouting chooses the mixture; fresh opening groups confirm once.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import torch
from az121 import Net121,Neural121
from curriculum121 import atomic,opening,replay,finish
from engine_c121 import CSearch121
from iterate121 import summary,paired_gate
import rules121 as r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=301035)
    args=ap.parse_args();torch.set_num_threads(1)
    source=Path('nn_checkpoints/models/checkpoints/neural121_v1_correct.pt');model=Net121.load(source)
    args.out.mkdir(parents=True,exist_ok=False)
    lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);began=time.monotonic()
    def update(stage,**extra):atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-began,**extra))
    config=dict(model=str(source.resolve()),model_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                seed=args.seed,profile=r.PROFILE,simulations=32,confirmation_games=60,independent_gate_p=.01,
                change='Search only: frozen learned policy, tanh(C static score / 40) mixed into nonterminal leaf values',
                initial_mixtures=[.5,1.],heuristic_normalizer=40.,c_seconds=.03,automatic_promotion=False,
                budget_note='Same MCTS32 with extra C static evaluations; wall time logged, not equal-time comparison.',
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['hybrid121_trial.py','az121.py','rules121.py','curriculum121.py','iterate121.py','ccore121.c']})
    atomic(args.out/'config.json',config);shutil.copyfile(source,args.out/'candidate.pt');snapshot=args.out/'source';snapshot.mkdir()
    for name in config['source_sha256']:(snapshot/name).write_bytes(Path(name).read_bytes())
    def matches(stage,mix,pairs,offset,opponent='baseline'):
        rows=[]
        for pair in range(pairs):
            prefix=opening(args.seed+offset+pair,0 if pair==0 else 4+(pair%5)*2)
            for side in (1,2):
                update(stage,mixture=mix,completed=len(rows),total=2*pairs)
                other=Neural121(model,32) if opponent=='baseline' else CSearch121(seconds=.03,max_depth=10)
                started=time.monotonic();row=finish(replay(prefix),{side:Neural121(model,32,heuristic_mix=mix),3-side:other})
                row.update(pair=pair,side=side,opening=prefix,mixture=mix,seconds=time.monotonic()-started)
                rows.append(row)
                with (args.out/(stage+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(dict(stage=stage,mixture=mix,**summary(rows))),flush=True)
        return rows
    try:
        scouting={}
        for mix in config['initial_mixtures']:
            scouting[str(mix)]=summary(matches('scouting_'+str(mix),mix,6,0))
        chosen=max(config['initial_mixtures'],key=lambda m:scouting[str(m)]['points'])
        atomic(args.out/'selection.json',dict(scouting=scouting,heuristic_mix=chosen))
        config['heuristic_mix']=chosen;atomic(args.out/'config.json',config)
        confirmation=matches('confirmation',chosen,30,100000)
        control=matches('fixed_c_control',0.,10,200000,'c')
        candidate=matches('fixed_c_candidate',chosen,10,200000,'c')
        gate=paired_gate(confirmation)
        result=dict(scouting=scouting,heuristic_mix=chosen,confirmation=summary(confirmation),paired_gate=gate,
                    fixed_c=dict(control=summary(control),candidate=summary(candidate)),
                    qualified=gate['passed'] and gate['one_sided_p']<=.01 and summary(candidate)['points']>=summary(control)['points'],
                    promoted=False,seconds=time.monotonic()-began,
                    note='Hybrid hand-guided search, not neural training improvement or pure AlphaZero; terminal rewards unchanged.')
        atomic(args.out/'result.json',result);update('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        update('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
