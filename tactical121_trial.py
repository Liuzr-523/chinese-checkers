"""Frozen NN-v1 with/without exact two-ply immediate-loss guard, fresh matches."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path
import time
import traceback
import torch
from az121 import Net121,Neural121
from curriculum121 import atomic,opening,replay,finish
from engine_c121 import CSearch121
from iterate121 import summary,paired_gate
import rules121 as r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=301034);ap.add_argument('--fpu-parent',action='store_true')
    args=ap.parse_args();torch.set_num_threads(1)
    source=Path('nn_checkpoints/models/checkpoints/neural121_v1_correct.pt');model=Net121.load(source)
    args.out.mkdir(parents=True,exist_ok=False)
    lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);began=time.monotonic()
    def update(stage,**extra):atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-began,**extra))
    def record(stage,row):
        with (args.out/(stage+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
    config=dict(model=str(source.resolve()),model_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                seed=args.seed,profile=r.PROFILE,simulations=32,confirmation_games=60,independent_gate_p=.01,
                change=('parent-value first-play baseline instead of zero; frozen pure neural weights and MCTS32' if args.fpu_parent else 'optional exact two-ply root tactical guard; weights/evaluation/legality unchanged'),
                root_safety=not args.fpu_parent,fpu_mode='parent' if args.fpu_parent else 'zero',c_seconds=.03,automatic_promotion=False,
                budget_note=('same MCTS32, changed unvisited-child baseline only; wall time recorded' if args.fpu_parent else 'same MCTS simulations plus explicit extra guard work; wall time recorded, not equal-time strength'),
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['tactical121_trial.py','az121.py','rules121.py','curriculum121.py','iterate121.py','ccore121.c']})
    atomic(args.out/'config.json',config);shutil.copyfile(source,args.out/'candidate.pt');snapshot=args.out/'source';snapshot.mkdir()
    for name in config['source_sha256']:(snapshot/name).write_bytes(Path(name).read_bytes())
    def matches(stage,guard,pairs,offset,opponent='c'):
        rows=[]
        for pair in range(pairs):
            prefix=opening(args.seed+offset+pair,0 if pair==0 else 4+(pair%5)*2)
            for side in (1,2):
                update(stage,completed=len(rows),total=2*pairs)
                other=CSearch121(seconds=.03,max_depth=10) if opponent=='c' else Neural121(model,32)
                start=time.monotonic();row=finish(replay(prefix),{side:Neural121(model,32,root_safety=guard and not args.fpu_parent,fpu_mode='parent' if guard and args.fpu_parent else 'zero'),3-side:other})
                row.update(pair=pair,side=side,opening=prefix,seconds=time.monotonic()-start);rows.append(row);record(stage,row)
                print(json.dumps(dict(stage=stage,**summary(rows))),flush=True)
        return rows
    try:
        confirmation=matches('confirmation',True,30,100000,'baseline')
        control=matches('fixed_c_control',False,10,200000)
        guarded=matches('fixed_c_candidate',True,10,200000)
        gate=paired_gate(confirmation)
        result=dict(confirmation=summary(confirmation),paired_gate=gate,
                    fixed_c=dict(control=summary(control),candidate=summary(guarded)),
                    qualified=gate['passed'] and gate['one_sided_p']<=.01 and summary(guarded)['points']>=summary(control)['points'],
                    promoted=False,seconds=time.monotonic()-began,
                    note=('Pure neural search configuration only, same weights and MCTS32; no training gain.' if args.fpu_parent else 'Search upgrade only, same network weights, extra root tactical work; not a training gain or equal-time comparison.'))
        atomic(args.out/'result.json',result);update('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        update('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
