"""Before more policy training, test whether the MCTS64 supervisor beats MCTS32.
Frozen registered CNN-v2, pure learned values, no training or model promotion.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import torch
from az121 import Net121,Neural121
from curriculum121 import atomic,opening,replay
from iterate121 import summary,paired_gate
import rules121 as r


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=301039);ap.add_argument('--games',type=int,default=100)
    args=ap.parse_args()
    if args.games<20 or args.games%2:ap.error('even game count >=20 required')
    torch.set_num_threads(1);source=Path('nn_checkpoints/models/checkpoints/neural121_v2_canonical.pt');model=Net121.load(source)
    args.out.mkdir(parents=True,exist_ok=False);lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic()
    def update(stage,**data):atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-start,**data))
    config=dict(model=str(source.resolve()),model_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),profile=r.PROFILE,
                encoding=model.encoding,seed=args.seed,games=args.games,teacher_simulations=64,student_simulations=32,
                threshold_p=.01,scope='Check selfplay search supervisor strength; weights frozen, no training or promotion',
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['budget121_teacher_check.py','az121.py','rules121.py','curriculum121.py','iterate121.py']})
    atomic(args.out/'config.json',config);snapshot=args.out/'source';snapshot.mkdir()
    for name in config['source_sha256']:(snapshot/name).write_bytes(Path(name).read_bytes())
    rows=[];timing={key:dict(seconds=0.,moves=0,simulations_per_move=sims) for key,sims in [('teacher',64),('student',32)]}
    try:
        for pair in range(args.games//2):
            prefix=opening(args.seed+100000+pair,0 if pair==0 else 4+(pair%5)*2)
            for side in (1,2):
                update('teacher_strength_confirmation',completed=len(rows),total=args.games,score=summary(rows))
                engines={side:Neural121(model,64),3-side:Neural121(model,32)};g=replay(prefix);moves=[];began=time.monotonic()
                while g.result()[0] is None:
                    key='teacher' if g.player==side else 'student';t=time.monotonic()
                    path,info=engines[g.player].think(g.board,g.player,g.ply,g.repetitions)
                    timing[key]['seconds']+=time.monotonic()-t;timing[key]['moves']+=1
                    if info['root_visits']!=timing[key]['simulations_per_move']:raise ValueError('wrong simulation budget')
                    if path is None:raise ValueError('no move in a live position')
                    g.play(path);moves.append(path)
                winner,reason=g.result();row=dict(pair=pair,side=side,opening=prefix,paths=moves,start_ply=len(prefix),plies=g.ply,
                                                  winner=winner,reason=reason,seconds=time.monotonic()-began)
                rows.append(row)
                with (args.out/'confirmation.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
                print(json.dumps(dict(completed=len(rows),**summary(rows))),flush=True)
        gate=paired_gate(rows)
        result=dict(teacher_vs_student=summary(rows),paired_gate=gate,
                    stronger_teacher_confirmed=gate['passed'] and gate['one_sided_p']<=.01,
                    timing=timing,qualified=False,promoted=False,seconds=time.monotonic()-start,
                    note='Larger search budget is not an equal-time improvement. No checkpoint was updated.')
        atomic(args.out/'result.json',result);update('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        update('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
