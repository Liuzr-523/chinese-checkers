"""Train on teacher continuations of neural failure states, with replay retention.
All labels come from actual strict terminal games. Independent match confirmation.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback
import torch
from az121 import Net121,Neural121
from curriculum121 import atomic,replay,finish,samples_from_game,train_curriculum,metrics,opening
from engine_c121 import CSearch121
from iterate121 import summary,paired_gate
import rules121 as r


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=301031);args=ap.parse_args()
    torch.set_num_threads(1);torch.manual_seed(args.seed);rng=random.Random(args.seed)
    source=Path('nn_checkpoints/experiments/curriculum121_100_20261029')
    loss_source=Path('nn_checkpoints/experiments/selfplay121_round1_20261029/candidate_vs_c.jsonl')
    base=Net121.load(source/'candidate.pt');args.out.mkdir(parents=True,exist_ok=False)
    lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    began=time.monotonic()
    def update(stage,**extra):atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-began,**extra))
    def record(stage,row):
        with (args.out/(stage+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
    config=dict(seed=args.seed,profile=r.PROFILE,model=str(source/'candidate.pt'),
                model_sha256=hashlib.sha256((source/'candidate.pt').read_bytes()).hexdigest(),
                loss_source=str(loss_source),loss_source_sha256=hashlib.sha256(loss_source.read_bytes()).hexdigest(),
                kind='C teacher correction plus original teacher retention; not pure selfplay',
                teacher_seconds=.02,teacher_stronger_seconds=.04,steps=600,confirmation_games=60,c_games=20,
                simulations=32,automatic_promotion=False,
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['correct121.py','curriculum121.py','az121.py','rules121.py','iterate121.py']})
    atomic(args.out/'config.json',config);snapshot=args.out/'source';snapshot.mkdir()
    for name in config['source_sha256']:(snapshot/name).write_bytes(Path(name).read_bytes())
    try:
        losses=[json.loads(x) for x in loss_source.read_text().splitlines()]
        groups=sorted({g.get('pair',i//2) for i,g in enumerate(losses)});rng.shuffle(groups)
        val_groups=set(groups[:max(1,len(groups)//5)])
        val_ids={i for i,g in enumerate(losses) if g.get('pair',i//2) in val_groups}
        atomic(args.out/'correction_split.json',dict(validation=sorted(val_ids),train=sorted(set(range(len(losses)))-val_ids),validation_pairs=sorted(val_groups)))
        new_train=[];new_val=[];completed=0;reasons=Counter()
        total=sum(2*len(set([min(len(g['steps'])-1,8),len(g['steps'])//2,max(0,len(g['steps'])-4)])) for g in losses)
        for index,game in enumerate(losses):
            points=sorted(set([min(len(game['steps'])-1,8),len(game['steps'])//2,max(0,len(game['steps'])-4)]))
            for ply in points:
                prefix=[s['path'] for s in game['steps'][:ply]]
                for strong_side in (1,2):
                    update('correction_generation',completed=completed,total=total)
                    engines={strong_side:CSearch121(seconds=.04,max_depth=12),3-strong_side:CSearch121(seconds=.02,max_depth=10)}
                    result=finish(replay(prefix),engines)
                    row=dict(group=index,source_pair=game.get('pair',index//2),opening=prefix,stronger_side=strong_side,source_ply=ply,**result)
                    record('corrections',row);completed+=1;reasons[row['reason']]+=1
                    samples=samples_from_game(row)
                    (new_val if index in val_ids else new_train).extend(samples)
                    print(json.dumps(dict(stage='correction',completed=completed,total=total,winner=row['winner'],reason=row['reason'])),flush=True)
        splits=json.loads((source/'split.json').read_text());old_train=[];old_val=[]
        for line in (source/'teacher_games.jsonl').read_text().splitlines():
            game=json.loads(line)
            if game['group'] in splits['train']:old_train.extend(samples_from_game(game))
            elif game['group'] in splits['validation']:old_val.extend(samples_from_game(game))
        if not new_train or not new_val:raise ValueError('No decisive correction examples')
        # Equal aggregate representation for correction and original data.
        n=max(len(new_train),len(old_train))
        mixed=[rng.choice(new_train) for _ in range(n)]+[rng.choice(old_train) for _ in range(n)]
        validation=new_val+old_val
        before=dict(correction=metrics(base,new_val),retention=metrics(base,old_val))
        atomic(args.out/'data.json',dict(correction_games=completed,reasons=dict(reasons),
               new_train=len(new_train),new_validation=len(new_val),old_train=len(old_train),old_validation=len(old_val),mixed=len(mixed)))
        candidate=train_curriculum(base,mixed,validation,600,args.seed,args.out,update)
        atomic(args.out/'validation.json',dict(before=before,after=dict(correction=metrics(candidate,new_val),retention=metrics(candidate,old_val))))
        baseline=Net121.load(source/'candidate.pt');matches=[]
        for pair in range(30):
            prefix=opening(args.seed+100000+pair,0 if pair==0 else 4+(pair%5)*2)
            for side in (1,2):
                update('independent_confirmation',completed=len(matches),total=60)
                row=finish(replay(prefix),{side:Neural121(candidate,32),3-side:Neural121(baseline,32)})
                row.update(pair=pair,side=side,opening=prefix);matches.append(row);record('confirmation',row)
                print(json.dumps(dict(stage='confirmation',**summary(matches))),flush=True)
        fixed={}
        for name,model in [('control',baseline),('candidate',candidate)]:
            rows=[]
            for pair in range(10):
                prefix=opening(args.seed+200000+pair,0 if pair==0 else 4+(pair%4)*2)
                for side in (1,2):
                    update('fixed_c_'+name,completed=len(rows),total=20)
                    row=finish(replay(prefix),{side:Neural121(model,32),3-side:CSearch121(seconds=.03,max_depth=10)})
                    row.update(pair=pair,side=side,opening=prefix);rows.append(row);record('fixed_c_'+name,row)
            fixed[name]=summary(rows)
        gate=paired_gate(matches)
        result=dict(confirmation=summary(matches),paired_gate=gate,fixed_c=fixed,
                    qualified=gate['passed'] and fixed['candidate']['points']>=fixed['control']['points'],
                    promoted=False,seconds=time.monotonic()-began,note='Counterexample teacher fine-tuning with retention, not pure selfplay.')
        atomic(args.out/'result.json',result);update('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        update('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
