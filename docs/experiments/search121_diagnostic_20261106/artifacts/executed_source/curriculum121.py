"""Bounded teacher/endgame curriculum with opening-group splits and locked output.
Run as a standalone process; progress and results are written without Codex.
"""
import argparse
from collections import Counter
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
import rules121 as r
from az121 import Net121, Neural121, encode, FORMAT, ENCODING
from engine_c121 import CSearch121


def atomic(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def opening(seed,length):
    rng=random.Random(seed);g=r.Game();paths=[]
    for _ in range(length):
        if g.result()[0] is not None: break
        path=rng.choice(r.get_all_moves(g.board,g.player))[1];g.play(path);paths.append(path)
    return paths


def replay(paths):
    g=r.Game()
    for path in paths: g.play(path)
    return g


def finish(g,engines):
    start=g.ply;paths=[]
    while g.result()[0] is None:
        path,_=engines[g.player].think(g.board,g.player,g.ply,g.repetitions)
        if path is None: raise RuntimeError('engine returned no move in a live position')
        g.play(path);paths.append(path)
    winner,reason=g.result()
    return dict(winner=winner,reason=reason,start_ply=start,plies=g.ply,paths=paths)


def samples_from_game(record):
    # No value training from max-ply draws or invented progress scores.
    if record['winner'] not in (1,2): return []
    g=replay(record['opening']);samples=[]
    for i,path in enumerate(record['paths']):
        moves=r.get_all_moves(g.board,g.player)
        actions=[s*121+p[-1] for s,p in moves];target=path[0]*121+path[-1]
        if target not in actions: raise ValueError('corrupt teacher game')
        samples.append(dict(x=encode(g.board,g.player,g.ply,g.repetitions),
                            actions=actions,target=actions.index(target),
                            z=1. if record['winner']==g.player else -1.,
                            endgame=len(record['paths'])-i<=16,
                            group=record['group'],winner=record['winner']))
        g.play(path)
    if g.result()!=(record['winner'],record['reason']): raise ValueError('incorrect terminal label')
    return samples


def batch(samples):
    xs=torch.from_numpy(np.stack([s['x'] for s in samples]));count=max(len(s['actions']) for s in samples)
    ids=torch.zeros((len(samples),count),dtype=torch.long);mask=torch.zeros_like(ids,dtype=torch.bool)
    for i,s in enumerate(samples): ids[i,:len(s['actions'])]=torch.tensor(s['actions']);mask[i,:len(s['actions'])]=True
    return xs,ids,mask,torch.tensor([s['target'] for s in samples]),torch.tensor([s['z'] for s in samples],dtype=torch.float32)


@torch.no_grad()
def metrics(model,samples):
    if not samples: raise ValueError('empty held-out split')
    model.eval();ce=mse=policy_hits=value_hits=0.;total=0
    for start in range(0,len(samples),32):
        xs,ids,mask,target,z=batch(samples[start:start+32]);logits,value=model(xs)
        logits=logits.gather(1,ids).masked_fill(~mask,-1e9)
        ce+=float(F.cross_entropy(logits,target,reduction='sum'));mse+=float((value-z).square().sum())
        policy_hits+=float((logits.argmax(1)==target).sum());value_hits+=float(((value>=0)==(z>0)).sum());total+=len(z)
    return dict(samples=total,policy_ce=ce/total,value_mse=mse/total,policy_accuracy=policy_hits/total,value_accuracy=value_hits/total)


def split_groups(groups,seed):
    ids=list(range(groups));random.Random(seed).shuffle(ids)
    ntest=max(1,groups//5);nval=max(1,groups//10)
    return {'test':set(ids[:ntest]),'validation':set(ids[ntest:ntest+nval]),'train':set(ids[ntest+nval:])}


def train_curriculum(model,train_samples,val_samples,steps,seed,out,update):
    rng=np.random.default_rng(seed);torch.manual_seed(seed)
    endings=[s for s in train_samples if s['endgame']]
    if not endings: raise ValueError('no genuine terminal endgames')
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    best=float('inf');history=[]
    for step in range(steps):
        # First practice closing; then retain half endgame and half full games.
        chosen=[endings[i] for i in rng.integers(len(endings),size=16 if step<steps//5 else 8)]
        if step>=steps//5: chosen += [train_samples[i] for i in rng.integers(len(train_samples),size=8)]
        xs,ids,mask,target,z=batch(chosen);model.train();logits,value=model(xs)
        loss=F.cross_entropy(logits.gather(1,ids).masked_fill(~mask,-1e9),target)+F.mse_loss(value,z)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if (step+1)%50==0 or step==steps-1:
            score=metrics(model,val_samples);criterion=score['policy_ce']+score['value_mse']
            row=dict(step=step+1,train_loss=float(loss.detach()),validation=score);history.append(row)
            if criterion<best:
                best=criterion;model.save(out/'candidate.pt')
            atomic(out/'training.json',history);update('training',step=step+1,validation=score)
            print(json.dumps(row),flush=True)
    return Net121.load(out/'candidate.pt')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--games',type=int,default=100);ap.add_argument('--steps',type=int,default=600)
    ap.add_argument('--seed',type=int,default=291029);ap.add_argument('--teacher-seconds',type=float,default=.02)
    ap.add_argument('--eval-games',type=int,default=20);ap.add_argument('--simulations',type=int,default=16)
    args=ap.parse_args()
    if args.games<20 or args.games%2 or args.steps<1 or args.eval_games<2 or args.eval_games%2: ap.error('even games >=20 and even eval-games >=2 required')
    args.out.mkdir(parents=True,exist_ok=False)
    lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1);torch.manual_seed(args.seed);started=time.monotonic()
    def update(stage,**extra):
        atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-started,**extra))
    config={**vars(args),'out':str(args.out),'format':FORMAT,'encoding':ENCODING,'profile':r.PROFILE,
            'max_ply':240,'automatic_promotion':False,'value_labels':'true decisive teacher games only',
            'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['curriculum121.py','az121.py','rules121.py','ccore121.c']}}
    atomic(args.out/'config.json',config)
    snapshot=args.out/'source';snapshot.mkdir()
    for name in config['source_sha256']:
        (snapshot/name).write_bytes(Path(name).read_bytes())
    try:
        splits=split_groups(args.games//2,args.seed);atomic(args.out/'split.json',{k:sorted(v) for k,v in splits.items()})
        records=[];teacher=CSearch121(seconds=args.teacher_seconds,max_depth=10)
        stronger=CSearch121(seconds=args.teacher_seconds*2,max_depth=12)
        for group in range(args.games//2):
            prefix=opening(args.seed+group,4+(group%5)*4)
            for side in (1,2):
                update('generation',completed=len(records),total=args.games)
                result=finish(replay(prefix),{side:stronger,3-side:teacher})
                record=dict(group=group,opening=prefix,stronger_side=side,**result);records.append(record)
                with (args.out/'teacher_games.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps({k:record[k] for k in ['group','stronger_side','winner','reason','plies']}),flush=True)
        summary={k:dict(games=sum(g['group'] in ids for g in records),
                        wins=dict(Counter(g['winner'] for g in records if g['group'] in ids)),
                        reasons=dict(Counter(g['reason'] for g in records if g['group'] in ids))) for k,ids in splits.items()}
        atomic(args.out/'data_summary.json',summary)
        sets={key:[sample for game in records if game['group'] in ids for sample in samples_from_game(game)] for key,ids in splits.items()}
        if any(not values for values in sets.values()): raise ValueError('split has no decisive games; do not train')
        # Validation/test groups are never sampled by the optimizer.
        model=Net121();model.save(args.out/'random_baseline.pt');before=metrics(model,sets['test'])
        candidate=train_curriculum(model,sets['train'],sets['validation'],args.steps,args.seed,args.out,update)
        after=metrics(candidate,sets['test'])
        endgames=[s for s in sets['test'] if s['endgame']]
        heldout=dict(before=before,after=after,endgame=metrics(candidate,endgames))
        atomic(args.out/'heldout.json',heldout)
        prior_path=Path('nn_checkpoints/experiments/az121_trial_20261029_b/bootstrap.pt')
        prior=Net121.load(prior_path)
        heldout['prior_bootstrap']=metrics(prior,sets['test'])
        atomic(args.out/'heldout.json',heldout)
        atomic(args.out/'eval_baseline.json',dict(path=str(prior_path),sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest()))
        evaluation=[]
        for pair in range(args.eval_games//2):
            prefix=opening(args.seed+100000+pair,4+(pair%5)*2)
            for side in (1,2):
                update('evaluation',completed=len(evaluation),total=args.eval_games)
                engines={side:Neural121(candidate,args.simulations),3-side:Neural121(prior,args.simulations)}
                row=dict(pair=pair,candidate_side=side,opening=prefix,**finish(replay(prefix),engines));evaluation.append(row)
                with (args.out/'evaluation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps({k:row[k] for k in ['pair','candidate_side','winner','reason','plies']}),flush=True)
        # Real play from held-out near-finish states, retaining the entire prefix history.
        ending_results=[]
        for game in [g for g in records if g['group'] in splits['test'] and g['winner'] in (1,2)][:10]:
            prefix=game['opening']+game['paths'][:-8]
            for name,network in [('prior',prior),('candidate',candidate)]:
                g=replay(prefix);side=game['winner']
                result=finish(g,{side:Neural121(network,args.simulations),3-side:teacher})
                row=dict(model=name,side=side,group=game['group'],**result);ending_results.append(row)
                update('endgame_evaluation',completed=len(ending_results),total=20)
        atomic(args.out/'endgame_evaluation.json',ending_results)
        wins=sum(g['winner']==g['candidate_side'] for g in evaluation);losses=sum(g['winner']==3-g['candidate_side'] for g in evaluation)
        result=dict(teacher_games=len(records),data=summary,samples={k:len(v) for k,v in sets.items()},heldout=heldout,
                    candidate_wins=wins,prior_wins=losses,draws=len(evaluation)-wins-losses,
                    evaluation_reasons=dict(Counter(g['reason'] for g in evaluation)),
                    endgame={name:dict(wins=sum(g['winner']==g['side'] for g in ending_results if g['model']==name),
                                      games=sum(g['model']==name for g in ending_results)) for name in ['prior','candidate']},
                    promoted=False,seconds=time.monotonic()-started,
                    note='Teacher imitation/endgame curriculum, not pure self-play; pilot evaluation, no rating claim.')
        atomic(args.out/'result.json',result);update('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        update('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
