"""New neural selfplay only: low-rate fine-tuning with a frozen-policy KL control."""
import argparse
from collections import Counter,defaultdict
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
from az121 import Net121,Neural121
from train121_trial import play
from curriculum121 import atomic,opening,replay,finish
from engine_c121 import CSearch121
from iterate121 import summary,paired_gate
import rules121 as r


def batch(samples):
    maximum=max(len(s['actions']) for s in samples);n=len(samples)
    xs=torch.from_numpy(np.stack([s['x'] for s in samples]));ids=torch.zeros((n,maximum),dtype=torch.long)
    mask=torch.zeros_like(ids,dtype=torch.bool);pi=torch.zeros((n,maximum))
    for i,s in enumerate(samples):
        k=len(s['actions']);ids[i,:k]=torch.tensor(s['actions']);mask[i,:k]=True;pi[i,:k]=torch.tensor(s['policy'])
    return xs,ids,mask,pi,torch.tensor([s['z'] for s in samples]),torch.tensor([s['value_weight'] for s in samples])


@torch.no_grad()
def metrics(model,samples):
    model.eval();ce=mse=weight=0.;n=0
    for start in range(0,len(samples),32):
        xs,ids,mask,pi,z,w=batch(samples[start:start+32]);logits,value=model(xs)
        logp=logits.gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)
        ce+=float(-(pi*logp).sum());mse+=float(((value-z).square()*w).sum());weight+=float(w.sum());n+=len(z)
    return dict(policy_ce=ce/max(1,n),value_mse=mse/max(1,weight),samples=n,value_samples=weight)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--games',type=int,default=100);ap.add_argument('--steps',type=int,default=400);ap.add_argument('--seed',type=int,default=301038)
    args=ap.parse_args();torch.set_num_threads(1);torch.manual_seed(args.seed)
    if args.games<20 or args.steps<1:ap.error('invalid budgets')
    baseline=Net121.load(args.model);args.out.mkdir(parents=True,exist_ok=False)
    lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);began=time.monotonic()
    def update(stage,**extra):atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-began,**extra))
    def record(stage,row):
        with (args.out/(stage+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
    config=dict(model=str(args.model.resolve()),model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                seed=args.seed,games=args.games,steps=args.steps,profile=r.PROFILE,encoding=baseline.encoding,
                selfplay_simulations=64,simulations=32,confirmation_games=60,independent_gate_p=.01,
                training='Only new frozen neural model selfplay; low-rate update with/without old neural policy KL; no teacher replay or handcrafted value',
                lr=3e-5,kl_controls=[0.,.3],max_ply_value_weight=0,minimum_decisive_train_games=20,minimum_games_per_winner=5,
                automatic_promotion=False,source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['selfplay121_retained.py','az121.py','train121_trial.py','rules121.py','curriculum121.py','iterate121.py']})
    atomic(args.out/'config.json',config);snapshot=args.out/'source';snapshot.mkdir();baseline.save(args.out/'baseline.pt')
    for name in config['source_sha256']:(snapshot/name).write_bytes(Path(name).read_bytes())
    try:
        shards=args.out/'selfplay';shards.mkdir();games=[];samples=[]
        for i in range(args.games):
            update('selfplay',completed=i,total=args.games)
            engine=Neural121(baseline,64)
            game,part=play({1:engine,2:engine},args.seed+i,explore=True,opening=4,source='pure-neural-selfplay')
            prefix=[s['path'] for s in game['steps'][:4]];g=replay(prefix)
            key=hashlib.sha256(json.dumps([g.board,g.player]).encode()).hexdigest()
            game.update(game_id=i,opening_group=key);record('selfplay_games',game)
            games.append({k:game[k] for k in ['game_id','opening_group','winner','reason']})
            for s in part:s.update(game_id=i,opening_group=key,winner=game['winner'])
            if any(s['encoding']!=baseline.encoding for s in part):raise ValueError('selfplay encoding mismatch')
            torch.save(dict(encoding=baseline.encoding,profile=r.PROFILE,samples=[{**s,'x':torch.from_numpy(s['x'])} for s in part]),shards/f'game_{i:03d}.pt')
            samples.extend(part);print(json.dumps(dict(stage='selfplay',completed=i+1,winner=game['winner'],reason=game['reason'])),flush=True)
        groups=sorted({g['opening_group'] for g in games});random.Random(args.seed).shuffle(groups)
        val_groups=set(groups[:max(1,len(groups)//5)])
        training=[s for s in samples if s['opening_group'] not in val_groups];validation=[s for s in samples if s['opening_group'] in val_groups]
        winners=Counter(g['winner'] for g in games if g['opening_group'] not in val_groups)
        atomic(args.out/'split.json',dict(validation_groups=sorted(val_groups),train_games=[g['game_id'] for g in games if g['opening_group'] not in val_groups]))
        atomic(args.out/'data.json',dict(games=len(games),reasons=dict(Counter(g['reason'] for g in games)),train_samples=len(training),validation_samples=len(validation),train_winners=dict(winners)))
        if winners[1]<5 or winners[2]<5 or winners[1]+winners[2]<20:
            result=dict(qualified=False,promoted=False,training_skipped=True,reason='Insufficient genuine outcomes from both winners; do not train a first-player shortcut.',train_winners=dict(winners),seconds=time.monotonic()-began)
            atomic(args.out/'result.json',result);update('complete',result=result);return
        # Balance actual decisive winners, cap natural draws at one quarter; labels unchanged.
        pools={winner:[s for s in training if s['winner']==winner] for winner in (0,1,2)}
        baseline_score=metrics(baseline,validation);arms={};models={}
        for kl in config['kl_controls']:
            model=copy.deepcopy(baseline);model.requires_grad_(True);opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=1e-4)
            rng=np.random.default_rng(args.seed);best=float('inf');history=[];name='kl_'+str(kl)
            for step in range(args.steps):
                chosen=[];draws=4 if pools[0] else 0
                for winner,count in [(0,draws),(1,(16-draws)//2),(2,(16-draws)//2)]:
                    chosen.extend(pools[winner][int(rng.integers(len(pools[winner])))] for _ in range(count))
                xs,ids,mask,pi,z,w=batch(chosen);model.train();logits,value=model(xs)
                logp=logits.gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)
                with torch.no_grad():
                    old_logits,_=baseline(xs);old_logp=old_logits.gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1);old_pi=old_logp.exp()
                retain=(old_pi*(old_logp-logp)).sum(1).mean()
                loss=-(pi*logp).sum(1).mean()+((value-z).square()*w).sum()/w.sum().clamp_min(1)+kl*retain
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                if (step+1)%50==0 or step+1==args.steps:
                    score=metrics(model,validation);criterion=score['policy_ce']+score['value_mse']
                    history.append(dict(step=step+1,loss=float(loss.detach()),validation=score))
                    if criterion<best:best=criterion;model.save(args.out/(name+'.pt'))
                    atomic(args.out/(name+'_training.json'),history);update('training',arm=name,step=step+1)
            selected=Net121.load(args.out/(name+'.pt'));arms[name]=metrics(selected,validation);models[name]=selected
        name=min(arms,key=lambda k:arms[k]['policy_ce']+arms[k]['value_mse']);candidate=models[name];candidate.save(args.out/'candidate.pt')
        atomic(args.out/'selection.json',dict(arm=name,baseline=baseline_score,arms=arms))
        matches=[]
        for pair in range(30):
            prefix=opening(args.seed+100000+pair,0 if pair==0 else 4+(pair%5)*2)
            for side in (1,2):
                update('independent_confirmation',completed=len(matches),total=60)
                row=finish(replay(prefix),{side:Neural121(candidate,32),3-side:Neural121(baseline,32)})
                row.update(pair=pair,side=side,opening=prefix);matches.append(row);record('confirmation',row)
        fixed={}
        for label,model in [('control',baseline),('candidate',candidate)]:
            rows=[]
            for pair in range(10):
                prefix=opening(args.seed+200000+pair,0 if pair==0 else 4+(pair%4)*2)
                for side in (1,2):
                    update('fixed_c_'+label,completed=len(rows),total=20)
                    row=finish(replay(prefix),{side:Neural121(model,32),3-side:CSearch121(seconds=.03,max_depth=10)})
                    row.update(pair=pair,side=side,opening=prefix);rows.append(row);record('fixed_c_'+label,row)
            fixed[label]=summary(rows)
        gate=paired_gate(matches)
        result=dict(arm=name,selfplay_games=len(games),selfplay_reasons=dict(Counter(g['reason'] for g in games)),validation=arms,
                    confirmation=summary(matches),paired_gate=gate,fixed_c=fixed,
                    qualified=gate['passed'] and gate['one_sided_p']<=.01 and fixed['candidate']['points']>=fixed['control']['points'],
                    promoted=False,seconds=time.monotonic()-began,note='Only new neural selfplay used for this update; pretrained teacher ancestry remains.')
        atomic(args.out/'result.json',result);update('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        update('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
