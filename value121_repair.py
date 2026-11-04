"""Repair observed always-P1 value shortcut while freezing the entire policy.
Balanced pooled-head control versus spatial-head replacement; select only on
validation, then once on fresh paired matches. No fabricated training outcomes.
"""
import argparse
from collections import defaultdict
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from az121 import Net121,Neural121
from curriculum121 import atomic,samples_from_game,batch,metrics,opening,replay,finish
from engine_c121 import CSearch121
from iterate121 import summary,paired_gate
import rules121 as r


def make_spatial(base,normalized=False):
    model=Net121(base.width,value_head="spatial_norm" if normalized else "spatial",canonical=getattr(base,"canonical",False))
    state={k:v for k,v in base.state_dict().items() if not k.startswith('value.')}
    missing=model.load_state_dict(state,strict=False)
    if any(not name.startswith('value.') for name in missing.missing_keys):raise ValueError('incomplete policy copy')
    return model


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=600);ap.add_argument('--seed',type=int,default=301032)
    ap.add_argument('--stable-spatial',action='store_true')
    args=ap.parse_args();torch.set_num_threads(1);torch.manual_seed(args.seed)
    parent=Path('nn_checkpoints/experiments/correct121_failures_20261030_b')
    baseline=Net121.load(parent/'candidate.pt');args.out.mkdir(parents=True,exist_ok=False)
    lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);began=time.monotonic()
    def update(stage,**data):atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-began,**data))
    def record(name,row):
        with (args.out/(name+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
    config=dict(model=str((parent/'candidate.pt').resolve()),model_sha256=hashlib.sha256((parent/'candidate.pt').read_bytes()).hexdigest(),
                seed=args.seed,steps=args.steps,stable_spatial=args.stable_spatial,profile=r.PROFILE,simulations=32,confirmation_games=60,c_seconds=.03,
                training='value head only; policy+backbone frozen; stratified by true winning player, side to move and endgame',
                independent_gate_p=.01,automatic_promotion=False,
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['value121_repair.py','az121.py','rules121.py','curriculum121.py','iterate121.py']})
    atomic(args.out/'config.json',config);snapshot=args.out/'source';snapshot.mkdir()
    for name in config['source_sha256']:(snapshot/name).write_bytes(Path(name).read_bytes())
    try:
        old=Path('nn_checkpoints/experiments/curriculum121_100_20261029');splits=json.loads((old/'split.json').read_text())
        train_samples=[];validation={'retention':[],'correction':[]}
        for line in (old/'teacher_games.jsonl').read_text().splitlines():
            g=json.loads(line)
            if g['group'] in splits['train']:train_samples.extend(samples_from_game(g))
            elif g['group'] in splits['validation']:validation['retention'].extend(samples_from_game(g))
        split=json.loads((parent/'correction_split.json').read_text());val=set(split['validation'])
        for line in (parent/'corrections.jsonl').read_text().splitlines():
            g=json.loads(line)
            if g['group'] in val:validation['correction'].extend(samples_from_game(g))
            else:train_samples.extend(samples_from_game(g))
        strata=defaultdict(list)
        for sample in train_samples:
            player=1 if sample['x'][3].sum()>0 else 2
            strata[(sample['winner'],player,sample['endgame'])].append(sample)
        keys=sorted(strata)
        if len(keys)!=8:raise ValueError('Need both winners and sides in opening/endgame training strata')
        atomic(args.out/'data.json',dict(train_samples=len(train_samples),validation={k:len(v) for k,v in validation.items()},strata={str(k):len(v) for k,v in strata.items()}))
        before={k:metrics(baseline,v) for k,v in validation.items()};atomic(args.out/'baseline_validation.json',before)
        # Same frozen policy/features and same sample schedule for both controls.
        candidates={'balanced_pooled':copy.deepcopy(baseline),'balanced_spatial':make_spatial(baseline)}
        if args.stable_spatial:
            candidates={'low_lr_spatial':make_spatial(baseline),'normalized_spatial':make_spatial(baseline,normalized=True)}
        diagnostics={};best_models={}
        for name,model in candidates.items():
            for p in model.parameters():p.requires_grad=False
            for p in model.value.parameters():p.requires_grad=True
            lr=(3e-5 if name=='low_lr_spatial' else 1e-4) if args.stable_spatial else 3e-4
            opt=torch.optim.AdamW(model.value.parameters(),lr=lr,weight_decay=1e-4)
            rng=np.random.default_rng(args.seed);best=float('inf');history=[]
            for step in range(args.steps):
                chosen=[strata[key][int(rng.integers(len(strata[key])))] for key in keys for _ in range(2)]
                xs,_,_,_,z=batch(chosen);model.train();_,value=model(xs);loss=F.mse_loss(value,z)
                opt.zero_grad();loss.backward();gradient_norm=float(torch.nn.utils.clip_grad_norm_(model.value.parameters(),5));opt.step()
                if (step+1)%50==0 or step+1==args.steps:
                    scored={k:metrics(model,v) for k,v in validation.items()};criterion=sum(v['value_mse'] for v in scored.values())/len(scored)
                    history.append(dict(step=step+1,loss=float(loss.detach()),saturated_fraction=float((value.detach().abs()>.999).float().mean()),gradient_norm_before_clip=gradient_norm,lr=lr,validation=scored))
                    if criterion<best:best=criterion;model.save(args.out/(name+'.pt'))
                    atomic(args.out/(name+'_training.json'),history);update('value_training',arm=name,step=step+1)
            selected=Net121.load(args.out/(name+'.pt'));best_models[name]=selected
            diagnostics[name]={k:metrics(selected,v) for k,v in validation.items()}
            # Policy parameters are bit-identical. CPU GEMM storage alignment can
            # produce a few ulps of forward rounding despite identical weights.
            for key,tensor in selected.state_dict().items():
                if not key.startswith('value.') and not torch.equal(baseline.state_dict()[key],tensor):
                    raise ValueError('policy parameters changed during value-only repair')
            with torch.no_grad():
                for start in range(0,len(validation['retention']),32):
                    xs,*_=batch(validation['retention'][start:start+32])
                    torch.testing.assert_close(baseline(xs)[0],selected(xs)[0],atol=2e-6,rtol=1e-6)
        arm=min(diagnostics,key=lambda k:sum(v['value_mse'] for v in diagnostics[k].values()))
        candidate=best_models[arm];candidate.save(args.out/'candidate.pt')
        atomic(args.out/'selection.json',dict(arm=arm,before=before,after=diagnostics,criterion='mean of retention and correction validation MSE'))
        matches=[]
        for pair in range(30):
            prefix=opening(args.seed+100000+pair,0 if pair==0 else 4+(pair%5)*2)
            for side in (1,2):
                update('independent_confirmation',completed=len(matches),total=60,arm=arm)
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
        result=dict(arm=arm,validation=diagnostics,confirmation=summary(matches),paired_gate=gate,fixed_c=fixed,
                    qualified=gate['passed'] and gate['one_sided_p']<=.01 and fixed['candidate']['points']>=fixed['control']['points'],
                    policy_unchanged=True,promoted=False,seconds=time.monotonic()-began,
                    note='Value-only repair; training is teacher supervision, not pure selfplay; confirmation threshold is 0.01.')
        atomic(args.out/'result.json',result);update('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        update('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
