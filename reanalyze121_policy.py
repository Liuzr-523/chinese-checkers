"""Reanalyse existing selfplay with clean MCTS64, freeze CNN/value, fit policy heads.
No teacher-engine labels or rule augmentation. Noise-target control shares states.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from az121 import Net121,Neural121,search,action_id
from curriculum121 import atomic,opening,replay,finish
from engine_c121 import CSearch121
from iterate121 import summary,paired_gate
import rules121 as r


def features(model,x):
    h=F.relu(model.stem(x))
    for block in model.blocks:h=F.relu(h+block(h))
    return h[:,:,model.rows,model.cols].transpose(1,2)


def logits_from_features(model,cells):
    bias=model.bias(cells)
    logits=torch.bmm(model.src(cells),model.dst(cells).transpose(1,2))/4
    return (logits+bias[:,:,0].unsqueeze(2)+bias[:,:,1].unsqueeze(1)).flatten(1)


def decision_target(probabilities,kind):
    p=np.asarray(probabilities,dtype=np.float64)
    if not np.isfinite(p).all() or (p<0).any() or p.sum()<=0:raise ValueError('invalid policy target')
    p=p/p.sum()
    if kind=='hard':
        result=np.zeros_like(p);result[int(p.argmax())]=1.;return result.tolist()
    if kind=='sharp':p=p**4;p=p/p.sum()
    return p.tolist()


def batch(samples,target):
    n=len(samples);maximum=max(len(s['actions']) for s in samples)
    cells=torch.stack([s['features'] for s in samples]);ids=torch.zeros((n,maximum),dtype=torch.long)
    mask=torch.zeros_like(ids,dtype=torch.bool);pi=torch.zeros((n,maximum))
    for i,s in enumerate(samples):
        k=len(s['actions']);ids[i,:k]=torch.tensor(s['actions']);mask[i,:k]=True;pi[i,:k]=torch.tensor(decision_target(s['clean'],target) if target in ('sharp','hard') else s[target])
    return cells,ids,mask,pi


@torch.no_grad()
def loss_on(model,samples,hard=False,temperature=1.):
    total=0.;n=0
    for start in range(0,len(samples),32):
        part=samples[start:start+32];cells,ids,mask,pi=batch(part,'hard' if hard else 'clean')
        logp=(logits_from_features(model,cells)/temperature).gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)
        total+=float(-(pi*logp).sum());n+=len(part)
    return total/n


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=301040);ap.add_argument('--steps',type=int,default=600)
    ap.add_argument('--max-states-per-game',type=int,default=40);ap.add_argument('--decision-targets',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    source=Path('nn_checkpoints/experiments/selfplay121_v2_retained_20261030')
    model_path=Path('nn_checkpoints/models/checkpoints/neural121_v2_canonical.pt');baseline=Net121.load(model_path)
    args.out.mkdir(parents=True,exist_ok=False);lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);began=time.monotonic()
    def update(stage,**extra):atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-began,**extra))
    def record(stage,row):
        with (args.out/(stage+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
    config=dict(model=str(model_path.resolve()),model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                profile=r.PROFILE,encoding=baseline.encoding,seed=args.seed,steps=args.steps,max_states_per_game=args.max_states_per_game,decision_targets=args.decision_targets,
                training='Only existing neural selfplay states; clean MCTS64 reanalysis vs original noisy target, frozen CNN backbone and value',
                simulations=32,teacher_simulations=64,lr=.0003,kl_to_frozen_policy=.1,confirmation_games=60,independent_gate_p=.01,
                source_games=str(source/'selfplay_games.jsonl'),source_games_sha256=hashlib.sha256((source/'selfplay_games.jsonl').read_bytes()).hexdigest(),
                automatic_promotion=False,source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['reanalyze121_policy.py','az121.py','rules121.py','curriculum121.py','iterate121.py']})
    atomic(args.out/'config.json',config);snapshot=args.out/'source';snapshot.mkdir()
    for name in config['source_sha256']:(snapshot/name).write_bytes(Path(name).read_bytes())
    try:
        if args.decision_targets:
            cache=Path('nn_checkpoints/experiments/reanalyze121_clean_policy_20261030/reanalysis.pt')
            cache_config=json.loads(cache.with_name('config.json').read_text())
            if cache_config['model_sha256']!=config['model_sha256']:raise ValueError('cached features from another baseline')
            data=torch.load(cache,weights_only=True)
            if data['encoding']!=baseline.encoding:raise ValueError('cached encoding mismatch')
            samples=data['samples'];config['cached_states']=str(cache.resolve());config['cached_states_sha256']=hashlib.sha256(cache.read_bytes()).hexdigest()
            config['training']='Frozen CNN/value: raw, sharpened and argmax clean MCTS64 decision targets; compare a temperature-only confidence control'
            atomic(args.out/'config.json',config)
        else:
            validation=set(json.loads((source/'split.json').read_text())['validation_groups'])
            samples=[];games=[json.loads(x) for x in (source/'selfplay_games.jsonl').read_text().splitlines()];rng=np.random.default_rng(args.seed)
            checksums={}
            for index,game in enumerate(games):
                update('clean_reanalysis',completed=index,total=len(games))
                path=source/'selfplay'/f"game_{game['game_id']:03d}.pt";checksums[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
                shard=torch.load(path,weights_only=True)
                if shard['encoding']!=baseline.encoding:raise ValueError('source encoding mismatch')
                original=shard['samples'];positions=np.arange(4,len(game['steps']))
                picked=set(map(int,rng.choice(positions,size=min(args.max_states_per_game,len(positions)),replace=False)))
                g=r.Game()
                for i,step in enumerate(game['steps']):
                    if g.board!=step['board'] or g.player!=step['player']:raise ValueError('source board mismatch')
                    if i in picked:
                        old=original[i-4];_,info=search(baseline,g.board,g.player,g.ply,g.repetitions,64,noise=False)
                        actions=[action_id(p[0],p[-1],g.player,baseline.canonical) for p in info['paths']]
                        if actions!=old['actions']:raise ValueError('action encoding/order mismatch')
                        with torch.no_grad():cells=features(baseline,old['x'].unsqueeze(0))[0].detach().clone()
                        samples.append(dict(features=cells,actions=actions,noisy=old['policy'],clean=info['policy'],
                                            validation=game['opening_group'] in validation,game_id=game['game_id'],x=old['x']))
                    g.play(step['path'])
                if g.result()!=(game['winner'],game['reason']):raise ValueError('source outcome mismatch')
            atomic(args.out/'source_shards.json',checksums)
            torch.save(dict(encoding=baseline.encoding,samples=samples),args.out/'reanalysis.pt')
        training=[s for s in samples if not s['validation']];heldout=[s for s in samples if s['validation']]
        atomic(args.out/'data.json',dict(states=len(samples),training=len(training),validation=len(heldout),baseline_validation_ce=loss_on(baseline,heldout,hard=args.decision_targets)))
        arms={};models={}
        for target in (['clean','sharp','hard'] if args.decision_targets else ['noisy','clean']):
            model=copy.deepcopy(baseline);model.requires_grad_(False)
            for head in [model.src,model.dst,model.bias]:head.requires_grad_(True)
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.0001)
            local=np.random.default_rng(args.seed);history=[];best=float('inf')
            for step in range(args.steps):
                chosen=[training[int(i)] for i in local.integers(len(training),size=16)]
                cells,ids,mask,pi=batch(chosen,target);logp=logits_from_features(model,cells).gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)
                with torch.no_grad():old=logits_from_features(baseline,cells).gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)
                kl=(old.exp()*(old-logp)).sum(1).mean();loss=-(pi*logp).sum(1).mean()+.1*kl
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);opt.step()
                if (step+1)%50==0 or step+1==args.steps:
                    ce=loss_on(model,heldout,hard=args.decision_targets);history.append(dict(step=step+1,loss=float(loss.detach()),clean_validation_ce=ce))
                    if ce<best:best=ce;model.save(args.out/(target+'.pt'))
                    atomic(args.out/(target+'_training.json'),history);update('policy_head_fit',arm=target,step=step+1)
            selected=Net121.load(args.out/(target+'.pt'))
            for name,tensor in selected.state_dict().items():
                if not name.startswith(('src.','dst.','bias.')) and not torch.equal(baseline.state_dict()[name],tensor):raise ValueError('frozen backbone/value changed')
            models[target]=selected;arms[target]=loss_on(selected,heldout,hard=args.decision_targets)
        arm=min(arms,key=arms.get);candidate=models[arm];temperature=1.
        calibration=loss_on(baseline,heldout,hard=True,temperature=.5) if args.decision_targets else None
        if args.decision_targets and calibration<=arms[arm]:
            arm='temperature_only';candidate=baseline;temperature=.5
            shutil.copyfile(model_path,args.out/'candidate.pt')
        else:candidate.save(args.out/'candidate.pt')
        config['policy_temperature']=temperature;atomic(args.out/'config.json',config)
        atomic(args.out/'selection.json',dict(arm=arm,validation_ce=arms,criterion='hard decision CE' if args.decision_targets else 'soft visit CE',temperature_only_ce=calibration))
        matches=[]
        for pair in range(30):
            prefix=opening(args.seed+100000+pair,0 if pair==0 else 4+(pair%5)*2)
            for side in (1,2):
                update('independent_confirmation',completed=len(matches),total=60)
                row=finish(replay(prefix),{side:Neural121(candidate,32,policy_temperature=temperature),3-side:Neural121(baseline,32)})
                row.update(pair=pair,side=side,opening=prefix);matches.append(row);record('confirmation',row)
        fixed={}
        for name,model in [('control',baseline),('candidate',candidate)]:
            rows=[]
            for pair in range(10):
                prefix=opening(args.seed+200000+pair,0 if pair==0 else 4+(pair%4)*2)
                for side in (1,2):
                    update('fixed_c_'+name,completed=len(rows),total=20)
                    row=finish(replay(prefix),{side:Neural121(model,32,policy_temperature=temperature if name=='candidate' else 1.),3-side:CSearch121(seconds=.03,max_depth=10)})
                    row.update(pair=pair,side=side,opening=prefix);rows.append(row);record('fixed_c_'+name,row)
            fixed[name]=summary(rows)
        gate=paired_gate(matches)
        result=dict(arm=arm,policy_temperature=temperature,decision_targets=args.decision_targets,validation_ce=arms,confirmation=summary(matches),paired_gate=gate,fixed_c=fixed,
                    qualified=gate['passed'] and gate['one_sided_p']<=.01 and fixed['candidate']['points']>=fixed['control']['points'],
                    backbone_and_value_unchanged=True,promoted=False,seconds=time.monotonic()-began,
                    note=('Confidence calibration only, unchanged checkpoint; no learned weight gain.' if arm=='temperature_only' else 'Pure neural policy-head distillation; frozen CNN/value, no teacher-engine examples.'))
        atomic(args.out/'result.json',result);update('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        update('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
