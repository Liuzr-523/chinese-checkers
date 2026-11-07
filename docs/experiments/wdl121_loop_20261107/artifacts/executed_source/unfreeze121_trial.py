"""User-authorized saved-v3 -> controlled CNN unfreezing -> audited accept/retain.
Real expert outcomes only for value targets; unplayed suggestions are policy-only.
"""
import argparse,copy,fcntl,hashlib,json,os,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import rules121 as r
from az121 import Net121,Neural121,encode,action_id
from canonical121_trial import canonical_sample
from curriculum121 import atomic,replay,opening,finish,samples_from_game,batch
from engine_c121 import CSearch121
from iterate121 import summary,paired_gate
from checkpoint121_save import verify,digest,REGISTRY


def objective(model,reference,xs,ids,mask,target,z,weight):
    logits,value=model(xs);logp=logits.gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)
    with torch.no_grad():old_logits,old_value=reference(xs);old=old_logits.gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)
    policy=F.nll_loss(logp,target);kl=(old.exp()*(old-logp)).sum(1).mean()
    genuine_value=((value-z).square()*weight).sum()/weight.sum().clamp_min(1)
    return policy+genuine_value+.1*kl+.1*F.mse_loss(value,old_value)


@torch.no_grad()
def metrics(model,samples):
    ce=value=hits=n=w=0
    for start in range(0,len(samples),32):
        group=samples[start:start+32];xs,ids,mask,target,z=batch(group);logits,v=model(xs);lp=logits.gather(1,ids).masked_fill(~mask,-1e9)
        ce+=float(F.cross_entropy(lp,target,reduction='sum'));hits+=int((lp.argmax(1)==target).sum());n+=len(group)
        weights=torch.tensor([s['value_weight'] for s in group]);value+=float(((v-z).square()*weights).sum());w+=float(weights.sum())
    return dict(samples=n,policy_ce=ce/n,policy_accuracy=hits/n,value_mse=value/max(w,1),genuine_value_samples=w)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--save',type=Path,required=True);ap.add_argument('--seed',type=int,default=301047);args=ap.parse_args()
    torch.set_num_threads(1);torch.manual_seed(args.seed);saved=verify(args.save)
    if digest(REGISTRY)!=saved['registry_sha256']:raise ValueError('active registry changed since save; do not overwrite')
    baseline_path=args.save/'model.pt';baseline=Net121.load(baseline_path)
    args.out.mkdir(parents=True,exist_ok=False);lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic()
    def update(stage,**kw):atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-started,**kw))
    def record(stage,row):
        with (args.out/(stage+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
    names=['unfreeze121_trial.py','checkpoint121_save.py','az121.py','rules121.py','canonical121_trial.py','curriculum121.py','iterate121.py','engine_c121.py','neural121_registry.py']
    config=dict(model=str(baseline_path.resolve()),model_sha256=digest(baseline_path),profile=r.PROFILE,encoding=baseline.encoding,save=str(args.save.resolve()),save_manifest_sha256=digest(args.save/'manifest.json'),
        seed=args.seed,simulations=32,steps=800,learning_rate=.00003,arms=['frozen_heads_control','unfrozen_cnn'],
        training='Unfreeze CNN/policy/value with expert correction + genuine teacher replay, 50:50. Control freezes CNN/value. Not pure selfplay.',
        policy_kl=.1,value_preservation=.1,selection='Best unfrozen checkpoint on mean correction/retention policy CE + genuine-retention value MSE; frozen arm is diagnostic control',
        confirmation_games=60,independent_gate_p=.01,automatic_promotion='Only after independent gate and full trajectory/hash audit; retain saved v3 otherwise',
        source_sha256={name:digest(name) for name in names})
    for name in names:
        dest=args.out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
    old=Path('nn_checkpoints/experiments/imitate_clysto121_20261030');dagger=Path('nn_checkpoints/experiments/dagger121_expert_correction_20261030')
    sources={'teacher_games.jsonl':old/'teacher_games.jsonl','teacher_split.json':old/'split.json','expert_rollout.jsonl':dagger/'expert_rollout.jsonl','annotations.jsonl':dagger/'annotations.jsonl','correction_split.json':dagger/'split.json'}
    data=args.out/'data_sources';data.mkdir();config['data_sha256']={name:digest(path) for name,path in sources.items()}
    for name,path in sources.items():shutil.copyfile(path,data/name)
    atomic(args.out/'config.json',config)
    try:
        update('reconstruct_training_data');sets={'train':[],'validation':[]};retention={'train':[],'validation':[]}
        split=json.loads((data/'correction_split.json').read_text());annotations=[json.loads(s) for s in (data/'annotations.jsonl').read_text().splitlines()]
        games=[json.loads(s) for s in (data/'expert_rollout.jsonl').read_text().splitlines()]
        for game,ann in zip(games,annotations):
            key=next((k for k in sets if game['pair'] in split[k]),None)
            if key is None:continue
            g=replay(game['opening']);labels={s['ply']:s for s in ann['annotations']}
            for path in game['paths']:
                if g.ply in labels:
                    label=labels[g.ply];teacher_path=label['expert_path']
                    if label['played_path']!=path or not r.valid_path(g.board,g.player,teacher_path):raise ValueError('corrupt expert annotation')
                    actions=[action_id(s,p[-1],g.player,True) for s,p in r.get_all_moves(g.board,g.player)]
                    target=actions.index(action_id(teacher_path[0],teacher_path[-1],g.player,True))
                    sets[key].append(dict(x=encode(g.board,g.player,g.ply,g.repetitions,canonical=True),actions=actions,target=target,z=0.,value_weight=0.))
                g.play(path)
            if g.result()!=(game['winner'],game['reason']):raise ValueError('incorrect rollout outcome')
        teacher_split=json.loads((data/'teacher_split.json').read_text())
        for line in (data/'teacher_games.jsonl').read_text().splitlines():
            game=json.loads(line);key=next((k for k in retention if game['group'] in teacher_split[k]),None)
            if key:
                for s in samples_from_game(game):retention[key].append({**canonical_sample(s),'value_weight':1.})
        if any(not s for domain in [sets,retention] for s in domain.values()):raise ValueError('empty train/validation domain')
        atomic(args.out/'data.json',dict(correction={k:len(v) for k,v in sets.items()},retention={k:len(v) for k,v in retention.items()},correction_value_labels=0))
        def evaluate(model):
            a=metrics(model,sets['validation']);b=metrics(model,retention['validation'])
            return dict(correction=a,retention=b,criterion=.5*(a['policy_ce']+b['policy_ce'])+b['value_mse'])
        before=evaluate(baseline);atomic(args.out/'baseline_validation.json',before);scores={}
        for arm in config['arms']:
            model=copy.deepcopy(baseline)
            if arm=='frozen_heads_control':
                model.requires_grad_(False)
                for head in [model.src,model.dst,model.bias]:head.requires_grad_(True)
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.00003,weight_decay=.0001)
            rng=np.random.default_rng(args.seed);best=float('inf');history=[]
            for step in range(800):
                selected=[sets['train'][int(i)] for i in rng.integers(len(sets['train']),size=8)]+[retention['train'][int(i)] for i in rng.integers(len(retention['train']),size=8)]
                xs,ids,mask,target,z=batch(selected);weight=torch.tensor([s['value_weight'] for s in selected])
                loss=objective(model,baseline,xs,ids,mask,target,z,weight)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                if (step+1)%50==0:
                    val=evaluate(model);history.append(dict(step=step+1,loss=float(loss.detach()),validation=val))
                    if val['criterion']<best:best=val['criterion'];model.save(args.out/(arm+'.pt'))
                    atomic(args.out/(arm+'_training.json'),history);update('training',arm=arm,step=step+1,total=800)
            chosen=Net121.load(args.out/(arm+'.pt'));scores[arm]=evaluate(chosen)
            if arm=='unfrozen_cnn':
                changed=[k for k,v in chosen.state_dict().items() if not torch.equal(v,baseline.state_dict()[k])]
                if not any(k.startswith('stem.') for k in changed) or not any(k.startswith('value.') for k in changed):raise ValueError('CNN/value did not unfreeze')
                atomic(args.out/'unfreeze_audit.json',dict(changed_parameters=changed))
        candidate=Net121.load(args.out/'unfrozen_cnn.pt');candidate.save(args.out/'candidate.pt')
        atomic(args.out/'selection.json',dict(selected='unfrozen_cnn',baseline=before,arms=scores,criterion=config['selection']))
        matches=[]
        for pair in range(30):
            prefix=opening(args.seed+100000+pair,0 if pair==0 else 4+(pair%5)*2)
            for side in (1,2):
                update('independent_confirmation',completed=len(matches),total=60)
                row=finish(replay(prefix),{side:Neural121(candidate,32),3-side:Neural121(baseline,32)})
                row.update(pair=pair,side=side,opening=prefix);matches.append(row);record('confirmation',row)
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
        gate=paired_gate(matches);qualified=gate['one_sided_p']<=.01 and fixed['candidate']['points']>=fixed['control']['points']
        result=dict(confirmation=summary(matches),paired_gate=gate,fixed_c=fixed,qualified=qualified,promoted=False,seconds=time.monotonic()-started)
        atomic(args.out/'result.json',result)
        verify(args.save)
        if digest(REGISTRY)!=saved['registry_sha256']:raise ValueError('Registry changed concurrently; do not adopt/restore over another change')
        if qualified:
            from neural121_registry import register,load_registered
            try:
                entry=register(args.out,'v4_unfrozen_cnn');engine=load_registered()
                for p in (1,2):
                    path,_=engine.think(r.init_board(),p)
                    if not r.valid_path(r.init_board(),p,path):raise ValueError('adopted engine failed smoke check')
                atomic(args.out/'adoption.json',dict(adopted=True,version=entry['version'],save_preserved=True))
            except BaseException:
                active=json.loads(REGISTRY.read_text())
                if active.get('version')=='v4_unfrozen_cnn' and active.get('sha256')==digest(args.out/'candidate.pt'):
                    from checkpoint121_save import restore
                    restore(args.save)
                    atomic(args.out/'adoption.json',dict(adopted=False,rolled_back_after_failed_smoke=True,version=saved['version']))
                raise
        else:
            atomic(args.out/'adoption.json',dict(adopted=False,retained_version=saved['version'],save_verified=True,active_registry_matches_save=True))
        update('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        update('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
