"""Fixed32 training-root fitting probe: root policy / replay / full objective.
Same initial weights, root samples, root coefficients and steps. No promotion.
"""
import argparse,copy,fcntl,hashlib,json,os,random,shutil,subprocess,sys,time,traceback
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from az121 import Net121
from curriculum121 import atomic,batch,samples_from_game
from canonical121_trial import canonical_sample
from corrected121_decisions import root_metrics,preference_loss
from generalization121_curve import remove_validation_duplicates
from cycle121_targeted import expert_samples
from unfreeze121_trial import metrics
from checkpoint121_save import digest,REGISTRY


def select32(roots,metadata,seed):
    lookup={x['decision_id']:x for x in metadata};groups=defaultdict(list)
    for sample in roots:groups[lookup[sample['decision_id']]['pair']].append(sample)
    complete=sorted(g for g,v in groups.items() if len(v)==4);random.Random(seed).shuffle(complete)
    if len(complete)<8:raise ValueError('need eight complete training opening groups')
    selected=[s for g in complete[:8] for s in groups[g]]
    if len(selected)!=32:raise ValueError('fixed training set not32')
    return selected,complete[:8]


def objective_terms(model,reference,roots,ordinary,arm):
    if arm not in ('root_policy','policy_replay','full_objective'):raise ValueError('unknown fitting arm')
    samples=roots if arm=='root_policy' else roots+ordinary
    xs,ids,mask,target,z=batch(samples);logits,value=model(xs)
    legal=logits.gather(1,ids).masked_fill(~mask,-1e9);root_ce=F.cross_entropy(legal[:len(roots)],target[:len(roots)])
    rank=preference_loss(logits[:len(roots)],roots);zero=root_ce*0
    terms=dict(root_ce=.25*root_ce,rank=.25*rank,replay_ce=zero,value=zero,kl=zero,value_preservation=zero)
    if arm!='root_policy':terms['replay_ce']=.75*F.cross_entropy(legal[len(roots):],target[len(roots):])
    if arm=='full_objective':
        weight=torch.tensor([s['value_weight'] for s in samples]);terms['value']=((value-z).square()*weight).sum()/weight.sum().clamp_min(1)
        with torch.no_grad():old_logits,old_value=reference(xs);old=old_logits.gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)
        logp=legal.log_softmax(1);terms['kl']=.1*(old.exp()*(old-logp)).sum(1).mean();terms['value_preservation']=.1*F.mse_loss(value,old_value)
    return sum(terms.values()),terms


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();torch.set_num_threads(1);seed=111053;torch.manual_seed(seed)
    parent=Path('nn_checkpoints/experiments/generalization121_200_20261101');source_config=json.loads((parent/'config.json').read_text());baseline_path=Path(source_config['model']);baseline=Net121.load(baseline_path);baseline.requires_grad_(False)
    registry_sha=digest(REGISTRY)
    args.out.mkdir(parents=True,exist_ok=False);lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic()
    def status(stage,**kw):atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-started,**kw))
    def log(message):
        with (args.out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-started:.1f}秒：{message}\n')
    names=['diagnostics/fit121_objectives.py','diagnostics/plot121_fit.py','az121.py','curriculum121.py','canonical121_trial.py','corrected121_decisions.py','generalization121_curve.py','cycle121_targeted.py','unfreeze121_trial.py','rules121.py','checkpoint121_save.py']
    config=dict(seed=seed,profile=source_config['profile'],encoding=baseline.encoding,model=str(baseline_path),model_sha256=digest(baseline_path),registry_sha256=registry_sha,
        arms=['root_policy','policy_replay','full_objective'],steps=1600,learning_rate=.00003,root_batch=4,ordinary_batch=12,root_ce_weight=.25,preference_weight=.25,
        control='Same32 training decisions/8 opening groups, same root sample order/coefficients/updates; policy_replay/full have same12 ordinary samples. Root-only has less compute, not more root exposures.',
        scope='Fitting diagnostic, no new data, game, training expansion or registry update. Full objective contains genuine ordinary-game value targets; no root outcome labels.',
        checkpoint='Primary at fixed1600, no validation-selected winner; thresholds are descriptive fitting checks, not architecture limits or playing strength.',
        source_sha256={name:digest(name) for name in names})
    for name in names:
        target=args.out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,target)
    data_dir=args.out/'data_sources';data_dir.mkdir();config['data_sha256']={}
    for name,path in {'decisions.pt':parent/'decisions.pt','decisions.jsonl':parent/'decisions.jsonl','old_teacher.jsonl':parent/'data_sources/old_teacher.jsonl','old_split.json':parent/'data_sources/old_split.json','normal_games.jsonl':parent/'data_sources/normal_games.jsonl'}.items():
        shutil.copyfile(path,data_dir/name);config['data_sha256'][name]=digest(data_dir/name)
    atomic(args.out/'config.json',config)
    try:
        root_data=torch.load(data_dir/'decisions.pt',weights_only=True)['sets'];metadata=[json.loads(s) for s in (data_dir/'decisions.jsonl').read_text().splitlines()]
        fixed,groups=select32(root_data['train'],metadata,seed);holdout=root_data['validation'];lookup={r['decision_id']:r for r in metadata}
        if set(groups)&{lookup[s['decision_id']]['pair'] for s in holdout}:raise ValueError('train/validation opening leak')
        atomic(args.out/'fixed32.json',dict(decision_ids=[s['decision_id'] for s in fixed],opening_groups=groups,players=dict(__import__('collections').Counter(lookup[s['decision_id']]['player'] for s in fixed))))
        old={'train':[],'validation':[]};normal={'train':[],'validation':[]};split=json.loads((data_dir/'old_split.json').read_text())
        for line in (data_dir/'old_teacher.jsonl').read_text().splitlines():
            row=json.loads(line);part=next((k for k in old if row['group'] in split[k]),None)
            if part:old[part].extend({**canonical_sample(s),'value_weight':1.} for s in samples_from_game(row))
        for line in (data_dir/'normal_games.jsonl').read_text().splitlines():
            row=json.loads(line);normal[row['split']].extend(expert_samples(row))
        old['train'],_=remove_validation_duplicates(old['train'],holdout);normal['train'],_=remove_validation_duplicates(normal['train'],holdout)
        def measure(model):return dict(train32=root_metrics(model,fixed),validation40=root_metrics(model,holdout),retention=metrics(model,old['validation']),normal=metrics(model,normal['validation']))
        before=measure(baseline);atomic(args.out/'baseline.json',before);histories={};log('32训练局面/8开局固定，16个P1/16个P2；40验证局面开局隔离。三组相同初值、根样本序列、根损失系数和1600次更新。')
        for arm in config['arms']:
            model=copy.deepcopy(baseline);model.requires_grad_(True)
            if not all(torch.equal(v,model.state_dict()[k]) for k,v in baseline.state_dict().items()):raise ValueError('initial parameter mismatch')
            if arm!='full_objective':model.value.requires_grad_(False)
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.00003,weight_decay=.0001)
            root_rng=np.random.default_rng(seed+1);ordinary_rng=np.random.default_rng(seed+2);history=[];exposure={};root_sequence=hashlib.sha256();ordinary_sequence=hashlib.sha256();arm_started=time.monotonic()
            for step in range(1601):
                if step%100==0:
                    scores=measure(model);history.append(dict(step=step,**scores));atomic(args.out/(arm+'_curve.json'),history);status('fitting_probe',arm=arm,step=step,total=1600)
                    if step%400==0:model.save(args.out/(arm+'_step'+str(step)+'.pt'))
                    if step==1600:break
                roots=[fixed[int(i)] for i in root_rng.integers(len(fixed),size=4)]
                old_ids=ordinary_rng.integers(len(old['train']),size=6);normal_ids=ordinary_rng.integers(len(normal['train']),size=6)
                ordinary=[old['train'][int(i)] for i in old_ids]+[normal['train'][int(i)] for i in normal_ids]
                root_sequence.update(json.dumps([s['decision_id'] for s in roots]).encode());ordinary_sequence.update(old_ids.tobytes()+normal_ids.tobytes())
                for sample in roots:exposure[sample['decision_id']]=exposure.get(sample['decision_id'],0)+1
                loss,terms=objective_terms(model,baseline,roots,ordinary,arm)
                if not torch.isfinite(loss):raise ValueError('nonfinite diagnostic loss')
                opt.zero_grad();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1));opt.step()
                if step%100==0:atomic(args.out/(arm+'_latest_loss.json'),dict(step=step+1,gradient_norm=norm,terms={k:float(v.detach()) for k,v in terms.items()}))
            histories[arm]=history;atomic(args.out/(arm+'_exposure.json'),dict(root_examples=sum(exposure.values()),per_decision=exposure,ordinary_examples=0 if arm=='root_policy' else 1600*12,updates=1600,root_sequence_sha256=root_sequence.hexdigest(),ordinary_sequence_sha256=ordinary_sequence.hexdigest(),seconds=time.monotonic()-arm_started))
            log(arm+'完成：'+json.dumps(history[-1],ensure_ascii=False))
        exposures=[json.loads((args.out/(arm+'_exposure.json')).read_text())['per_decision'] for arm in config['arms']]
        if exposures[0]!=exposures[1] or exposures[1]!=exposures[2]:raise ValueError('root sampling differed across arms')
        manifests=[json.loads((args.out/(arm+'_exposure.json')).read_text()) for arm in config['arms']]
        if len({x['root_sequence_sha256'] for x in manifests})!=1 or len({x['ordinary_sequence_sha256'] for x in manifests})!=1:raise ValueError('sample order differed across arms')
        if digest(REGISTRY)!=registry_sha or digest(baseline_path)!=config['model_sha256']:raise ValueError('champion changed')
        result=dict(final={a:h[-1] for a,h in histories.items()},baseline=before,fixed_train32=True,heldout40=True,identical_root_exposure=True,promoted=False,arena_run=False,seconds=time.monotonic()-started)
        atomic(args.out/'result.json',result)
        plotted=subprocess.run(['/Library/Frameworks/Python.framework/Versions/3.10/bin/python3','diagnostics/plot121_fit.py',str(args.out)],capture_output=True,text=True)
        atomic(args.out/'plot_status.json',dict(exit_code=plotted.returncode,stderr=plotted.stderr[-2000:]));log('拟合对照完成，源模型与GUI未变；本轮结果不能直接作为棋力提升或CNN能力上限的证明。');status('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        status('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
