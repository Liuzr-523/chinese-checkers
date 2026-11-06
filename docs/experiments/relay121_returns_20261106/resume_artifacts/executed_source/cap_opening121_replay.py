"""同批同预算对照：仅限制普通策略回放在共享参数上的原始梯度范数。"""
import argparse,copy,fcntl,hashlib,json,os,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from az121 import Net121
from ablate_opening121_hard32 import terms
from curriculum121 import atomic,batch,samples_from_game
from canonical121_trial import canonical_sample
from cycle121_targeted import expert_samples
from generalization121_curve import remove_validation_duplicates
from opening121 import OpeningBook
from train_opening121 import book_samples,mastery
from fit_opening121_hard32 import predict
from unfreeze121_trial import metrics
from checkpoint121_save import REGISTRY,digest
SOURCE=Path('nn_checkpoints/experiments/ablate_opening121_hard32_20261105')
ARMS=('control','capped')


def policy_gradients(loss,params):
    return tuple(torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)))


def norm_of(grads):
    return torch.stack([g.square().sum() for g in grads]).sum().sqrt()


def dot_of(left,right):
    return torch.stack([(a*b).sum() for a,b in zip(left,right)]).sum()


def cap_coefficient(root_norm,replay_norm,enabled=True):
    """Detached coefficient; norm cap is on already weighted raw gradients, before AdamW."""
    with torch.no_grad():
        if not bool(torch.isfinite(root_norm)) or not bool(torch.isfinite(replay_norm)):
            raise ValueError('非有限专项/回放梯度范数')
        if not enabled or float(replay_norm)==0:
            return torch.ones_like(replay_norm)
        return torch.minimum(torch.ones_like(replay_norm),root_norm/replay_norm).detach()


def load_data(folder):
    fixed=torch.load(folder/'fixed32.pt',weights_only=True)['samples'];old={'train':[],'validation':[]};normal={'train':[],'validation':[]};split=json.loads((folder/'old_split.json').read_text())
    for line in (folder/'old_teacher.jsonl').read_text().splitlines():
        row=json.loads(line);part=next((p for p in old if row['group'] in split[p]),None)
        if part:old[part].extend({**canonical_sample(s),'value_weight':1.} for s in samples_from_game(row))
    for line in (folder/'normal_games.jsonl').read_text().splitlines():
        row=json.loads(line);normal[row['split']].extend(expert_samples(row))
    sets,_=book_samples(OpeningBook())
    for pool in (old,normal):pool['train'],_=remove_validation_duplicates(pool['train'],sets['validation'])
    return fixed,old,normal,sets


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();torch.set_num_threads(1)
    previous=json.loads((SOURCE/'config.json').read_text());model_path=Path(previous['model']);registry=digest(REGISTRY)
    if registry!=previous['registry_sha256'] or digest(model_path)!=previous['model_sha256']:raise ValueError('现役/初值已变化')
    args.out.mkdir(parents=True,exist_ok=False);lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic()
    def status(stage,**kw):
        atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),seconds=time.monotonic()-started,**kw))
        if time.monotonic()-started>3600 and stage not in ('complete','failed'):raise TimeoutError('一小时预算耗尽')
    def log(text):
        with (args.out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-started:.1f}秒：{text}\n')
    sources=['cap_opening121_replay.py','ablate_opening121_hard32.py','fit_opening121_hard32.py','opening121_ratios.py','train_opening121.py','opening121.py','opening121_book.json','az121.py','rules121.py','curriculum121.py','canonical121_trial.py','cycle121_targeted.py','generalization121_curve.py','unfreeze121_trial.py','checkpoint121_save.py','tests/test_opening121_replay_cap.py','diagnostics/plot121_replay_cap.py','diagnostics/plot_style_zh.py']
    for name in sources:
        target=args.out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,target)
    shutil.copytree(SOURCE/'data_sources',args.out/'data_sources');shutil.copyfile(SOURCE/'fixed32.json',args.out/'fixed32.json')
    config=dict(source=str(SOURCE),model=str(model_path),model_sha256=digest(model_path),registry_sha256=registry,source_sha256={name:digest(name) for name in sources},data_sha256={name:digest(args.out/'data_sources'/name) for name in previous['data_sha256']},arms=list(ARMS),steps=2400,lr=3e-5,root_seed=110414,ordinary_seed=110415,max_seconds=3600,
        intervention='仅以detached min(1,专项梯度范数/回放梯度范数)缩放已含.75系数的回放CE。范数取共享CNN和policy参数，排除value头；不投影，aux不变。所有组每步同样计算梯度。',
        criteria='最终难例>=29/32；旧策略CE、旧value MSE、普通CE分别<=v3+0.05，均满足才可计划扩大训练与全新开发集。原留出已多轮诊断，无部署无独立对战。')
    atomic(args.out/'config.json',config)
    try:
        if config['data_sha256']!=previous['data_sha256']:raise ValueError('原数据哈希变化')
        fixed,old,normal,sets=load_data(args.out/'data_sources');base=Net121.load(model_path);base.requires_grad_(False)
        def measure(m):return dict(fixed32=mastery(m,fixed),validation=mastery(m,sets['validation']),retention=metrics(m,old['validation']),normal=metrics(m,normal['validation']))
        before=measure(base);atomic(args.out/'baseline.json',before);final={};audits={};log('两组开始，固定32/相同采样/2400更新；仅回放梯度范数上限干预。')
        for arm in ARMS:
            m=copy.deepcopy(base);m.requires_grad_(True);params=[p for n,p in m.named_parameters() if not n.startswith('value.')];opt=torch.optim.AdamW(m.parameters(),lr=3e-5,weight_decay=1e-4);rr=np.random.default_rng(110414);ro=np.random.default_rng(110415);seq=hashlib.sha256();curve=[];cases=[];updates=[];probes=[]
            for step in range(2401):
                if step%100==0:
                    curve.append(dict(step=step,**measure(m)));cases+=predict(m,fixed,step);atomic(args.out/f'{arm}_curve.json',curve);atomic(args.out/f'{arm}_cases.json',cases);status('回放梯度上限对照',arm=arm,step=step)
                    atomic(args.out/f'{arm}_updates.json',updates);atomic(args.out/f'{arm}_probes.json',probes)
                    if step%600==0:m.save(args.out/f'{arm}_step{step}.pt')
                    if step==2400:break
                ri=rr.integers(32,size=4);oi=ro.integers(len(old['train']),size=6);ni=ro.integers(len(normal['train']),size=6);seq.update(ri.tobytes()+oi.tobytes()+ni.tobytes());roots=[fixed[int(i)] for i in ri];ordinary=[old['train'][int(i)] for i in oi]+[normal['train'][int(i)] for i in ni]
                root,replay,aux=terms(m,base,roots,ordinary);gr=policy_gradients(root,params);gp=policy_gradients(replay,params);rn=norm_of(gr);pn=norm_of(gp);weight=cap_coefficient(rn,pn,arm=='capped');loss=root+weight*replay+aux
                if not torch.isfinite(loss) or not torch.isfinite(weight):raise ValueError('非有限损失/系数')
                if arm=='capped' and float(weight*pn)>float(rn)*1.00001+1e-7:raise ValueError('回放范数超过专项上限')
                probe=None
                if step%200==0:
                    ga=policy_gradients(aux,params);an=norm_of(ga);saved=[p.detach().clone() for p in params];probe=dict(step=step,aux_norm=float(an),root_aux_cosine=float(dot_of(gr,ga)/(rn*an).clamp_min(1e-12)))
                opt.zero_grad();loss.backward()
                total_dot=float(dot_of(tuple(torch.zeros_like(p) if p.grad is None else p.grad for p in params),gr))
                preclip=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True))
                if probe is not None:probe['postclip_total_norm']=float(norm_of(tuple(p.grad for p in m.parameters() if p.grad is not None)))
                opt.step()
                updates.append(dict(step=step,root_loss=float(root.detach()),replay_loss=float(replay.detach()),aux_loss=float(aux.detach()),root_norm=float(rn),replay_norm=float(pn),weight=float(weight),capped_replay_norm=float(weight*pn),root_replay_cosine=float(dot_of(gr,gp)/(rn*pn).clamp_min(1e-12)),total_gradient_dot_root=total_dot,preclip_norm=preclip,clipped=preclip>1))
                if probe is not None:
                    delta=[p.detach()-oldp for p,oldp in zip(params,saved)];dn=norm_of(delta);probe.update(actual_adam_update_norm=float(dn),actual_root_descent_dot=float(-dot_of(delta,gr)),actual_update_root_descent_cosine=float(-dot_of(delta,gr)/(dn*rn).clamp_min(1e-12)));probes.append(probe)
            final[arm]=curve[-1];audits[arm]=dict(sequence_sha256=seq.hexdigest(),root_exposures=9600,ordinary_exposures=28800,clipped_updates=sum(u['clipped'] for u in updates),cap_active=sum(u['weight']<1 for u in updates),mean_weight=sum(u['weight'] for u in updates)/2400);atomic(args.out/'sampling_audit.json',audits);log(f'{arm}完成：{round(32*final[arm]["fixed32"]["top1_in_book"])}/32；旧CE {final[arm]["retention"]["policy_ce"]:.3f}')
        if len({v['sequence_sha256'] for v in audits.values()})!=1:raise ValueError('采样不一致')
        if digest(REGISTRY)!=registry or digest(model_path)!=config['model_sha256']:raise ValueError('现役被改动')
        control=final['control'];expected=json.loads((SOURCE/'result.json').read_text())['final']['full'];checks={k:abs(control['retention'][k]-expected['retention'][k])<1e-6 for k in ('policy_ce','value_mse')};checks['fixed32']=control['fixed32']['top1_in_book']==expected['fixed32']['top1_in_book']
        cap=final['capped'];gates=dict(fit=cap['fixed32']['top1_in_book']>=29/32,old_policy=cap['retention']['policy_ce']<=before['retention']['policy_ce']+.05,old_value=cap['retention']['value_mse']<=before['retention']['value_mse']+.05,normal_policy=cap['normal']['policy_ce']<=before['normal']['policy_ce']+.05)
        atomic(args.out/'result.json',dict(baseline=before,final=final,control_reproduction=checks,gates=gates,diagnostic_pass=all(gates.values()) and all(checks.values()),arena_run=False,promoted=False,seconds=time.monotonic()-started))
        if not all(checks.values()):raise ValueError('对照未复现，应先审查，不解释干预效果')
        status('complete');log('结束；原注册表/模型未改变，无对战无晋升。')
    except BaseException:status('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
