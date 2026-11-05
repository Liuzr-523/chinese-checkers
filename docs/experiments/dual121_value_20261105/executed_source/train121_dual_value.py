"""路线图第3步：双权重固定预算诊断；只价值头与联合CNN两组。"""
import argparse,copy,fcntl,hashlib,json,os,shutil,time,traceback
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from az121 import Net121
from curriculum121 import atomic,batch
from cap_opening121_replay import load_data
from dual121_data import load_pilot,dual_loss
from value121_metrics import value_metrics,gate
from unfreeze121_trial import metrics as old_metrics
from checkpoint121_save import REGISTRY,digest
PILOT=Path('nn_checkpoints/experiments/expert121_pilot48_20261105')
OLD=Path('nn_checkpoints/experiments/fresh121_projection_20261105/data_sources')
ARMS=('head_only','joint')


def group_index(samples,weight):
    groups=defaultdict(list)
    for s in samples:
        if s['split']!='train' or float(s[weight])<=0:raise ValueError('采样池包含开发或无有效原始权重')
        groups[int(s['group'])].append(s)
    return dict(sorted(groups.items()))


def draw_group(groups,rng,count):
    keys=list(groups)
    if not keys:raise ValueError('空分组池')
    out=[]
    for _ in range(count):
        pool=groups[keys[int(rng.integers(len(keys)))]];out.append(pool[int(rng.integers(len(pool)))])
    return out


def draw_batch(value_groups,late_groups,policy_groups,rng):
    vv=draw_group(late_groups,rng,8)+draw_group(value_groups,rng,8);pp=draw_group(policy_groups,rng,16)
    return [{**s,'policy_weight':0.} for s in vv]+[{**s,'value_weight':0.} for s in pp],vv,pp


def freeze_arm(model,arm):
    if arm not in ARMS:raise ValueError(arm)
    for name,p in model.named_parameters():p.requires_grad_(arm=='joint' or name.startswith('value.'))


def nonvalue_hash(model):
    h=hashlib.sha256()
    for name,v in model.state_dict().items():
        if not name.startswith('value.'):h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def retention_kl(model,reference,samples):
    x,ids,mask,*_=batch(samples);logits,_=model(x);lp=logits.gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)
    with torch.no_grad():old,_=reference(x);oldlp=old.gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)
    return (oldlp.exp()*(oldlp-lp)).sum(1).mean()


@torch.no_grad()
def policy_metrics(model,samples):
    ce=hits=weight=0.;fingerprint=hashlib.sha256();was=model.training;model.eval()
    try:
        for start in range(0,len(samples),32):
            ss=[{**s,'value_weight':0.} for s in samples[start:start+32]];x,*_=batch(ss);l,v=model(x);loss=dual_loss(l,v,ss);pw=float(loss['policy_weight_sum']);ce+=float(loss['policy_ce'])*pw;weight+=pw
            for i,s in enumerate(ss):
                legal=l[i,s['actions']];fingerprint.update(legal.contiguous().numpy().tobytes());hits+=float(s['policy_weight'])*(int(legal.argmax())==int(s['target']))
    finally:model.train(was)
    return dict(n=len(samples),policy_ce=ce/max(weight,1),accuracy=hits/max(weight,1),legal_logits_sha256=fingerprint.hexdigest())


def brief(value):
    return {k:value[k] for k in ('n','groups','point','group_macro','late16') if k in value}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(1);torch.manual_seed(111071)
    a.out.mkdir(parents=True,exist_ok=False);lock=(a.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic();registry=digest(REGISTRY);entry=json.loads(REGISTRY.read_text());parent=Path(entry['checkpoint']);modelsha=digest(parent)
    if modelsha!=entry['sha256']:raise ValueError('原冠军SHA错误')
    def status(stage,**kw):
        atomic(a.out/'status.json',dict(roadmap_step=3,stage=stage,pid=os.getpid(),seconds=time.monotonic()-started,**kw))
        if time.monotonic()-started>3600 and stage not in ('complete','failed'):raise TimeoutError('一小时固定训练预算耗尽')
    def log(s):
        with (a.out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-started:.1f}秒：{s}\n')
    names=['train121_dual_value.py','dual121_data.py','value121_metrics.py','az121.py','rules121.py','curriculum121.py','cap_opening121_replay.py','canonical121_trial.py','cycle121_targeted.py','generalization121_curve.py','train_opening121.py','opening121.py','opening121_book.json','unfreeze121_trial.py','history121_exclusions.py','opening121_split_audit.py','tests/test_dual121_data.py','tests/test_value121_metrics.py','tests/test_train121_dual_value.py','diagnostics/plot121_dual_value.py']
    for name in names:
        dest=a.out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
    data=load_pilot(PILOT);_,old,normal,_=load_data(OLD);base=Net121.load(parent);base.requires_grad_(False);baseline_nonvalue=nonvalue_hash(base)
    vg=group_index(data['train']['value'],'value_weight');lg=group_index([s for s in data['train']['value'] if s['terminal_distance']<=16],'value_weight');pg=group_index(data['train']['policy'],'policy_weight')
    if set(vg)!=set(lg) or set(vg)!=set(pg) or len(vg)!=16:raise ValueError('训练16组/late/policy覆盖不符')
    data_hashes={str(PILOT/n):digest(PILOT/n) for n in ('dataset.pt','result.json','schedule.json','sample_metadata.json','policy_probes.json')};data_hashes.update({str(OLD/n):digest(OLD/n) for n in ('old_teacher.jsonl','old_split.json','normal_games.jsonl')})
    cfg=dict(roadmap_step=3,profile='CCZ-121-Strict-LG-v2',parent=str(parent),model_sha256=modelsha,registry_sha256=registry,source_sha256={n:digest(n) for n in names},data_sha256=data_hashes,arms=list(ARMS),steps=2000,lr=1e-4,weight_decay=1e-4,clip_norm=1,policy_weight=.25,kl_weight=.1,new_seed=111071,old_seed=111072,max_seconds=3600,
        sampling='按train开局组均匀再组内有放回抽样：8真实末16手value+8全程value；另16derived policy。另一头置0前先按原mask筛选。KL旧train8+普通train8，两组序列一致，记录重复率。',
        objectives='head_only仅value.*训练真实MSE；joint全网络valueMSE+.25独立归一策略CE+.1旧合法策略KL。两组同初值/优化器/更新/数据序列；参数范围和目标范围不同。',
        acceptance='新开发8组macro MSE较v3至少-.05且优于零值/训练玩家先验；末16手双方macro各不劣v3，合计优于零/先验；旧策略CE、旧价值MSE、普通CE各<=v3+.05；新策略CE<=v3+.05；head_only非value状态及策略logits逐值不变。两组通过按macro较低选一次，差<=1e-6优先head_only，只准备步骤4，不晋升。',
        development='开发不进入optimizer，不作中途评估；v3基线和固定最终2000各组仅评一次。自然胜负分布翻转保留，为压力诊断，非独立棋力验收。',pool_counts=data['pool_counts'])
    atomic(a.out/'config.json',cfg);atomic(a.out/'loader_manifest.json',data['manifest']);common_trace=[];audits={};final={};policy_invariance={}
    try:
        def evaluate(model):return dict(development=value_metrics(model,data['development']['value'],data['train']['value']),new_policy=policy_metrics(model,data['development']['policy']),retention=old_metrics(model,old['validation']),normal=old_metrics(model,normal['validation']))
        status('冻结基线和采样');before=evaluate(base);atomic(a.out/'baseline.json',before);log('数据双mask审查通过，冻结原v3基线，开始两组固定2000更新；开发不作中途评估。')
        for arm in ARMS:
            m=copy.deepcopy(base);freeze_arm(m,arm);m.train();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);rng=np.random.default_rng(111071);ro=np.random.default_rng(111072);seq=hashlib.sha256();curve=[];losses=[];vcount=Counter();pcount=Counter();vdups=pdups=0;clipped=0
            for step in range(2001):
                if step%1000==0:m.save(a.out/f'{arm}_step{step}.pt')
                if step%400==0:
                    train_value=value_metrics(m,data['train']['value'],data['train']['value']);curve.append(dict(step=step,value_train=brief(train_value),policy_train=policy_metrics(m,data['train']['policy'])));atomic(a.out/f'{arm}_curve.json',curve);status('双头价值对照训练',arm=arm,step=step,total=2000)
                    if step==2000:break
                ss,vv,pp=draw_batch(vg,lg,pg,rng);oi=ro.integers(len(old['train']),size=8);ni=ro.integers(len(normal['train']),size=8);ordinary=[old['train'][int(i)] for i in oi]+[normal['train'][int(i)] for i in ni];trace=dict(value=[s['sample_id'] for s in vv],policy=[s['sample_id'] for s in pp],old=oi.tolist(),normal=ni.tolist())
                raw=json.dumps(trace,sort_keys=True).encode();seq.update(raw)
                if arm=='head_only':common_trace.append(trace)
                elif trace!=common_trace[step]:raise ValueError('两组逐步抽样不一致')
                vcount.update(int(s['group']) for s in vv);pcount.update(int(s['group']) for s in pp);vdups+=16-len(set(trace['value']));pdups+=16-len(set(trace['policy']))
                x,*_=batch(ss);l,v=m(x);terms=dual_loss(l,v,ss);kl=retention_kl(m,base,ordinary)
                loss=terms['value_mse'] if arm=='head_only' else terms['value_mse']+.25*terms['policy_ce']+.1*kl
                if not torch.isfinite(loss):raise ValueError('非有限双头loss')
                opt.zero_grad();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True));clipped+=norm>1;opt.step()
                if step%100==0:losses.append(dict(step=step,value_mse=float(terms['value_mse'].detach()),policy_ce=float(terms['policy_ce'].detach()),old_kl=float(kl.detach()),preclip_norm=norm))
            policy_invariance[arm]=nonvalue_hash(m)==baseline_nonvalue
            if arm=='head_only' and not policy_invariance[arm]:raise ValueError('head_only改动了非value参数')
            audits[arm]=dict(sequence_sha256=seq.hexdigest(),value_draws=32000,policy_draws=32000,value_duplicate_draws=vdups,policy_duplicate_draws=pdups,value_group_exposure=dict(vcount),policy_group_exposure=dict(pcount),clipped_updates=clipped,final_sha256=digest(a.out/f'{arm}_step2000.pt'));atomic(a.out/'sampling_audit.json',audits);atomic(a.out/f'{arm}_losses.json',losses);log(arm+f'训练结束；最终训练value MSE {curve[-1]["value_train"]["point"]["mse"]:.4f}')
        atomic(a.out/'common_sample_sequence.json',common_trace)
        if len({r['sequence_sha256'] for r in audits.values()})!=1:raise ValueError('采样hash不一致')
        gates={};eligible=[]
        for arm in ARMS:
            status('固定终点开发评测',arm=arm);model=Net121.load(a.out/f'{arm}_step2000.pt');row=evaluate(model);final[arm]=row;passed,checks=gate(row,before);checks['new_policy']=row['new_policy']['policy_ce']<=before['new_policy']['policy_ce']+.05
            if arm=='head_only':checks['policy_unchanged']=policy_invariance[arm] and row['new_policy']['legal_logits_sha256']==before['new_policy']['legal_logits_sha256']
            passed=all(checks.values());gates[arm]=dict(passed=passed,checks=checks)
            if passed:eligible.append(arm)
            atomic(a.out/f'{arm}_final.json',row);log(arm+f'新开发group macro MSE {row["development"]["group_macro_mse"]:.4f}，门槛'+('通过' if passed else '未通过'))
        selected=None
        if eligible:
            selected=min(eligible,key=lambda arm:final[arm]['development']['group_macro_mse'])
            if 'head_only' in eligible and final['head_only']['development']['group_macro_mse']<=final[selected]['development']['group_macro_mse']+1e-6:selected='head_only'
        if digest(REGISTRY)!=registry or digest(parent)!=modelsha:raise ValueError('现役变化')
        if any(digest(name)!=sha for name,sha in data_hashes.items()):raise ValueError('冻结数据变化')
        atomic(a.out/'result.json',dict(roadmap_step=3,baseline=before,final=final,gates=gates,selected=selected,ready_for_step4=selected is not None,policy_nonvalue_unchanged=policy_invariance,promoted=False,arena_run=False,seconds=time.monotonic()-started));status('complete',selected=selected);log('本轮结束；'+('候选只准备步骤4，不登记冠军。' if selected else '未通过，不补调本配置，不登记冠军。'))
    except BaseException:status('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
