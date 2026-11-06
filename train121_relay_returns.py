"""路线图第3步：真实续弈动作偏好固定训练；不自动晋升/竞技验收。"""
import argparse,copy,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import rules121 as r
from az121 import Net121
from curriculum121 import atomic
from cap_opening121_replay import load_data
from checkpoint121_save import REGISTRY,digest
from order121_learning import configure_policy_training,prepare_samples,order_batch,paired_policy_loss,retention_terms,value_state_hash
from relay121_metrics import prepare_rows,relay_metrics,relay_gate
OLD=Path('nn_checkpoints/experiments/fresh121_projection_20261105/data_sources')


def freeze_collection(folder):
    return {str(p):digest(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='run.lock'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);lock=(a.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(111092)
    import ccore121,clysto121_native
    native={str(Path(m.__file__)):digest(m.__file__) for m in (ccore121,clysto121_native)};registry=digest(REGISTRY);entry=json.loads(REGISTRY.read_text());parent=Path(entry['checkpoint']);parent_sha=digest(parent)
    if parent_sha!=entry['sha256']:raise ValueError('冠军检查点SHA错误')
    if entry.get('version')!='v3_expert_imitation':raise ValueError('本实验只允许已预声明的原v3初值')
    def status(stage,**kw):
        atomic(a.out/'status.json',dict(roadmap_step=3,stage=stage,pid=os.getpid(),seconds=time.monotonic()-started,**kw))
        if stage not in ('failed','complete') and time.monotonic()-started>5400:raise TimeoutError('90分钟总预算耗尽')
    def log(message):
        with (a.out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-started:.1f}秒：{message}\n')
    names=['train121_relay_returns.py','collect121_relay_returns.py','collect121_order_pairs.py','relay121_targets.py','relay121_metrics.py','order121_learning.py','az121.py','rules121.py','curriculum121.py','corrected121_decisions.py','engine_clysto121.py','engine_c121.py','expert121_pilot48.py','history121_exclusions.py','opening121_split_audit.py','opening121.py','opening121_book.json','cap_opening121_replay.py','canonical121_trial.py','cycle121_targeted.py','generalization121_curve.py','train_opening121.py','unfreeze121_trial.py','checkpoint121_save.py','tests/test_relay121_collection.py','tests/test_relay121_targets.py','tests/test_relay121_metrics.py','docs/experiments/relay121_returns_20261106/protocol.md','diagnostics/relay121_historical_regression.py']
    for name in names:
        dest=a.out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
    cfg=dict(roadmap_step=3,profile=r.PROFILE,model=str(parent),model_sha256=parent_sha,registry_sha256=registry,native_sha256=native,source_sha256={n:digest(n) for n in names},old_data_sha256={str(OLD/n):digest(OLD/n) for n in ('old_teacher.jsonl','old_split.json','normal_games.jsonl')},new_seed=111092,old_seed=111093,steps=2000,lr=1e-4,weight_decay=1e-4,clip=1,collection_max_seconds=3600,training_max_seconds=1800,total_max_seconds=5400,
        objective='同根rear/front softplus偏好 + .5旧合法策略KL(v3) + .1旧value输出保持MSE(v3)。全CNN及策略头可训，value.*参数冻结但共享CNN仍改变value输出；标签是相对策略偏好，不是真胜负value标签。',
        sampling='有效train组均匀有放回每批16根，每组一根；旧train8+普通train8，不按标签方向重采样。固定2000更新，保存0/1000/2000，400步仅评train。',
        evaluation='新开发v3基线一次与固定最终候选一次；开发覆盖不足不阻止train就绪后的训练，但验收失败。只准备后续搜索诊断，不自动晋升或竞技验收；历史7例仅固定final后回归，不进入训练或选择。',
        acceptance='relay_gate按执行源码冻结专项开发覆盖、balanced accuracy、双方向准确率、pair CE与两续弈配置的point及适用tempo regret均不劣门槛（无适用tempo记N/A）；另旧策略CE/旧valueMSE/普通CE各<=v3+.05，value头参数哈希必须不变。',
        readiness='仅训练覆盖决定可否启动：有效train>=8，正负方向各>=2；开发覆盖独立影响验收，不据开发结果补根、重分或重训。')
    atomic(a.out/'config.json',cfg)
    def guards(collection_hashes=None):
        if digest(REGISTRY)!=registry or digest(parent)!=parent_sha:raise ValueError('冠军/注册表变化')
        for hashes in (cfg['native_sha256'],cfg['source_sha256'],cfg['old_data_sha256'],collection_hashes or {}):
            if any(not Path(name).is_file() or digest(name)!=sha for name,sha in hashes.items()):raise ValueError('冻结源码/native/输入数据变化')
    try:
        guards();status('收集相对后排接力标签');log('配置与代码哈希冻结，先收集48根（24train/24development），尚未训练。')
        with (a.out/'collection.log').open('w') as stream:
            subprocess.run([sys.executable,'-u','collect121_relay_returns.py','--out',str(a.out/'collection'),'--exclude-parent'],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=3700)
        collection=a.out/'collection';raw=json.loads((collection/'dataset.json').read_text());collection_hashes=freeze_collection(collection);atomic(a.out/'collection_manifest.json',collection_hashes);guards(collection_hashes)
        if raw['profile']!=r.PROFILE:raise ValueError('收集数据规则不匹配')
        prepared=prepare_rows(raw['rows']);train=[s for s in prepared if s['split']=='train'];dev=[s for s in prepared if s['split']=='development']
        if any(s['split'] not in ('train','development') for s in prepared):raise ValueError('无效split')
        tg={s['group'] for s in train};dg={s['group'] for s in dev}
        if len(tg)!=len(train) or len(dg)!=len(dev) or tg&dg:raise ValueError('根组重复/跨split')
        directions=Counter(s['label'] for s in train);training_ready=len(train)>=8 and min(directions[1],directions[-1])>=2
        if bool(raw['readiness']['training'])!=training_ready:raise ValueError('收集器与runner训练覆盖定义不符')
        freeze=dict(dataset_sha256=digest(collection/'dataset.json'),collection_manifest_sha256=digest(a.out/'collection_manifest.json'),train_groups=sorted(tg),development_groups=sorted(dg),train_directions=dict(directions),development_directions=dict(Counter(s['label'] for s in dev)),readiness=raw['readiness']);atomic(a.out/'dataset_freeze.json',freeze)
        if not training_ready:
            guards(collection_hashes);atomic(a.out/'result.json',dict(roadmap_step=3,training_started=False,readiness=raw['readiness'],development_pass=False,promoted=False,arena_run=False,ready_for_search_probe=False,reason='train覆盖不足，固定数据收集结束，不追加根或重分。',seconds=time.monotonic()-started));status('complete',training_started=False);log('训练覆盖不足，本轮仅数据结束；未训练。');return
        train=prepare_samples(train);_,old,normal,_=load_data(OLD);base=Net121.load(parent);base.requires_grad_(False);initial_value=value_state_hash(base)
        def measure(model):
            was=model.training;flags=[p.requires_grad for p in model.parameters()];model.eval();model.requires_grad_(False)
            try:return dict(order=relay_metrics(model,dev),retention=old_metrics(model,old['validation']),normal=old_metrics(model,normal['validation']))
            finally:
                model.train(was)
                for p,flag in zip(model.parameters(),flags):p.requires_grad_(flag)
        status('冻结v3新开发基线');before=measure(base);atomic(a.out/'baseline.json',before);model=copy.deepcopy(base);configure_policy_training(model);model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);rr=np.random.default_rng(111092);ro=np.random.default_rng(111093);trace=[];curve=[];losses=[];exposure=Counter();duplicates=0;clipped=0;train_started=time.monotonic();log(f'训练就绪：有效train {len(train)}根；有效dev {len(dev)}根，开发不足仍训练、只影响验收。')
        for step in range(2001):
            if step%1000==0:model.save(a.out/f'candidate_step{step}.pt')
            if step%400==0:
                curve.append(dict(step=step,train=relay_metrics(model,train)));atomic(a.out/'curve.json',curve);atomic(a.out/'losses.json',losses);atomic(a.out/'sample_trace.json',trace);status('相对接力专项训练',step=step,total=2000)
                if time.monotonic()-train_started>1800:raise TimeoutError('30分钟训练预算耗尽')
                if step==2000:break
            ri=rr.integers(len(train),size=16);oi=ro.integers(len(old['train']),size=8);ni=ro.integers(len(normal['train']),size=8);pairs=[train[int(i)] for i in ri];ordinary=[old['train'][int(i)] for i in oi]+[normal['train'][int(i)] for i in ni];groups=[s['group'] for s in pairs];trace.append(dict(groups=groups,old=oi.tolist(),normal=ni.tolist()));exposure.update(groups);duplicates+=16-len(set(groups))
            logits,_=model(order_batch(pairs));pair_loss=paired_policy_loss(logits,pairs);terms=retention_terms(model,base,ordinary);loss=pair_loss+.5*terms['policy_kl']+.1*terms['value_preservation']
            if not torch.isfinite(loss):raise ValueError('非有限loss')
            opt.zero_grad();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True));clipped+=norm>1;opt.step()
            if step%100==0:losses.append(dict(step=step,pair_loss=float(pair_loss.detach()),policy_kl=float(terms['policy_kl'].detach()),value_preservation=float(terms['value_preservation'].detach()),preclip_norm=norm))
        atomic(a.out/'sampling_audit.json',dict(sequence_sha256=hashlib.sha256(json.dumps(trace,sort_keys=True).encode()).hexdigest(),groups=dict(exposure),draws=32000,within_batch_duplicate_draws=duplicates,duplicate_fraction=duplicates/32000,clipped_updates=clipped));unchanged=value_state_hash(model)==initial_value
        if not unchanged:raise ValueError('value头参数变化')
        status('固定终点新开发评测');final_model=Net121.load(a.out/'candidate_step2000.pt');after=measure(final_model);_,checks=relay_gate(after['order'],before['order']);checks.update(old_policy=after['retention']['policy_ce']<=before['retention']['policy_ce']+.05,old_value=after['retention']['value_mse']<=before['retention']['value_mse']+.05,normal_policy=after['normal']['policy_ce']<=before['normal']['policy_ce']+.05,value_parameters_unchanged=value_state_hash(final_model)==initial_value);passed=all(checks.values());guards(collection_hashes)
        atomic(a.out/'final.json',after);atomic(a.out/'result.json',dict(roadmap_step=3,training_started=True,readiness=raw['readiness'],before=before,after=after,checks=checks,development_pass=passed,ready_for_search_probe=passed,promoted=False,arena_run=False,value_parameters_unchanged=unchanged,shared_cnn_can_change_value_outputs=True,checkpoint_sha256=digest(a.out/'candidate_step2000.pt'),seconds=time.monotonic()-started));log('专项结束；'+('通过开发门槛，仅可规划后续搜索探针。' if passed else '开发门槛未通过，不补调、不晋升。'));status('complete',development_pass=passed)
    except BaseException:status('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
