"""落子次序专项：先冻结新根与标签，再一次固定CNN策略训练；不晋升。"""
import argparse,copy,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from az121 import Net121
from curriculum121 import atomic
from cap_opening121_replay import load_data
from checkpoint121_save import REGISTRY,digest
from order121_learning import configure_policy_training,paired_policy_loss,retention_terms,value_state_hash,order_batch,prepare_samples
from order121_metrics import prepare_rows,order_metrics,order_gate
from unfreeze121_trial import metrics as old_metrics
OLD=Path('nn_checkpoints/experiments/fresh121_projection_20261105/data_sources')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);lock=(a.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(111082)
    import ccore121,clysto121_native
    native_sha={str(Path(mod.__file__)):digest(mod.__file__) for mod in (ccore121,clysto121_native)}
    registry=digest(REGISTRY);entry=json.loads(REGISTRY.read_text());parent=Path(entry['checkpoint']);parent_sha=digest(parent)
    if parent_sha!=entry['sha256']:raise ValueError('原冠军SHA不匹配')
    def status(stage,**kw):
        atomic(a.out/'status.json',dict(roadmap_step=3,stage=stage,pid=os.getpid(),seconds=time.monotonic()-started,**kw))
        if time.monotonic()-started>5400 and stage not in ('complete','failed'):raise TimeoutError('本轮90分钟预算耗尽')
    def log(msg):
        with (a.out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-started:.1f}秒：{msg}\n')
    sources=['train121_order_trial.py','collect121_order_pairs.py','order121_learning.py','order121_metrics.py','az121.py','rules121.py','curriculum121.py','corrected121_decisions.py','engine_clysto121.py','engine_c121.py','expert121_pilot48.py','history121_exclusions.py','opening121_split_audit.py','opening121.py','cap_opening121_replay.py','canonical121_trial.py','cycle121_targeted.py','generalization121_curve.py','unfreeze121_trial.py','checkpoint121_save.py','tests/test_order121_collection.py','tests/test_order121_learning.py','tests/test_order121_metrics.py']
    for name in sources:
        dest=a.out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
    cfg=dict(roadmap_step=3,profile='CCZ-121-Strict-LG-v2',model=str(parent),model_sha256=parent_sha,registry_sha256=registry,native_sha256=native_sha,source_sha256={n:digest(n) for n in sources},old_data_sha256={str(OLD/n):digest(OLD/n) for n in ('old_teacher.jsonl','old_split.json','normal_games.jsonl')},new_seed=111082,old_seed=111083,steps=2000,lr=1e-4,weight_decay=1e-4,clip=1,collection_max_seconds=3600,training_max_seconds=1800,total_max_seconds=5400,
        objective='同根动作pair softplus(-label*(logit_rear-logit_front)) + .5旧合法策略KL(v3) + .1旧value输出保留MSE(v3)，全共享CNN/policy可训，value.*参数冻结。不是整局胜负训练。',
        sampling='每批16根，从所有有效train组均匀有放回抽样（每组一根），旧train8+普通train8；不按方向重采样，不使用开发。',
        evaluation='v3开发基线一次，固定2000步最终一次；每400只train。无开发择点、无参数扫描。',
        gates='数据：有效train>=16/dev>=8，train每方向>=4/dev>=2。候选：开发balanced_accuracy>=.75且>=v3+.125，两方向各>=.60，pairCE<=v3，平均teacher分差regret<=v3；旧CE/旧valueMSE/普通CE各<=v3+.05；value参数逐值不变。通过只准备后续搜索验证，不登记冠军。')
    atomic(a.out/'config.json',cfg)
    try:
        status('生成与冻结48个新根');log('单配置预声明已冻结，先收集新动作对；没有开始训练。')
        with (a.out/'collection.log').open('w') as stream:
            subprocess.run([sys.executable,'-u','collect121_order_pairs.py','--out',str(a.out/'collection'),'--exclude-parent'],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=3700)
        collection=a.out/'collection';dataset=collection/'dataset.json';raw=json.loads(dataset.read_text());frozen_sha=digest(dataset)
        if not raw['data_quality_pass']:
            atomic(a.out/'result.json',dict(roadmap_step=3,data_quality_pass=False,data_gates=raw['gates'],training_started=False,promoted=False,reason='固定数据覆盖门槛未过，不追加根、不重分、不训练',seconds=time.monotonic()-started));status('complete',training_started=False);log('数据门槛未过，保留全部原记录，本配置停止。');return
        prepared=prepare_rows(raw['rows']);train=prepare_samples([s for s in prepared if s['split']=='train']);dev=[s for s in prepared if s['split']=='development'];assert len({s['group'] for s in train})==len(train) and len({s['group'] for s in dev})==len(dev)
        atomic(a.out/'dataset_freeze.json',dict(dataset_sha256=frozen_sha,train_groups=[s['group'] for s in train],development_groups=[s['group'] for s in dev],train_directions=dict(Counter(s['label'] for s in train)),development_directions=dict(Counter(s['label'] for s in dev))))
        _,old,normal,_=load_data(OLD);base=Net121.load(parent);base.requires_grad_(False);initial_value=value_state_hash(base)
        def measure(m):
            m.eval();flags=[p.requires_grad for p in m.parameters()];m.requires_grad_(False)
            try:return dict(order=order_metrics(m,dev),retention=old_metrics(m,old['validation']),normal=old_metrics(m,normal['validation']))
            finally:
                for p,flag in zip(m.parameters(),flags):p.requires_grad_(flag)
        status('冻结开发基线');before=measure(base);atomic(a.out/'baseline.json',before);m=copy.deepcopy(base);configure_policy_training(m);m.train();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);rr=np.random.default_rng(111082);ro=np.random.default_rng(111083);curve=[];losses=[];trace=[];exposure=Counter();clipped=0;train_started=time.monotonic();log(f'新根标签冻结：train {len(train)}组/dev {len(dev)}组；开始固定2000更新。')
        for step in range(2001):
            if step%1000==0:m.save(a.out/f'candidate_step{step}.pt')
            if step%400==0:
                curve.append(dict(step=step,train=order_metrics(m,train)));atomic(a.out/'curve.json',curve);m.train();status('专项CNN训练',step=step,total=2000)
                if time.monotonic()-train_started>1800:raise TimeoutError('30分钟训练预算耗尽')
                if step==2000:break
            ri=rr.integers(len(train),size=16);oi=ro.integers(len(old['train']),size=8);ni=ro.integers(len(normal['train']),size=8);pairs=[train[int(i)] for i in ri];ordinary=[old['train'][int(i)] for i in oi]+[normal['train'][int(i)] for i in ni];trace.append(dict(groups=[s['group'] for s in pairs],old=oi.tolist(),normal=ni.tolist()));exposure.update(s['group'] for s in pairs)
            logits,_=m(order_batch(pairs));pair=paired_policy_loss(logits,pairs);terms=retention_terms(m,base,ordinary);loss=pair+.5*terms['policy_kl']+.1*terms['value_preservation']
            if not torch.isfinite(loss):raise ValueError('非有限训练loss')
            opt.zero_grad();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True));clipped+=norm>1;opt.step()
            if step%100==0:losses.append(dict(step=step,pair_loss=float(pair.detach()),policy_kl=float(terms['policy_kl'].detach()),value_preservation=float(terms['value_preservation'].detach()),preclip_norm=norm))
        atomic(a.out/'sample_trace.json',trace);atomic(a.out/'losses.json',losses);atomic(a.out/'sampling_audit.json',dict(sequence_sha256=hashlib.sha256(json.dumps(trace,sort_keys=True).encode()).hexdigest(),groups=dict(exposure),draws=32000,clipped_updates=clipped));assert value_state_hash(m)==initial_value
        status('固定终点开发评测');final_model=Net121.load(a.out/'candidate_step2000.pt');after=measure(final_model);_,checks=order_gate(after['order'],before['order']);checks.update(old_policy=after['retention']['policy_ce']<=before['retention']['policy_ce']+.05,old_value=after['retention']['value_mse']<=before['retention']['value_mse']+.05,normal_policy=after['normal']['policy_ce']<=before['normal']['policy_ce']+.05,value_parameters_unchanged=value_state_hash(final_model)==initial_value);passed=all(checks.values())
        if digest(REGISTRY)!=registry or digest(parent)!=parent_sha or digest(dataset)!=frozen_sha:raise ValueError('冠军/数据冻结SHA变化')
        if any(digest(n)!=sha for n,sha in cfg['native_sha256'].items()):raise ValueError('native内核变化')
        if any(digest(n)!=sha for n,sha in cfg['old_data_sha256'].items()):raise ValueError('旧输入数据变化')
        atomic(a.out/'final.json',after);atomic(a.out/'result.json',dict(roadmap_step=3,data_quality_pass=True,training_started=True,before=before,after=after,checks=checks,development_pass=passed,ready_for_search_probe=passed,promoted=False,arena_run=False,value_parameters_unchanged=True,shared_cnn_changes_value_outputs=True,checkpoint_sha256=digest(a.out/'candidate_step2000.pt'),seconds=time.monotonic()-started));log('固定终点评测'+('通过专项开发门槛，仅可安排搜索探针。' if passed else '未通过专项开发门槛，不补调，不晋升。'));status('complete',development_pass=passed)
    except BaseException:status('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
