"""固定32训练根：分别隔离同分CE和最优对差棋排序，不做部署。"""
import argparse,copy,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from az121 import Net121
from curriculum121 import atomic,batch,samples_from_game,replay
from canonical121_trial import canonical_sample
from corrected121_decisions import root_metrics,score_action,ConsistentExpert
from generalization121_curve import remove_validation_duplicates
from cycle121_targeted import expert_samples
from unfreeze121_trial import metrics
from checkpoint121_save import digest,REGISTRY,verify
from diagnostics.fit121_objectives import objective_terms as original_objective
from multicandidate121_trial import multi_loss,witness
PARENT=Path('nn_checkpoints/experiments/multicandidate121_20261101')
FIXED=Path('nn_checkpoints/experiments/fit121_objectives32_20261101')
ARMS=('control','ties_ce_only','best_rank_only')
LABELS={'control':'原方案','ties_ce_only':'仅接受同分好棋','best_rank_only':'仅最优棋对差棋排序'}

def best_rank(logits,samples):
 losses=[]
 for row,s in zip(logits,samples):
  scores=s['candidate_scores'];best=max(scores.values());good=[int(a) for a,v in scores.items() if v==best];bad=[int(a) for a,v in scores.items() if best-v>=4]
  terms=[F.relu(.5-row[a]+row[b]) for a in good for b in bad]
  losses.append(torch.stack(terms).mean() if terms else row.sum()*0)
 return torch.stack(losses).mean()

def objective_terms(model,reference,roots,ordinary,arm):
 if arm not in ARMS:raise ValueError('unknown arm')
 loss,terms=original_objective(model,reference,roots,ordinary,'full_objective')
 if arm=='control':return loss,terms
 xs,*_=batch(roots);logits,_=model(xs)
 if arm=='ties_ce_only':
  ce,_=multi_loss(logits,roots);loss=loss-terms['root_ce']+.25*ce;terms={**terms,'root_ce':.25*ce}
 else:
  rank=best_rank(logits,roots);loss=loss-terms['rank']+.25*rank;terms={**terms,'rank':.25*rank}
 return loss,terms

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(1);seed=111053;torch.manual_seed(seed)
 saved=verify(Path('nn_checkpoints/saves/v3_before_unfreeze_20261030'));source_config=json.loads((PARENT/'config.json').read_text());baseline_path=Path(source_config['model']);baseline=Net121.load(baseline_path);baseline.requires_grad_(False);registry=digest(REGISTRY)
 if registry!=saved['registry_sha256'] or digest(baseline_path)!=source_config['model_sha256']:raise ValueError('冠军/基线已变化')
 a.out.mkdir(parents=True,exist_ok=False);lock=(a.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic()
 def status(stage,**kw):
  atomic(a.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-started,**kw))
  if stage not in ('complete','failed') and time.monotonic()-started>3600:raise TimeoutError('一小时预算耗尽')
 def log(msg):
  with (a.out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-started:.1f}秒：{msg}\n')
 names=['diagnostics/ablate121_supervision32.py','diagnostics/plot121_ablation32.py','diagnostics/plot_style_zh.py','diagnostics/fit121_objectives.py','multicandidate121_trial.py','tests/test_ablate121_supervision32.py','az121.py','curriculum121.py','corrected121_decisions.py','generalization121_curve.py','canonical121_trial.py','cycle121_targeted.py','unfreeze121_trial.py','engine_clysto121.py','engine_c121.py','rules121.py','checkpoint121_save.py']
 import clysto121_native
 names+=[Path(clysto121_native.__file__).name,'clysto121_generated/core.c']
 for name in ('rules121.py','engine_clysto121.py','engine_c121.py',Path(clysto121_native.__file__).name,'clysto121_generated/core.c'):
  if digest(name)!=source_config['source_sha256'][name]:raise ValueError('固定老师来源已变化')
 for name in names:
  dst=a.out/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dst)
 data=a.out/'data_sources';data.mkdir();sources={name:PARENT/'data_sources'/name for name in ('multicandidate.pt','decisions.jsonl','old_teacher.jsonl','old_split.json','normal_games.jsonl')};sources.update({'fixed32.json':FIXED/'fixed32.json','prior_score_cache.jsonl':PARENT/'score_cache.jsonl'})
 for name,p in sources.items():shutil.copyfile(p,data/name)
 config=dict(profile=source_config['profile'],model=str(baseline_path.resolve()),model_sha256=digest(baseline_path),registry_sha256=registry,seed=seed,arms=list(ARMS),steps=1600,root_batch=4,ordinary_batch=12,learning_rate=.00003,max_seconds=3600,max_new_child_queries=256,
  scope='固定原32根/8训练开局组，单次因素拆分诊断。旧40开发验证已多次使用，不是新独立验收；不自动扩大训练、不对战、不登记。',
  loss='根CE .25+排序 .25；普通CE .75、真实value 1、KL .1、value函数保留 .1一致。control原唯一CE/原好坏对；ties_ce_only仅CE换同最优分集合；best_rank_only仅排序换最优集合对分差>=4动作。各局面排序均值权重一致。',
  control='相同CNN/v3初值/1600更新/根普通样本序列，哈希核验；不声称额外前向计算相同。固定最终1600，不选最优检查点。',
  source_sha256={n:digest(n) for n in names},data_sha256={n:digest(data/n) for n in sources})
 atomic(a.out/'config.json',config)
 try:
  status('准备固定训练数据');sets=torch.load(data/'multicandidate.pt',weights_only=True)['sets'];ids=json.loads((data/'fixed32.json').read_text())['decision_ids'];lookup={s['decision_id']:s for s in sets['train']};fixed=[lookup[i] for i in ids];holdout=sets['validation'];metadata={r['decision_id']:r for r in map(json.loads,(data/'decisions.jsonl').read_text().splitlines())}
  if len(fixed)!=32 or {metadata[s['decision_id']]['pair'] for s in fixed}&{metadata[s['decision_id']]['pair'] for s in holdout}:raise ValueError('固定数据/分组错误')
  old={'train':[],'validation':[]};normal={'train':[],'validation':[]};split=json.loads((data/'old_split.json').read_text())
  for line in (data/'old_teacher.jsonl').read_text().splitlines():
   row=json.loads(line);part=next((k for k in old if row['group'] in split[k]),None)
   if part:old[part].extend({**canonical_sample(s),'value_weight':1.} for s in samples_from_game(row))
  for line in (data/'normal_games.jsonl').read_text().splitlines():
   row=json.loads(line);normal[row['split']].extend(expert_samples(row))
  for pool in (old,normal):pool['train'],_=remove_validation_duplicates(pool['train'],holdout)
  def measure(model):return dict(train32=root_metrics(model,fixed),validation40=root_metrics(model,holdout),retention=metrics(model,old['validation']),normal=metrics(model,normal['validation']))
  before=measure(baseline);atomic(a.out/'baseline.json',before);final={};audits={};log('开始固定32根三组拆分对照；普通训练/真实价值/正则保持相同，不运行独立对战或更改冠军。')
  for arm in ARMS:
   model=copy.deepcopy(baseline);model.requires_grad_(True);opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.0001);rr=np.random.default_rng(seed+1);ro=np.random.default_rng(seed+2);seq=hashlib.sha256();history=[]
   for step in range(1601):
    if step%100==0:
     history.append(dict(step=step,**measure(model)));atomic(a.out/(arm+'_curve.json'),history);status('拆分监督训练',arm=arm,step=step,total=1600)
     if step%400==0:model.save(a.out/(arm+f'_step{step}.pt'))
     if step==1600:break
    ri=rr.integers(32,size=4);oi=ro.integers(len(old['train']),size=6);ni=ro.integers(len(normal['train']),size=6);seq.update(ri.tobytes()+oi.tobytes()+ni.tobytes());roots=[fixed[int(i)] for i in ri];ordinary=[old['train'][int(i)] for i in oi]+[normal['train'][int(i)] for i in ni]
    loss,_=objective_terms(model,baseline,roots,ordinary,arm)
    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
   final[arm]=history[-1];audits[arm]=dict(sequence_sha256=seq.hexdigest(),root_exposures=6400,ordinary_exposures=19200,updates=1600);atomic(a.out/'sampling_audit.json',audits);log(LABELS[arm]+'完成1600步')
  if len({x['sequence_sha256'] for x in audits.values()})!=1:raise ValueError('对照采样序列不同')
  # Validate the control reproduces its original saved weights, not just metrics.
  reference=Net121.load(FIXED/'full_objective_step1600.pt');control=Net121.load(a.out/'control_step1600.pt');delta=max(float((v-control.state_dict()[k]).abs().max()) for k,v in reference.state_dict().items());atomic(a.out/'control_reproduction.json',dict(max_parameter_abs_difference=delta,exact=delta==0))
  if delta>1e-6:raise ValueError('原控制组无法复现')
  scores={};queries=0;ConsistentExpert.failure_dir=a.out
  for s in fixed+holdout:
   for action,value in s['candidate_scores'].items():scores[s['decision_id'],int(action)]=value
  for line in (data/'prior_score_cache.jsonl').read_text().splitlines():
   row=json.loads(line);scores[row['decision_id'],row['action']]=row['score']
  for arm in ARMS:
   model=Net121.load(a.out/(arm+'_step1600.pt'))
   for part,ss in [('train32',fixed),('validation40',holdout)]:
    rows=[]
    for s in ss:
     with torch.no_grad():xs,*_=batch([s]);logits,_=model(xs);action=s['actions'][int(logits[0,s['actions']].argmax())]
     key=s['decision_id'],int(action)
     if key not in scores:
      if queries>=256:raise RuntimeError('评分预算耗尽')
      status('实际动作评分',arm=arm,completed=len(rows),new_queries=queries);g=replay(metadata[s['decision_id']]['opening']);scores[key]=score_action(g,witness(g,action));queries+=1
     regret=metadata[s['decision_id']]['expert_score']-scores[key]
     if regret<0:raise ValueError('强制动作超出固定老师最优')
     rows.append(dict(decision_id=s['decision_id'],action=action,regret=regret))
    atomic(a.out/(arm+'_'+part+'_actions.json'),rows);final[arm][part+'_actual']=dict(n=len(rows),equal_best=sum(r['regret']==0 for r in rows),bad_moves=sum(r['regret']>=4 for r in rows),mean_regret=sum(r['regret'] for r in rows)/len(rows))
  if digest(REGISTRY)!=registry or digest(baseline_path)!=config['model_sha256']:raise ValueError('登记或基线改变')
  result=dict(final=final,baseline=before,new_child_queries=queries,promoted=False,arena_run=False,seconds=time.monotonic()-started,control_max_parameter_delta=delta);atomic(a.out/'result.json',result)
  plot=subprocess.run(['/Library/Frameworks/Python.framework/Versions/3.10/bin/python3','diagnostics/plot121_ablation32.py',str(a.out)],capture_output=True,text=True);atomic(a.out/'plot_status.json',dict(exit_code=plot.returncode,stderr=plot.stderr))
  if plot.returncode:raise RuntimeError('中文图表渲染失败')
  lines=['# 固定32根监督因素拆分','', '三组从同v3开始，相同根/普通样本序列和1600更新。唯一因素分别为同分集合交叉熵、最优集合对差棋排序；不增加训练数据、不进行棋力验收。','|方案|训练同最优分|训练差棋|训练平均遗憾|旧40验证同最优分|旧40验证差棋|旧策略交叉熵|','|---|---:|---:|---:|---:|---:|---:|']
  for arm,v in final.items():
   t=v['train32_actual'];h=v['validation40_actual'];lines.append(f'|{LABELS[arm]}|{t["equal_best"]}/32|{t["bad_moves"]}/32|{t["mean_regret"]:.3f}|{h["equal_best"]}/40|{h["bad_moves"]}/40|{v["retention"]["policy_ce"]:.3f}|')
  lines+=['','原控制参数最大差异：'+str(delta)+'。固定32局面中存在同分动作，准确率外另按固定专家评分核对实际top1。','原40开发验证已反复使用，不能称全新独立验证。结果用于下一步设计，不依据此直接登记或宣称棋力提升。','后续若某因素改善训练实际好棋且保住旧能力，需用160训练和全新开局组开发数据复核；若都不改善，转向目标/采样与共享表示诊断，不重复扩大同配方。','中文曲线：监督拆分曲线.png；所有检查点/动作评分/采样哈希/源码/中文日志保存。']
  (a.out/'REPORT.md').write_text('\n'.join(lines));log('拆分实验完成；未升级、未启动更大训练，需先分析结果。');status('complete',result=result)
 except BaseException:
  status('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
