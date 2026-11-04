"""难例时间轨迹、分阶段梯度和隔离单步探针；原控制精确重放，不产生部署候选。"""
import copy,hashlib,json,os,sys,time,traceback,subprocess
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from az121 import Net121
from curriculum121 import batch,atomic,replay,samples_from_game
from canonical121_trial import canonical_sample
from corrected121_decisions import score_action,ConsistentExpert
from cycle121_targeted import expert_samples
from generalization121_curve import remove_validation_duplicates,input_digest
from diagnostics.fit121_objectives import objective_terms
from multicandidate121_trial import witness
from checkpoint121_save import digest,REGISTRY
FIT=Path('nn_checkpoints/experiments/fit121_objectives32_20261101')
ABL=Path('nn_checkpoints/experiments/ablate121_supervision32_20261101')
MULTI=Path('nn_checkpoints/experiments/multicandidate121_20261101')
OUT=Path('nn_checkpoints/experiments/diagnose121_hardroots_20261101')

def flatgrad(loss,params,retain=True):
 gs=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True)
 return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for g,p in zip(gs,params)])

def learning_history(records):
 good=[r['step'] for r in records if r['regret']==0];bad=[r['step'] for r in records if r['regret']>=4]
 return dict(first_recorded_good=min(good) if good else None,ever_recorded_good=bool(good),final_good=records[-1]['regret']==0,final_bad=records[-1]['regret']>=4,
  recorded_forgetting=any(a['regret']==0 and b['regret']>0 for a,b in zip(records,records[1:])),good_checkpoints=len(good),bad_checkpoints=len(bad))

def main():
 torch.set_num_threads(1);torch.manual_seed(111053);OUT.mkdir(exist_ok=False);began=time.monotonic();registry=digest(REGISTRY);config=json.loads((ABL/'config.json').read_text());ref=Net121.load(config['model']);ref.requires_grad_(False);ConsistentExpert.failure_dir=OUT
 def status(stage,**kw):
  atomic(OUT/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-began,**kw))
  if stage not in ('failed','complete') and time.monotonic()-began>3600:raise TimeoutError('一小时诊断预算耗尽')
 def log(s):
  with (OUT/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-began:.1f}秒：{s}\n')
 sources=['diagnostics/diagnose121_hardroots.py','diagnostics/plot121_hardroots.py','diagnostics/plot_style_zh.py','diagnostics/fit121_objectives.py','az121.py','curriculum121.py','canonical121_trial.py','corrected121_decisions.py','cycle121_targeted.py','generalization121_curve.py','multicandidate121_trial.py','unfreeze121_trial.py','rules121.py','engine_clysto121.py','engine_c121.py','checkpoint121_save.py']
 import clysto121_native
 sources +=[Path(clysto121_native.__file__).name,'clysto121_generated/core.c']
 for name in ('rules121.py','engine_clysto121.py','engine_c121.py',Path(clysto121_native.__file__).name,'clysto121_generated/core.c'):
  if digest(name)!=config['source_sha256'][name]:raise ValueError('固定评分实现变化')
 for name in sources:
  dst=OUT/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(Path(name).read_bytes())
 manifest=dict(scope='诊断原训练过程：精确重放控制1600步，100步采样实际决策；5阶段各8批梯度；最终8批各6种隔离单步AdamW探针，全部丢弃。不生成部署候选，不独立对战。',max_seconds=3600,max_new_expert_queries=256,source_sha256={s:digest(s) for s in sources},registry_sha256=registry,model_sha256=digest(config['model']),inputs={str(p):digest(p) for p in [ABL/'config.json',ABL/'data_sources/multicandidate.pt',ABL/'data_sources/decisions.jsonl',ABL/'data_sources/old_teacher.jsonl',ABL/'data_sources/normal_games.jsonl',ABL/'data_sources/old_split.json',ABL/'data_sources/fixed32.json',ABL/'control_step1600.pt',FIT/'root_policy_step1600.pt']})
 atomic(OUT/'config.json',manifest)
 try:
  data=ABL/'data_sources';sets=torch.load(data/'multicandidate.pt',weights_only=True)['sets'];fixed_ids=json.loads((data/'fixed32.json').read_text())['decision_ids'];lookup={s['decision_id']:s for s in sets['train']};roots=[lookup[i] for i in fixed_ids];meta={r['decision_id']:r for r in map(json.loads,(data/'decisions.jsonl').read_text().splitlines())}
  old={'train':[],'validation':[]};normal={'train':[],'validation':[]};split=json.loads((data/'old_split.json').read_text())
  for line in (data/'old_teacher.jsonl').read_text().splitlines():
   row=json.loads(line);part=next((p for p in old if row['group'] in split[p]),None)
   if part:old[part].extend({**canonical_sample(s),'value_weight':1.} for s in samples_from_game(row))
  for line in (data/'normal_games.jsonl').read_text().splitlines():
   row=json.loads(line);normal[row['split']].extend(expert_samples(row))
  for pool in (old,normal):pool['train'],_=remove_validation_duplicates(pool['train'],sets['validation'])
  ordinary=old['train']+normal['train'];byinput={}
  for s in ordinary:byinput.setdefault(input_digest(s),[]).append(s)
  overlaps=dict(exact_duplicate_roots=sum(input_digest(s) in byinput for s in roots),exact_conflicting_roots=sum(any(t['actions'][t['target']]!=s['actions'][s['target']] for t in byinput.get(input_digest(s),[])) for s in roots))
  cache={}
  for s in roots:
   for a,v in s['candidate_scores'].items():cache[s['decision_id'],int(a)]=v
  for line in (MULTI/'score_cache.jsonl').read_text().splitlines():
   row=json.loads(line);cache[row['decision_id'],row['action']]=row['score']
  for folder in (ABL,FIT):
   for path in folder.glob('*train*actions.json'):
    for row in json.loads(path.read_text()):cache[row['decision_id'],row['action']]=meta[row['decision_id']]['expert_score']-row['regret']
  queries=0
  def action_score(s,a):
   nonlocal queries
   key=(s['decision_id'],int(a))
   if key not in cache:
    if queries>=256:raise RuntimeError('新评分预算耗尽')
    status('轨迹动作评分',new_queries=queries);g=replay(meta[s['decision_id']]['opening']);cache[key]=score_action(g,witness(g,a));queries+=1
    with (OUT/'new_scores.jsonl').open('a') as f:f.write(json.dumps(dict(decision_id=s['decision_id'],action=a,score=cache[key]))+'\n')
   regret=meta[s['decision_id']]['expert_score']-cache[key]
   if regret<0:raise ValueError('评分超过原最优')
   return regret
  @torch.no_grad()
  def predict(model,step,ss=roots):
   xs,*_=batch(ss);logits,_=model(xs);rows=[]
   for i,s in enumerate(ss):
    a=s['actions'][int(logits[i,s['actions']].argmax())];good=s['actions'][s['target']];rows.append(dict(step=step,decision_id=s['decision_id'],action=a,regret=action_score(s,a),expert_logit_gap=float(logits[i,good]-logits[i,a])))
   return rows
  final_control=json.loads((ABL/'control_train32_actions.json').read_text());hard_ids=[s['decision_id'] for s in final_control if s['regret']>=4];bad_actions={s['decision_id']:s['action'] for s in final_control if s['regret']>=4};hard=[lookup[i] for i in hard_ids]
  def get_batch(rr,ro):
   ri=rr.integers(32,size=4);oi=ro.integers(len(old['train']),size=6);ni=ro.integers(len(normal['train']),size=6)
   return [roots[int(i)] for i in ri],[old['train'][int(i)] for i in oi]+[normal['train'][int(i)] for i in ni],ri.tobytes()+oi.tobytes()+ni.tobytes()
  def margins(model,ss=hard):
   xs,*_=batch(ss);logits,_=model(xs);return torch.stack([logits[i,s['actions'][s['target']]]-logits[i,bad_actions[s['decision_id']]] for i,s in enumerate(ss)])
  gradients=[]
  def probe_gradients(model,step):
   params=[p for n,p in model.named_parameters() if not n.startswith('value.')];mr=margins(model).mean();mg=flatgrad(mr,params,False);rr=np.random.default_rng(8801);ro=np.random.default_rng(8802)
   for i in range(8):
    rs,ordinary,_=get_batch(rr,ro);_,terms=objective_terms(model,ref,rs,ordinary,'full_objective');losses=dict(root=terms['root_ce']+terms['rank'],replay=terms['replay_ce'],aux=terms['value']+terms['kl']+terms['value_preservation']);gs={name:flatgrad(v,params) for name,v in losses.items()}
    a,b=gs['root'],gs['replay'];gradients.append(dict(step=step,batch=i,root_norm=float(a.norm()),replay_norm=float(b.norm()),aux_norm=float(gs['aux'].norm()),root_replay_cosine=float(F.cosine_similarity(a,b,dim=0)),**{name+'_gradient_descent_margin_direction':float(-torch.dot(mg,g)) for name,g in gs.items()}))
   atomic(OUT/'gradient_batches.json',gradients)
  model=copy.deepcopy(ref);model.requires_grad_(True);opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.0001);rr=np.random.default_rng(111054);ro=np.random.default_rng(111055);seq=hashlib.sha256();trajectory=[];exposures=Counter();checkpoint_audit=[]
  log('开始精确重放原控制，只添加100步实际首选记录及5阶段梯度审计，不产生晋升候选。')
  for step in range(1601):
   if step%100==0:
    trajectory.extend(predict(model,step));atomic(OUT/'control_trajectory.json',trajectory);status('精确训练重放诊断',step=step,total=1600)
    if step%400==0:
     expected=Net121.load(ABL/f'control_step{step}.pt');delta=max(float((v-model.state_dict()[k]).abs().max()) for k,v in expected.state_dict().items());checkpoint_audit.append(dict(step=step,max_parameter_delta=delta))
     if delta!=0:raise ValueError('重放未逐值复现原控制')
     probe_gradients(model,step)
    if step==1600:break
   rs,ordinary,bytes_=get_batch(rr,ro);seq.update(bytes_)
   for s in rs:exposures[s['decision_id']]+=1
   loss,_=objective_terms(model,ref,rs,ordinary,'full_objective');opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  if seq.hexdigest()!=json.loads((ABL/'sampling_audit.json').read_text())['control']['sequence_sha256']:raise ValueError('重放采样序列不符')
  atomic(OUT/'control_reproduction.json',dict(checkpoints=checkpoint_audit,sequence_sha256=seq.hexdigest(),exact=True));log('原控制所有400步检查点逐值复现，继续隔离单步AdamW探针。')
  # Independent one-step interventions all start at the SAME final model and optimizer state.
  probes=[];rr=np.random.default_rng(9901);ro=np.random.default_rng(9902);pre_margin=margins(model).detach();all_x,all_ids,all_mask,all_targets,_=batch(roots)
  @torch.no_grad()
  def policy_ce(m,ss):
   x,ids,mask,target,_=batch(ss);l,_=m(x);return float(F.cross_entropy(l.gather(1,ids).masked_fill(~mask,-1e9),target))
  # Fixed ordinary heldout slice for cost control; descriptive, not arena evidence.
  retention=old['validation'][:128];before_old=policy_ce(model,retention)
  for i in range(8):
   rs,ordinary,_=get_batch(rr,ro)
   for name in ('full','root_only','replay_only','aux_only','without_replay','without_root'):
    child=copy.deepcopy(model);child_opt=torch.optim.AdamW(child.parameters(),lr=3e-5,weight_decay=.0001);child_opt.load_state_dict(copy.deepcopy(opt.state_dict()));_,terms=objective_terms(child,ref,rs,ordinary,'full_objective');root=terms['root_ce']+terms['rank'];replay_loss=terms['replay_ce'];aux=terms['value']+terms['kl']+terms['value_preservation'];loss={'full':root+replay_loss+aux,'root_only':root,'replay_only':replay_loss,'aux_only':aux,'without_replay':root+aux,'without_root':replay_loss+aux}[name]
    child_opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(child.parameters(),1);child_opt.step();after=margins(child).detach();probes.append(dict(batch=i,component=name,hard_mean_margin_delta=float((after-pre_margin).mean()),hard_positive_count=int((after>pre_margin).sum()),per_hard_delta=[float(v) for v in after-pre_margin],old128_policy_ce_delta=policy_ce(child,retention)-before_old))
   status('隔离单步探针',completed=i+1,total=8)
  atomic(OUT/'adam_one_step_probes.json',probes)
  other_trajectories={}
  for name,folder,prefix in [('ties_ce_only',ABL,'ties_ce_only'),('best_rank_only',ABL,'best_rank_only'),('root_policy',FIT,'root_policy')]:
   rows=[]
   for step in (0,400,800,1200,1600):rows.extend(predict(Net121.load(folder/f'{prefix}_step{step}.pt'),step))
   other_trajectories[name]=rows
  atomic(OUT/'other_trajectories.json',other_trajectories)
  cases=[]
  ordinary_x=np.stack([s['x'][:2] for s in old['train']+normal['train']]);ordinary_samples=old['train']+normal['train'];ordinary_identity=np.array([float(np.asarray(s['x'])[3].max()) for s in ordinary_samples])
  for s in roots:
   did=s['decision_id'];rows=[v for v in trajectory if v['decision_id']==did];history=learning_history(rows);meta_row=meta[did];dist=(ordinary_x!=np.asarray(s['x'])[:2]).any(axis=1).sum(axis=(1,2));dist=np.where(ordinary_identity==float(np.asarray(s['x'])[3].max()),dist,10000);near=np.argsort(dist)[:5]
   cases.append(dict(decision_id=did,pair=meta_row['pair'],player=meta_row['player'],ply=meta_row['ply'],score_gap_original=meta_row['score_gap'],exposures=exposures[did],final_regret=rows[-1]['regret'],original_expert_path=meta_row['preferred'],final_actual_path=witness(replay(meta_row['opening']),rows[-1]['action']),learning=history,trajectory=rows,
     other_final_regret={name:next(r['regret'] for r in v if r['decision_id']==did and r['step']==1600) for name,v in other_trajectories.items()},nearest_ordinary=[dict(distance_changed_cells=int(dist[j]),target_action=ordinary_samples[int(j)]['actions'][ordinary_samples[int(j)]['target']]) for j in near]))
  atomic(OUT/'cases.json',cases)
  gradient_summary={}
  for step in (0,400,800,1200,1600):
   gs=[v for v in gradients if v['step']==step];gradient_summary[step]=dict(negative_batches=sum(v['root_replay_cosine']<0 for v in gs),mean_cosine=np.mean([v['root_replay_cosine'] for v in gs]),mean_replay_root_norm_ratio=np.mean([v['replay_norm']/v['root_norm'] for v in gs]),**{name+'_mean_margin_direction':float(np.mean([v[name+'_gradient_descent_margin_direction'] for v in gs])) for name in ('root','replay','aux')})
  probe_summary={name:dict(mean_margin_delta=float(np.mean([r['hard_mean_margin_delta'] for r in probes if r['component']==name])),positive_batches=sum(r['hard_mean_margin_delta']>0 for r in probes if r['component']==name),old_ce_delta=float(np.mean([r['old128_policy_ce_delta'] for r in probes if r['component']==name]))) for name in ('full','root_only','replay_only','aux_only','without_replay','without_root')}
  hardcases=[c for c in cases if c['decision_id'] in hard_ids];summary=dict(hard_ids=hard_ids,hard_never_recorded_good=[c['decision_id'] for c in hardcases if not c['learning']['ever_recorded_good']],hard_recorded_forgetting=[c['decision_id'] for c in hardcases if c['learning']['recorded_forgetting']],all_recorded_forgetting=[c['decision_id'] for c in cases if c['learning']['recorded_forgetting']],pure_root_final_good_for_hard=sum(c['other_final_regret']['root_policy']==0 for c in hardcases),hard_exposures=[c['exposures'] for c in hardcases],all_exposure_min=min(exposures.values()),all_exposure_max=max(exposures.values()),hard_groups=sorted(set(c['pair'] for c in hardcases)),gradient=gradient_summary,one_step=probe_summary,duplicates=overlaps,new_queries=queries,registry_unchanged=digest(REGISTRY)==registry,exact_control_reproduction=True)
  if not summary['registry_unchanged']:raise ValueError('冠军变动')
  if digest(config['model'])!=manifest['model_sha256']:raise ValueError('基线变动')
  atomic(OUT/'result.json',summary)
  labels={'full':'完整目标','root_only':'仅专项目标','replay_only':'仅普通策略回放','aux_only':'仅价值与正则','without_replay':'移除普通策略回放','without_root':'移除专项目标'}
  lines=['# 32根难例：训练过程与回放取舍诊断','', '## 范围与可靠性','原控制1600步精确重放，0/400/800/1200/1600参数逐值一致，根/普通样本序列哈希一致。没有修改冠军，没有生成新候选或独立对战。每100步检查实际首选的固定深度5专家分数，根仍无真实胜负标签。','', '## 难例逐个核查','|编号|开局组|走棋方|手数|专项抽样次数|首次记录到同最优分|记录到退步|最终分差|纯专项模型最终分差|','|---|---:|---:|---:|---:|---:|---|---:|---:|']
  for c in hardcases:lines.append(f'|{c["decision_id"]}|{c["pair"]}|{c["player"]}|{c["ply"]}|{c["exposures"]}|{c["learning"]["first_recorded_good"]}|{c["learning"]["recorded_forgetting"]}|{c["final_regret"]}|{c["other_final_regret"]["root_policy"]}|')
  lines+=['','100步采样可能漏掉两次记录之间的短暂学会/遗忘；“从未记录到”不能说整个过程中从未学会。开局组内相关，不能把32局面当32个独立开局。','', '## 分阶段专项与普通策略梯度','只比较共享CNN及策略参数，排除独立价值头参数；每阶段固定相同8批用于诊断。负余弦表示该处局部方向冲突，不等于整段训练因果证据。','|更新步|负余弦批次|平均余弦|普通/专项梯度范数比|','|---|---:|---:|---:|']
  for step,v in gradient_summary.items():lines.append(f'|{step}|{v["negative_batches"]}/8|{v["mean_cosine"]:.4f}|{v["mean_replay_root_norm_ratio"]:.3f}|')
  lines+=['','## 同起点、同AdamW状态的单步反事实探针','从原控制1600步状态分别复制模型及优化器，每种目标试同8个批次，每次都从相同起点开始，只走一步后丢弃。指标是7个难例上“专家首选logit减最终差棋logit”的平均变化；正数表示此局部指标改善。优化器动量保留，因此不是从零梯度下降，也不等于长期训练效果。旧能力只检查固定128条普通保留样本。','|单步目标|平均难例间隔变化|正向批次|旧保留策略CE变化|','|---|---:|---:|---:|']
  for name,v in probe_summary.items():lines.append(f'|{labels[name]}|{v["mean_margin_delta"]:.6f}|{v["positive_batches"]}/8|{v["old_ce_delta"]:.6f}|')
  lines+=['','## 数据重复与抽样排查',f'专项根与普通训练的完全相同输入：{overlaps["exact_duplicate_roots"]}；完全相同输入却有不同首选标签：{overlaps["exact_conflicting_roots"]}。所有根抽样范围{summary["all_exposure_min"]}至{summary["all_exposure_max"]}次，难例不是零次曝光。近邻棋盘仅供逐例查看，局面不同而走法不同不算标签矛盾。','', '## 结论边界','上述证据能检查回放取舍是否从早期至后期变化，并定位是哪些训练根；不能证明老师评分是真实最优，也不能把单步探针当作长期去掉回放能变强的证据。纯专项模型拟合能力与其旧能力严重退化必须一起考虑。后续任何修正都需先预声明有限开发对照，保住普通能力，使用新开局组验证；不继续反复使用旧40根作为独立验收。','详细原始数据：cases.json、control_trajectory.json、gradient_batches.json、adam_one_step_probes.json、control_reproduction.json。']
  (OUT/'REPORT.md').write_text('\n'.join(lines));plot=subprocess.run(['/Library/Frameworks/Python.framework/Versions/3.10/bin/python3','diagnostics/plot121_hardroots.py',str(OUT)],capture_output=True,text=True);atomic(OUT/'plot_status.json',dict(exit_code=plot.returncode,stderr=plot.stderr))
  if plot.returncode:raise RuntimeError('中文图表失败')
  log('难例、过程梯度和隔离单步探针完成；精确复现/冠军未变。');status('complete',summary=summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
 except BaseException:
  status('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
