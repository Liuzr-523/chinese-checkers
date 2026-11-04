"""只审查既有多候选实验；不训练、不新评分、不改登记。"""
import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from az121 import Net121
from curriculum121 import batch,atomic
from corrected121_decisions import preference_loss
from multicandidate121_trial import multi_loss,SEEDS
from checkpoint121_save import digest,REGISTRY

RUN=Path('nn_checkpoints/experiments/multicandidate121_20261101')
OUT=Path('nn_checkpoints/experiments/diagnose121_multicandidate_20261101')
def mean(xs):return sum(xs)/len(xs) if xs else None

def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=False);sha=digest(REGISTRY);config=json.loads((RUN/'config.json').read_text());sets=torch.load(RUN/'data_sources/multicandidate.pt',weights_only=True)['sets'];model=Net121.load(config['model']);rows=[]
 for part,samples in sets.items():
  for s in samples:
   scores=s['candidate_scores'];best=max(scores.values());good={int(a) for a,v in scores.items() if v==best};pairs=[(int(a),int(b)) for a,va in scores.items() for b,vb in scores.items() if va-vb>=4];preferred=s['actions'][s['target']];rejected=s['rejected_action']
   with torch.no_grad():xs,*_=batch([s]);logits,_=model(xs)
   logits=logits.detach().requires_grad_(True);legal=logits[:,s['actions']];old_ce=F.cross_entropy(legal,torch.tensor([s['target']]));old_rank=preference_loss(logits,[s]);new_ce,new_rank=multi_loss(logits,[s])
   old_grad=torch.autograd.grad(.25*(old_ce+old_rank),logits,retain_graph=True)[0][0];new_grad=torch.autograd.grad(.25*(new_ce+new_rank),logits,retain_graph=True)[0][0]
   new_rank_grad=torch.autograd.grad(new_rank,logits)[0][0]
   rows.append(dict(decision_id=s['decision_id'],split=part,legal_count=len(s['actions']),candidate_count=len(scores),equal_best_count=len(good),pair_count=len(pairs),best_pairs=sum(a in good for a,b in pairs),inferior_pairs=sum(a not in good for a,b in pairs),
       original_pair_present=(preferred,rejected) in pairs,original_pair_weight=1/len(pairs) if pairs else 0.,best_set_grad_old=float(old_grad[list(good)].sum()),best_set_grad_new=float(new_grad[list(good)].sum()),
       rank_promoted_inferior=sum(float(new_rank_grad[int(a)])<0 for a,v in scores.items() if v<best),root_grad_norm_old=float(old_grad.norm()),root_grad_norm_new=float(new_grad.norm()),gradient_cosine=float(F.cosine_similarity(old_grad,new_grad,dim=0)),
       old_rank=float(old_rank.detach()),new_rank=float(new_rank.detach()),old_ce=float(old_ce.detach()),new_ce=float(new_ce.detach())))
 summary={}
 for part in sets:
  rs=[r for r in rows if r['split']==part];pairs=sum(r['pair_count'] for r in rs);oldnorm=mean([r['root_grad_norm_old'] for r in rs]);newnorm=mean([r['root_grad_norm_new'] for r in rs])
  summary[part]=dict(n=len(rs),mean_candidates=mean([r['candidate_count'] for r in rs]),multiple_equal_best=sum(r['equal_best_count']>1 for r in rs),mean_pairs=mean([r['pair_count'] for r in rs]),inferior_comparison_fraction=sum(r['inferior_pairs'] for r in rs)/max(pairs,1),
       original_pair_present=sum(r['original_pair_present'] for r in rs),mean_original_pair_weight=mean([r['original_pair_weight'] for r in rs]),states_rank_promotes_inferior=sum(r['rank_promoted_inferior']>0 for r in rs),mean_old_root_gradient_norm=oldnorm,mean_new_root_gradient_norm=newnorm,new_to_old_mean_gradient_norm=newnorm/oldnorm,
       mean_gradient_cosine=mean([r['gradient_cosine'] for r in rs]),mean_best_set_downhill_old=-mean([r['best_set_grad_old'] for r in rs]),mean_best_set_downhill_new=-mean([r['best_set_grad_new'] for r in rs]))
 labels={s['decision_id']:s['candidate_scores'] for ss in sets.values() for s in ss};coverage={}
 for seed in SEEDS:
  for arm in ('old','multi'):
   for part in ('train','validation'):
    name=f'{arm}_{seed}';rs=json.loads((RUN/(name+'_'+part+'_actions.json')).read_text());outside=[r for r in rs if r['action'] not in labels[r['decision_id']]]
    coverage[name+'_'+part]=dict(n=len(rs),outside_candidate_count=len(outside),outside_bad_count=sum(r['regret']>=4 for r in outside),inside_bad_count=sum(r['regret']>=4 for r in rs if r not in outside))
 audit=json.loads((RUN/'sampling_audit.json').read_text())
 if any(audit[f'old_{s}']!=audit[f'multi_{s}'] for s in SEEDS):raise ValueError('对照抽样不一致')
 if digest(REGISTRY)!=sha:raise ValueError('登记变化')
 result=dict(summary=summary,final_candidate_coverage=coverage,identical_sampling=True,registry_unchanged=True,scope='固定原200局面的初始logits梯度与已保存最终动作审计；不能据此证明整个优化过程的因果归属。',source_sha256=digest(__file__),input_config_sha256=digest(RUN/'config.json'))
 atomic(OUT/'result.json',result);atomic(OUT/'per_decision.json',rows);(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
 lines=['# 多候选监督未提升：既有数据诊断','', '两种监督同时改变了同分集合交叉熵和排序比较集合，原实验不能单独归因某项。数据/模型均不修改，不新训练或对战。','', '|指标|训练160|验证40|','|---|---:|---:|']
 for label,key in [('含多个同分最优的局面数','multiple_equal_best'),('每局面平均候选数','mean_candidates'),('每局面平均排序对数','mean_pairs'),('次优之间比较占比','inferior_comparison_fraction'),('原好坏对平均权重','mean_original_pair_weight'),('排序梯度抬高某个次优动作的局面数','states_rank_promotes_inferior'),('新旧根梯度范数均值比','new_to_old_mean_gradient_norm'),('新旧根梯度平均余弦','mean_gradient_cosine')]:lines.append(f'|{label}|{summary["train"][key]:.4f}|{summary["validation"][key]:.4f}|')
 lines+=['','全部分差足够的候选对都参与排序，会包含“较差棋优于更差棋”；因此部分梯度可能抬高仍次优的动作。每局面按比较对数量取均值，也使原关键好坏对的单项权重降低。这个目标没有数学实现错误，但优化重点不等于只提升最优动作、压制差棋。','这些是固定初始参数下的梯度事实，不证明它们独自造成最终验证下降；共享CNN、普通回放和后续优化也有影响。候选来自固定旧学生，最终策略可能走到候选池外。','', '## 下一步有限对照建议','先保持原监督为控制，单独检查两个因素：①唯一专家交叉熵改为同分集合交叉熵，仍用原好坏排序；②保留唯一专家交叉熵，仅将排序改为“最优集合对差棋”，保持每局面总权重。两者分别对照，禁止同时换多项再凭单轮结果归因。使用固定32训练根先审查拟合/梯度，再使用160训练与按新开局组留出的开发集检查泛化；现有40局面已反复用于诊断，不再作为新独立验收。','本轮未启动训练、未升级，GUI仍为v3。实际候选覆盖数字见result.json。']
 (OUT/'REPORT.md').write_text('\n'.join(lines));(OUT/'中文日志.md').write_text('已完成既有标签、初始根梯度、最终动作覆盖及抽样一致性审计。未训练、未新评分、未对战、未修改冠军。\n');atomic(OUT/'status.json',dict(stage='complete'));print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
