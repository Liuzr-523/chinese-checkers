"""最终参数下32根之间的局部梯度传递矩阵；不更新模型。"""
import json,sys,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from az121 import Net121
from curriculum121 import batch,atomic
from diagnostics.diagnose121_hardroots import flatgrad
from checkpoint121_save import digest
B=Path('nn_checkpoints/experiments/ablate121_supervision32_20261101');O=Path('nn_checkpoints/experiments/diagnose121_hardroots_20261101')
torch.set_num_threads(1);sets=torch.load(B/'data_sources/multicandidate.pt',weights_only=True)['sets'];lookup={s['decision_id']:s for s in sets['train']};ids=json.loads((B/'data_sources/fixed32.json').read_text())['decision_ids'];roots=[lookup[i] for i in ids];bad={r['decision_id']:r['action'] for r in json.loads((B/'control_train32_actions.json').read_text()) if r['regret']>=4};m=Net121.load(B/'control_step1600.pt');params=[p for n,p in m.named_parameters() if not n.startswith('value.')];x,indices,mask,target,_=batch(roots);logits,_=m(x);legal=logits.gather(1,indices).masked_fill(~mask,-1e9);ce=F.cross_entropy(legal,target,reduction='none');losses=[];margins=[]
for i,s in enumerate(roots):
 good=s['actions'][s['target']];losses.append(.25*ce[i]+.25*F.relu(.5-logits[i,good]+logits[i,s['rejected_action']]))
 if s['decision_id'] in bad:margins.append((s['decision_id'],logits[i,good]-logits[i,bad[s['decision_id']]]))
gradients=torch.stack([flatgrad(v,params) for v in losses]);mg=torch.stack([flatgrad(v,params) for _,v in margins]);directions=(-mg@gradients.T).detach();rows=[]
for i,(did,_) in enumerate(margins):
 own=ids.index(did);others=[j for j in range(32) if j!=own];values=directions[i];opposing=sorted(others,key=lambda j:float(values[j]))[:3]
 rows.append(dict(decision_id=did,own_gradient_direction=float(values[own]),other31_mean_direction=float(values[others].mean()),uniform_root32_mean_direction=float(values.mean()),own_contribution_at_uniform_weight=float(values[own]/32),other31_contribution_at_uniform_weight=float(values[others].sum()/32),opposing_roots=[dict(decision_id=ids[j],direction=float(values[j])) for j in opposing]))
result=dict(rows=rows,self_positive=sum(r['own_gradient_direction']>0 for r in rows),other31_negative=sum(r['other31_mean_direction']<0 for r in rows),uniform_negative=sum(r['uniform_root32_mean_direction']<0 for r in rows),scope='固定最终参数、原专项CE+原好坏排序，7难例间隔对32个单根损失的梯度方向。无动量的无穷小梯度下降近似，不等于Adam实际步或长期棋力。',source_sha256=digest(__file__))
atomic(O/'root_interference.json',result);(O/'root_interference_source.py').write_bytes(Path(__file__).read_bytes())
with (O/'专项根相互影响.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['难例编号']+['训练根'+str(did) for did in ids]);w.writerows([[did]+[float(v) for v in directions[i]] for i,(did,_) in enumerate(margins)])
print(json.dumps(result,ensure_ascii=False,indent=2))
