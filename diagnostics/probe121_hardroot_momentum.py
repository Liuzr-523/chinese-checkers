"""补充隔离原AdamW历史：零梯度、清一阶动量、全新AdamW、无动量SGD。"""
import copy,json,sys,time,os,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from az121 import Net121
from curriculum121 import batch,atomic,samples_from_game
from canonical121_trial import canonical_sample
from cycle121_targeted import expert_samples
from generalization121_curve import remove_validation_duplicates
from diagnostics.fit121_objectives import objective_terms
from checkpoint121_save import digest,REGISTRY
BASE=Path('nn_checkpoints/experiments/ablate121_supervision32_20261101');RUN=Path('nn_checkpoints/experiments/diagnose121_hardroots_20261101');OUT=RUN/'动量补充审计'

def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=False);started=time.monotonic();registry=digest(REGISTRY);cfg=json.loads((BASE/'config.json').read_text());data=BASE/'data_sources';ref=Net121.load(cfg['model']);ref.requires_grad_(False);sets=torch.load(data/'multicandidate.pt',weights_only=True)['sets'];lookup={s['decision_id']:s for s in sets['train']};roots=[lookup[i] for i in json.loads((data/'fixed32.json').read_text())['decision_ids']];bad={s['decision_id']:s['action'] for s in json.loads((BASE/'control_train32_actions.json').read_text()) if s['regret']>=4};hard=[lookup[i] for i in bad]
 def status(stage,**kw):atomic(OUT/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-started,**kw))
 try:
  old=[];normal=[];split=json.loads((data/'old_split.json').read_text())
  for line in (data/'old_teacher.jsonl').read_text().splitlines():
   row=json.loads(line)
   if row['group'] in split['train']:old.extend({**canonical_sample(s),'value_weight':1.} for s in samples_from_game(row))
  for line in (data/'normal_games.jsonl').read_text().splitlines():
   row=json.loads(line)
   if row['split']=='train':normal.extend(expert_samples(row))
  old,_=remove_validation_duplicates(old,sets['validation']);normal,_=remove_validation_duplicates(normal,sets['validation'])
  def get_batch(rr,ro):return [roots[int(i)] for i in rr.integers(32,size=4)],[old[int(i)] for i in ro.integers(len(old),size=6)]+[normal[int(i)] for i in ro.integers(len(normal),size=6)]
  model=copy.deepcopy(ref);model.requires_grad_(True);opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=1e-4);rr=np.random.default_rng(111054);ro=np.random.default_rng(111055)
  for step in range(1600):
   rs,ordinary=get_batch(rr,ro);loss,_=objective_terms(model,ref,rs,ordinary,'full_objective');opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
   if step%100==0:status('精确重建原优化器状态',step=step+1,total=1600)
  expected=Net121.load(BASE/'control_step1600.pt');delta=max(float((v-model.state_dict()[k]).abs().max()) for k,v in expected.state_dict().items())
  if delta!=0:raise ValueError('原模型未逐值复现')
  @torch.no_grad()
  def margin(m):
   x,*_=batch(hard);l,_=m(x);return torch.stack([l[i,s['actions'][s['target']]]-l[i,bad[s['decision_id']]] for i,s in enumerate(hard)])
  before=margin(model);state=copy.deepcopy(opt.state_dict());probe=[]
  zero=copy.deepcopy(model);zo=torch.optim.AdamW(zero.parameters(),lr=3e-5,weight_decay=1e-4);zo.load_state_dict(copy.deepcopy(state))
  for p in zero.parameters():p.grad=torch.zeros_like(p)
  zo.step();zero_delta=float((margin(zero)-before).mean())
  rr=np.random.default_rng(9901);ro=np.random.default_rng(9902)
  for i in range(8):
   rs,ordinary=get_batch(rr,ro)
   for mode in ('inherited','reset_first_moment','fresh_adam','sgd'):
    for component in ('full','root_only','replay_only','aux_only'):
     child=copy.deepcopy(model)
     if mode=='sgd':co=torch.optim.SGD(child.parameters(),lr=3e-5,weight_decay=1e-4)
     else:
      co=torch.optim.AdamW(child.parameters(),lr=3e-5,weight_decay=1e-4)
      if mode!='fresh_adam':co.load_state_dict(copy.deepcopy(state))
      if mode=='reset_first_moment':
       for s in co.state.values():s['exp_avg'].zero_()
     _,terms=objective_terms(child,ref,rs,ordinary,'full_objective');loss={'full':sum(terms.values()),'root_only':terms['root_ce']+terms['rank'],'replay_only':terms['replay_ce'],'aux_only':terms['value']+terms['kl']+terms['value_preservation']}[component];co.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(child.parameters(),1);co.step();change=margin(child)-before;probe.append(dict(batch=i,mode=mode,component=component,margin_delta=float(change.mean()),per_hard_delta=[float(x) for x in change]))
   status('隔离优化器历史探针',completed=i+1,total=8)
  summary={mode:{component:dict(mean_delta=float(np.mean([x['margin_delta'] for x in probe if x['mode']==mode and x['component']==component])),positive_batches=sum(x['margin_delta']>0 for x in probe if x['mode']==mode and x['component']==component)) for component in ('full','root_only','replay_only','aux_only')} for mode in ('inherited','reset_first_moment','fresh_adam','sgd')}
  if digest(REGISTRY)!=registry:raise ValueError('冠军变动')
  atomic(OUT/'probes.json',probe);atomic(OUT/'result.json',dict(summary=summary,zero_gradient_inherited_delta=zero_delta,control_max_parameter_delta=delta,source_sha256=digest(__file__),registry_unchanged=True,scope='从相同最终模型隔离单步优化器状态。重置动量/SGD并非长期训练或棋力验收，不直接部署。'));(OUT/'source.py').write_bytes(Path(__file__).read_bytes());status('complete');print(json.dumps(dict(summary=summary,zero_gradient_inherited_delta=zero_delta),indent=2))
 except BaseException:status('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
