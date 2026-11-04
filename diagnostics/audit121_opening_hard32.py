"""Post-fit case history, duplicate labels and weighted policy-gradient audit."""
import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from az121 import Net121
from curriculum121 import batch,atomic,samples_from_game
from canonical121_trial import canonical_sample
from cycle121_targeted import expert_samples
from opening121 import OpeningBook
from train_opening121 import book_samples,set_ce
from generalization121_curve import remove_validation_duplicates,input_digest
from diagnostics.diagnose121_hardroots import flatgrad

def main(folder):
 torch.set_num_threads(1);p=Path(folder);cfg=json.loads((p/'config.json').read_text());ss=torch.load(p/'data_sources/fixed32.pt',weights_only=True)['samples'];histories={}
 for arm in ('root_only','mixed'):
  rows=json.loads((p/(arm+'_cases.json')).read_text());details=[]
  for s in ss:
   h=[r for r in rows if r['decision_id']==s['decision_id']];good=[r['step'] for r in h if r['correct']];details.append(dict(decision_id=s['decision_id'],ever_recorded_correct=bool(good),first_correct=good[0] if good else None,final_correct=h[-1]['correct'],recorded_forgetting=any(a['correct'] and not b['correct'] for a,b in zip(h,h[1:]))))
  curve=json.loads((p/(arm+'_curve.json')).read_text());histories[arm]=dict(ever_correct=sum(d['ever_recorded_correct'] for d in details),final_correct=sum(d['final_correct'] for d in details),recorded_forgetting=sum(d['recorded_forgetting'] for d in details),ever_correct_but_final_wrong=sum(d['ever_recorded_correct'] and not d['final_correct'] for d in details),peak_correct=max(round(r['fixed32']['top1_in_book']*32) for r in curve),first_full_fit=next((r['step'] for r in curve if r['fixed32']['top1_in_book']==1.),None),details=details)
 old=[];normal=[];split=json.loads((p/'data_sources/old_split.json').read_text())
 for line in (p/'data_sources/old_teacher.jsonl').read_text().splitlines():
  row=json.loads(line)
  if row['group'] in split['train']:old.extend({**canonical_sample(s),'value_weight':1.} for s in samples_from_game(row))
 for line in (p/'data_sources/normal_games.jsonl').read_text().splitlines():
  row=json.loads(line)
  if row['split']=='train':normal.extend(expert_samples(row))
 val=book_samples(OpeningBook())[0]['validation'];old,_=remove_validation_duplicates(old,val);normal,_=remove_validation_duplicates(normal,val);index={}
 for s in old+normal:index.setdefault(input_digest(s),[]).append(s)
 overlap=[]
 for s in ss:
  matches=index.get(input_digest(s),[])
  if matches:overlap.append(dict(decision_id=s['decision_id'],matches=len(matches),conflicting=sum(t['actions'][t['target']] not in s['positive'] for t in matches)))
 reference=Net121.load(cfg['model']);reference.requires_grad_(False);gradients=[]
 for name,path in [('v3',Path(cfg['model'])),('mixed_final',p/'mixed_step2400.pt')]:
  m=Net121.load(path);params=[v for n,v in m.named_parameters() if not n.startswith('value.')];rr=np.random.default_rng(110414);ro=np.random.default_rng(110415)
  for i in range(8):
   roots=[ss[int(j)] for j in rr.integers(32,size=4)];ordinary=[old[int(j)] for j in ro.integers(len(old),size=6)]+[normal[int(j)] for j in ro.integers(len(normal),size=6)];samples=roots+ordinary;x,ids,mask,target,z=batch(samples);l,v=m(x);legal=l.gather(1,ids).masked_fill(~mask,-1e9);root=.25*set_ce(l[:4],roots);replay=.75*F.cross_entropy(legal[4:],target[4:]);weight=torch.tensor([s['value_weight'] for s in samples]);genuine=((v-z).square()*weight).sum()/weight.sum().clamp_min(1)
   with torch.no_grad():ol,ov=reference(x);oldlp=ol.gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)
   lp=legal.log_softmax(1);aux=genuine+.1*(oldlp.exp()*(oldlp-lp)).sum(1).mean()+.1*F.mse_loss(v,ov);a,b,c=[flatgrad(loss,params) for loss in (root,replay,aux)]
   gradients.append(dict(model=name,batch=i,root_norm=float(a.norm()),replay_norm=float(b.norm()),aux_norm=float(c.norm()),root_replay_cosine=float(F.cosine_similarity(a,b,dim=0)),root_aux_cosine=float(F.cosine_similarity(a,c,dim=0))))
 summary={}
 for name in ('v3','mixed_final'):
  rows=[g for g in gradients if g['model']==name];summary[name]=dict(root_replay_negative=sum(r['root_replay_cosine']<0 for r in rows),mean_root_replay_cosine=float(np.mean([r['root_replay_cosine'] for r in rows])),root_aux_negative=sum(r['root_aux_cosine']<0 for r in rows),mean_root_aux_cosine=float(np.mean([r['root_aux_cosine'] for r in rows])),mean_replay_root_norm_ratio=float(np.mean([r['replay_norm']/r['root_norm'] for r in rows])),mean_aux_root_norm_ratio=float(np.mean([r['aux_norm']/r['root_norm'] for r in rows])))
 result=dict(histories=histories,ordinary_duplicate_inputs=overlap,gradient_summary=summary,gradient_details=gradients,scope='每100步记录的训练过程；固定两参数点相同8批梯度为局部检查。未更新权重，不单独归因整个混合目标的长期作用。')
 atomic(p/'case_and_gradient_audit.json',result);print(json.dumps(dict(histories={k:{a:b for a,b in v.items() if a!='details'} for k,v in histories.items()},overlaps=overlap,gradient_summary=summary),ensure_ascii=False,indent=2))
if __name__=='__main__':main(sys.argv[1])
