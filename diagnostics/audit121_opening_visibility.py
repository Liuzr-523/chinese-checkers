"""Exact local-policy input alias audit; no model update or training labels changed."""
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from az121 import Net121
from train_opening121 import book_samples
from opening121 import OpeningBook
from curriculum121 import batch,atomic
import rules121 as r

def support(a,b,radius=5):
 mask=np.zeros((17,17),dtype=bool)
 for action in (a,b):
  for cell in divmod(action,121):
   q,y=r.COORDS[cell];rr,cc=y+8,q+8;mask[max(0,rr-radius):min(17,rr+radius+1),max(0,cc-radius):min(17,cc+radius+1)]=True
 return mask

def find_pairs(samples):
 groups=defaultdict(list)
 for i,s in enumerate(samples):
  if len(s['positive'])==1:groups[(round(float(np.asarray(s['x'])[4].max())*240),int(np.asarray(s['x'])[3].max()))].append(i)
 pairs=[];masks={}
 for members in groups.values():
  for j,i in enumerate(members):
   s=samples[i];a=s['positive'][0]
   for k in members[j+1:]:
    t=samples[k];b=t['positive'][0]
    if a==b or a not in t['actions'] or b not in s['actions']:continue
    if (s['actions'].index(a)<s['actions'].index(b))!=(t['actions'].index(a)<t['actions'].index(b)):continue
    if (a,b) not in masks:masks[a,b]=support(a,b)
    mask=masks[a,b]
    if np.array_equal(np.asarray(s['x'])[:,mask],np.asarray(t['x'])[:,mask]):pairs.append(dict(left=i,right=k,left_action=a,right_action=b,left_id=s['decision_id'],right_id=t['decision_id']))
 return pairs

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(1);sets,_=book_samples(OpeningBook());result={}
 for part,samples in sets.items():
  pairs=find_pairs(samples);used=set();disjoint=[]
  for p in pairs:
   if p['left'] not in used and p['right'] not in used:used.update((p['left'],p['right']));disjoint.append(p)
  result[part]=dict(n=len(samples),contradictory_pairs=len(pairs),vertex_disjoint_pairs=len(disjoint),top1_upper_bound_from_disjoint_pairs=1-len(disjoint)/len(samples),examples=pairs[:10])
  if part=='train' and pairs:
   cfg=json.loads((a.out/'config.json').read_text());model=Net121.load(cfg['model']);errors=[]
   with torch.no_grad():
    for p in pairs[:10]:
     x,*_=batch([samples[p['left']],samples[p['right']]]);l,_=model(x);errors.append(max(float(abs(l[0,p['left_action']]-l[1,p['left_action']])),float(abs(l[0,p['right_action']]-l[1,p['right_action']]))))
   result[part]['baseline_pair_logit_max_difference']=max(errors)
 result['architecture']='stem3x3 + 2 residual blocks x two3x3 => policy radius5. Each source/destination head reads only its local cell feature, value pooling does not feed policy.'
 atomic(a.out/'visibility_audit.json',result);print(json.dumps({k:{a:b for a,b in v.items() if a!='examples'} if isinstance(v,dict) else v for k,v in result.items()},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
