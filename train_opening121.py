"""Fixed-budget opening imitation versus prior decision replay; raw NN, no book wrapper."""
import argparse,copy,json,hashlib,os,time,traceback,shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import rules121 as r
from az121 import Net121,encode,action_id,Neural121
from opening121 import OpeningBook,MIRROR,key
from curriculum121 import atomic,batch,samples_from_game
from canonical121_trial import canonical_sample
from cycle121_targeted import expert_samples
from diagnostics.fit121_objectives import objective_terms
from generalization121_curve import remove_validation_duplicates
from unfreeze121_trial import metrics
from checkpoint121_save import REGISTRY,digest

def book_samples(book):
 sets={'train':[],'validation':[]};seen=set();groups={}
 for state,entry in book.data['nodes'].items():
  raw,p=state.split(':');p=int(p);board=list(map(int,raw));reflected=[0]*121
  for i,v in enumerate(board):reflected[MIRROR[i]]=v
  group=min(key(board,p),key(reflected,p));part='validation' if entry['depth']>=6 and int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%5==0 else 'train';groups[group]=part
  for b,paths in [(board,entry['moves']),(reflected,[[MIRROR[i] for i in path] for path in entry['moves']])]:
   identity=key(b,p)
   if identity in seen:continue
   seen.add(identity);actions=[action_id(s,path[-1],p,True) for s,path in r.get_all_moves(b,p)];positive=list(dict.fromkeys(action_id(path[0],path[-1],p,True) for path in book.moves(b,p,entry['depth'])))
   sets[part].append(dict(x=encode(b,p,entry['depth'],canonical=True),actions=actions,target=actions.index(positive[0]),positive=positive,z=0.,value_weight=0.,decision_id=identity))
 return sets,groups

def set_ce(logits,samples):
 return torch.stack([torch.logsumexp(row[s['actions']],0)-torch.logsumexp(row[s.get('positive',[s['actions'][s['target']]])],0) for row,s in zip(logits,samples)]).mean()

@torch.no_grad()
def mastery(model,samples):
 hits=mass=loss=0.
 for offset in range(0,len(samples),32):
  ss=samples[offset:offset+32];x,*_=batch(ss);l,_=model(x)
  for row,s in zip(l,ss):
   positive=s['positive'];a=s['actions'][int(row[s['actions']].argmax())];hits+=a in positive;lp=torch.logsumexp(row[positive],0)-torch.logsumexp(row[s['actions']],0);mass+=float(lp.exp());loss-=float(lp)
 return dict(n=len(samples),top1_in_book=hits/len(samples),book_probability=mass/len(samples),set_ce=loss/len(samples))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(1);a.out.mkdir(parents=True,exist_ok=False);began=time.monotonic();registry=digest(REGISTRY);entry=json.loads(REGISTRY.read_text());model_path=Path(entry['checkpoint']);base=Net121.load(model_path);base.requires_grad_(False);book=OpeningBook();sets,groups=book_samples(book)
 def status(stage,**kw):atomic(a.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-began,**kw))
 cfg=dict(profile=r.PROFILE,model=str(model_path),model_sha256=digest(model_path),registry_sha256=registry,book_sha256=digest('opening121_book.json'),seed=110405,steps=1200,lr=3e-5,root_batch=4,ordinary_batch=12,arms=['prior_decisions','opening_book'],split_counts={p:len(v) for p,v in sets.items()},scope='原始NN关闭外部开局库；镜像同组，前6总手数作为训练回归，其余按局面镜像轨道留出。非完整对局验收，不登记模型。')
 source_names=['train_opening121.py','opening121.py','opening121_book.json','az121.py','rules121.py','curriculum121.py','canonical121_trial.py','cycle121_targeted.py','diagnostics/fit121_objectives.py','unfreeze121_trial.py','generalization121_curve.py']
 for name in source_names:
  dst=a.out/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dst)
 cfg['source_sha256']={n:digest(n) for n in source_names};atomic(a.out/'config.json',cfg)
 try:
  source=Path('nn_checkpoints/experiments/dynamic121_actual_20261104/data_sources');old={'train':[],'validation':[]};normal={'train':[],'validation':[]};split=json.loads((source/'old_split.json').read_text());prior=torch.load(source/'multicandidate.pt',weights_only=True)['sets']['train']
  for line in (source/'old_teacher.jsonl').read_text().splitlines():
   row=json.loads(line);part=next((p for p in old if row['group'] in split[p]),None)
   if part:old[part].extend({**canonical_sample(s),'value_weight':1.} for s in samples_from_game(row))
  for line in (source/'normal_games.jsonl').read_text().splitlines():
   row=json.loads(line);normal[row['split']].extend(expert_samples(row))
  removed={}
  for name,pool in [('old',old),('normal',normal)]:pool['train'],removed[name]=remove_validation_duplicates(pool['train'],sets['validation'])
  prior,prior_removed=remove_validation_duplicates(prior,sets['validation']);atomic(a.out/'split.json',dict(groups=groups,removed_ordinary=removed,removed_prior=prior_removed))
  def measure(m):return dict(book_train=mastery(m,sets['train']),book_validation=mastery(m,sets['validation']),retention=metrics(m,old['validation']),normal=metrics(m,normal['validation']))
  before=measure(base);atomic(a.out/'baseline.json',before);results={};sequences={}
  for arm,pool in [('prior_decisions',prior),('opening_book',sets['train'])]:
   m=copy.deepcopy(base);m.requires_grad_(True);opt=torch.optim.AdamW(m.parameters(),lr=3e-5,weight_decay=1e-4);rr=np.random.default_rng(110406);ro=np.random.default_rng(110407);curve=[];sequence=hashlib.sha256()
   for step in range(1201):
    if step%100==0:
     curve.append(dict(step=step,**measure(m)));atomic(a.out/(arm+'_curve.json'),curve);status('原始NN开局模仿',arm=arm,step=step,total=1200)
     if step%400==0:m.save(a.out/(arm+f'_step{step}.pt'))
     if step==1200:break
    rs=[pool[int(i)] for i in rr.integers(len(pool),size=4)];oi=ro.integers(len(old['train']),size=6);ni=ro.integers(len(normal['train']),size=6);ordinary=[old['train'][int(i)] for i in oi]+[normal['train'][int(i)] for i in ni];sequence.update(oi.tobytes()+ni.tobytes())
    # Rank is omitted in BOTH arms; roots differ only in the supervised decision source.
    prepared=[{**s,'rejected_action':s.get('rejected_action',s['actions'][s['target']])} for s in rs];loss,terms=objective_terms(m,base,prepared,ordinary,'full_objective');x,*_=batch(rs);l,_=m(x);loss=loss-terms['root_ce']-terms['rank']+.25*set_ce(l,rs)
    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
   results[arm]=curve[-1];sequences[arm]=sequence.hexdigest()
  if len(set(sequences.values()))!=1:raise ValueError('普通回放样本顺序不同')
  candidate=Net121.load(a.out/'opening_book_step1200.pt');candidate.save(a.out/'candidate.pt');result=dict(baseline=before,final=results,ordinary_sequence_sha256=sequences,promoted=False,arena_run=False,seconds=time.monotonic()-began)
  if digest(REGISTRY)!=registry or digest(model_path)!=cfg['model_sha256']:raise ValueError('冠军变化')
  atomic(a.out/'result.json',result);lines=['# 原始NN开局库模仿学习','',f'数据：训练{len(sets["train"])}，留出{len(sets["validation"])}，镜像同组。两组同v3/1200更新/普通回放，固定最终检查点。','|模型|训练首选在库中|留出首选在库中|留出库招总概率|旧策略交叉熵|','|---|---:|---:|---:|---:|']
  for label,v in [('v3',before)]+list(results.items()):lines.append(f'|{label}|{v["book_train"]["top1_in_book"]:.1%}|{v["book_validation"]["top1_in_book"]:.1%}|{v["book_validation"]["book_probability"]:.1%}|{v["retention"]["policy_ce"]:.3f}|')
  lines+=['','网络评估不使用BookEngine，开局样本只有策略标签、无伪造胜负。该候选未登记；GUI仍使用原冠军+可关闭的独立开局库。完整对局验收后再决定模型升级。'];(a.out/'REPORT.md').write_text('\n'.join(lines));status('complete',result=result)
 except BaseException:status('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
