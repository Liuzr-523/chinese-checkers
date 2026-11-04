"""25/50/75% opening replay, fixed v3/CNN/optimizer/update budget."""
import argparse,copy,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from az121 import Net121,encode,action_id
from opening121 import OpeningBook
import rules121 as r
from train_opening121 import book_samples,mastery,set_ce
from curriculum121 import atomic,batch,samples_from_game
from canonical121_trial import canonical_sample
from cycle121_targeted import expert_samples
from generalization121_curve import remove_validation_duplicates
from unfreeze121_trial import metrics
from checkpoint121_save import REGISTRY,digest,verify
FRACTIONS=(4,8,12)

def mixed_loss(model,reference,roots,ordinary):
 ss=roots+ordinary;x,ids,mask,target,z=batch(ss);logits,value=model(x);n=len(roots);f=n/len(ss);legal=logits.gather(1,ids).masked_fill(~mask,-1e9)
 policy=f*set_ce(logits[:n],roots)+(1-f)*F.cross_entropy(legal[n:],target[n:]);weights=torch.tensor([s['value_weight'] for s in ss]);genuine=((value-z).square()*weights).sum()/weights.sum().clamp_min(1)
 with torch.no_grad():old_l,old_v=reference(x);old=old_l.gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)
 lp=legal.log_softmax(1);kl=(old.exp()*(old-lp)).sum(1).mean();preservation=F.mse_loss(value,old_v)
 return policy+genuine+.1*kl+.1*preservation

def stage_of(sample):return min(10,round(float(np.asarray(sample['x'])[4].max())*240)//2+1)
def stage_metrics(model,samples):
 out={}
 for name,lo,hi in [('moves1_3',1,3),('moves4_6',4,6),('moves7_10',7,10)]:
  part=[s for s in samples if lo<=stage_of(s)<=hi];out[name]=mastery(model,part) if part else dict(n=0,top1_in_book=None)
 return out

def acceptance(row,before):
 checks=dict(train90=row['book_train']['top1_in_book']>=.9,validation_gain10pp=row['book_validation']['top1_in_book']>=before['book_validation']['top1_in_book']+.1,
  first_three=row['standard_first6']['top1_in_book']==1.,old_policy=row['retention']['policy_ce']<=before['retention']['policy_ce']+.05,
  old_value=row['retention']['value_mse']<=before['retention']['value_mse']+.05,normal_policy=row['normal']['policy_ce']<=before['normal']['policy_ce']+.05)
 return all(checks.values()),checks

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(1);saved=verify(Path('nn_checkpoints/saves/v3_before_unfreeze_20261030'))
 if digest(REGISTRY)!=saved['registry_sha256']:raise ValueError('现役已变化，请保护新冠军')
 entry=json.loads(REGISTRY.read_text());path=Path(entry['checkpoint']);base=Net121.load(path);base.requires_grad_(False);a.out.mkdir(parents=True,exist_ok=False);lock=(a.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);began=time.monotonic();book=OpeningBook();sets,groups=book_samples(book)
 def status(stage,**kw):
  atomic(a.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-began,**kw))
  if stage not in ('complete','failed') and time.monotonic()-began>10800:raise TimeoutError('三小时预算耗尽')
 def log(text):
  with (a.out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-began:.1f}秒：{text}\n')
 names=['opening121_ratios.py','train_opening121.py','opening121.py','opening121_book.json','az121.py','rules121.py','curriculum121.py','canonical121_trial.py','cycle121_targeted.py','generalization121_curve.py','unfreeze121_trial.py','confirm121_policy_20261104.py','diagnostics/report121_policy_confirmation.py','diagnostics/plot121_opening_ratios.py','diagnostics/plot_style_zh.py','tests/test_opening121_ratios.py']
 cfg=dict(profile=r.PROFILE,model=str(path.resolve()),model_sha256=digest(path),registry_sha256=digest(REGISTRY),seed=110408,steps=2400,lr=3e-5,batch_size=16,root_counts=list(FRACTIONS),opening_weights=[.25,.5,.75],max_seconds=10800,
  weights='策略损失随真实样本份额25/50/75%变化；真实value MSE、KL .1、value保留 .1保持原定义，根无胜负标签。',
  selection='固定最终2400；训练>=90%、验证较v3至少+10pp、标准前6总手全命中、旧CE<=v3+.05且value<=v3+.05、新普通CE<=v3+.05。达标者按验证首选命中最高选一次，同分优先低份额。',
  arena='达标后只选一次，候选/v3双方开同一开局库，MCTS32；120对v3、新旧各40对同样开库C30ms。配对p<=.01且C积分不降，完整审计后登记。',source_sha256={n:digest(n) for n in names})
 for n in names:
  target=a.out/'source'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,target)
 source=Path('nn_checkpoints/experiments/dynamic121_actual_20261104/data_sources');data=a.out/'data_sources';data.mkdir();cfg['data_sha256']={}
 for n in ('old_teacher.jsonl','old_split.json','normal_games.jsonl'):
  shutil.copyfile(source/n,data/n);cfg['data_sha256'][n]=digest(data/n)
 atomic(a.out/'config.json',cfg);atomic(a.out/'split.json',groups)
 try:
  old={'train':[],'validation':[]};normal={'train':[],'validation':[]};split=json.loads((data/'old_split.json').read_text())
  for line in (data/'old_teacher.jsonl').read_text().splitlines():
   row=json.loads(line);part=next((p for p in old if row['group'] in split[p]),None)
   if part:old[part].extend({**canonical_sample(s),'value_weight':1.} for s in samples_from_game(row))
  for line in (data/'normal_games.jsonl').read_text().splitlines():
   row=json.loads(line);normal[row['split']].extend(expert_samples(row))
  for pool in (old,normal):pool['train'],_=remove_validation_duplicates(pool['train'],sets['validation'])
  standard=[];g=r.Game()
  for _ in range(6):
   moves=book.moves(g.board,g.player,g.ply,g.repetitions);actions=[action_id(s,p[-1],g.player,True) for s,p in r.get_all_moves(g.board,g.player)];positive=[action_id(p[0],p[-1],g.player,True) for p in moves];standard.append(dict(x=encode(g.board,g.player,g.ply,g.repetitions,canonical=True),actions=actions,positive=positive,target=actions.index(positive[0]),z=0.,value_weight=0.));g.play(book.choose(g.board,g.player,g.ply,g.repetitions))
  def measure(m):return dict(book_train=mastery(m,sets['train']),book_validation=mastery(m,sets['validation']),train_stages=stage_metrics(m,sets['train']),validation_stages=stage_metrics(m,sets['validation']),standard_first6=mastery(m,standard),retention=metrics(m,old['validation']),normal=metrics(m,normal['validation']))
  before=measure(base);atomic(a.out/'baseline.json',before);final={};audits={};log('开始25/50/75%对照。三组同v3、2400更新与16总样本；前3/4—6/7—10步分别记录。')
  for n in FRACTIONS:
   name=f'book{n*100//16}';m=copy.deepcopy(base);m.requires_grad_(True);opt=torch.optim.AdamW(m.parameters(),lr=3e-5,weight_decay=1e-4);rr=np.random.default_rng(110409);ro=np.random.default_rng(110410);master=hashlib.sha256();curve=[]
   for step in range(2401):
    if step%200==0:
     curve.append(dict(step=step,**measure(m)));atomic(a.out/(name+'_curve.json'),curve);status('开局份额对照',arm=name,step=step,total=2400)
     if step%600==0:m.save(a.out/(name+f'_step{step}.pt'))
     if step==2400:break
    ri=rr.integers(len(sets['train']),size=12);oi=ro.integers(len(old['train']),size=6);ni=ro.integers(len(normal['train']),size=6);master.update(ri.tobytes()+oi.tobytes()+ni.tobytes());k=(16-n)//2;roots=[sets['train'][int(i)] for i in ri[:n]];ordinary=[old['train'][int(i)] for i in oi[:k]]+[normal['train'][int(i)] for i in ni[:k]]
    loss=mixed_loss(m,base,roots,ordinary)
    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
   final[name]=curve[-1];audits[name]=dict(master_sequence_sha256=master.hexdigest(),opening_exposures=2400*n,ordinary_exposures=2400*(16-n),total_examples=38400,updates=2400);atomic(a.out/'sampling_audit.json',audits);log(name+'完成：训练命中'+str(final[name]['book_train']['top1_in_book'])+'，验证命中'+str(final[name]['book_validation']['top1_in_book']))
  if len({v['master_sequence_sha256'] for v in audits.values()})!=1:raise ValueError('共享抽样序列不一致')
  gates={};eligible=[]
  for name,row in final.items():
   passed,checks=acceptance(row,before);gates[name]=dict(passed=passed,checks=checks)
   if passed:eligible.append(name)
  selected=max(eligible,key=lambda name:(final[name]['book_validation']['top1_in_book'],-int(name[4:]))) if eligible else None
  result=dict(baseline=before,final=final,gates=gates,selected=selected,promoted=False,arena_run=False,seconds=time.monotonic()-began);atomic(a.out/'result.json',result);atomic(a.out/'selection.json',dict(selected=selected,gates=gates))
  if digest(REGISTRY)!=cfg['registry_sha256'] or digest(path)!=cfg['model_sha256']:raise ValueError('冠军并发变化')
  if selected:
   shutil.copyfile(a.out/(selected+'_step2400.pt'),a.out/'candidate.pt');log('达标选定'+selected+'，进入双方开库的完整200局验收。');status('进入完整对局验收',selected=selected)
   confirm=a.out/'confirmation';done=subprocess.run([sys.executable,'confirm121_policy_20261104.py','--out',str(confirm),'--parent',str(a.out),'--candidate',str(a.out/'candidate.pt'),'--seed','110411','--version','v4_opening_ratios_20261104','--with-book'],stdout=(a.out/'confirmation.log').open('w'),stderr=subprocess.STDOUT)
   if done.returncode:raise RuntimeError('完整验收失败，见confirmation.log')
   result.update(arena_run=True,confirmation=json.loads((confirm/'result.json').read_text()));result['promoted']=result['confirmation']['promoted'];atomic(a.out/'result.json',result)
  plotted=subprocess.run(['/Library/Frameworks/Python.framework/Versions/3.10/bin/python3','diagnostics/plot121_opening_ratios.py',str(a.out)],capture_output=True,text=True);atomic(a.out/'plot_status.json',dict(exit_code=plotted.returncode,stderr=plotted.stderr))
  if plotted.returncode:raise RuntimeError('中文图表渲染失败')
  lines=['# 开局数据份额对照','',f'固定2400步、相同v3和优化器。最终选定：{selected}；整局验收：{result["arena_run"]}；升级：{result["promoted"]}。','|开局份额|训练命中|留出命中|标准前6总手|旧策略CE|旧价值MSE|','|---|---:|---:|---:|---:|---:|']
  for name,row in final.items():lines.append(f'|{name[4:]}%|{row["book_train"]["top1_in_book"]:.1%}|{row["book_validation"]["top1_in_book"]:.1%}|{row["standard_first6"]["top1_in_book"]:.1%}|{row["retention"]["policy_ce"]:.3f}|{row["retention"]["value_mse"]:.3f}|')
  lines+=['','三组相同38400样本暴露总量，普通样本和开局样本从共享候选序列取前缀，实际开局份额及策略损失权重一致。无开局伪胜负标签。原始NN测量关闭外部开局库，GUI开局库功能独立保留。','前1—3/4—6/7—10步细分、曲线、哈希和采样审计见result.json与PNG/SVG/CSV。固定最终模型验收，不按验证最优步挑模型。']
  (a.out/'REPORT.md').write_text('\n'.join(lines));log('本轮完成，选定'+str(selected)+'，升级'+str(result['promoted']));status('complete',selected=selected,promoted=result['promoted'])
 except BaseException:status('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
