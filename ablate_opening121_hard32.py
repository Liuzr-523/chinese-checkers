"""固定开局32难例的2×2损失拆分；诊断用途，不选冠军。"""
import argparse,copy,fcntl,json,os,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from az121 import Net121
from curriculum121 import batch,atomic,samples_from_game
from canonical121_trial import canonical_sample
from cycle121_targeted import expert_samples
from generalization121_curve import remove_validation_duplicates
from opening121 import OpeningBook
from train_opening121 import book_samples,mastery,set_ce
from fit_opening121_hard32 import predict
from unfreeze121_trial import metrics
from checkpoint121_save import REGISTRY,digest
ARMS=('root_only','with_replay','with_aux','full')
SOURCE=Path('nn_checkpoints/experiments/fit_opening121_hard32_20261104')

def terms(model,reference,roots,ordinary):
 ss=roots+ordinary;x,ids,mask,target,z=batch(ss);logits,value=model(x);n=len(roots);legal=logits.gather(1,ids).masked_fill(~mask,-1e9)
 root=.25*set_ce(logits[:n],roots);replay=.75*F.cross_entropy(legal[n:],target[n:]);weights=torch.tensor([s['value_weight'] for s in ss]);genuine=((value-z).square()*weights).sum()/weights.sum().clamp_min(1)
 with torch.no_grad():ol,ov=reference(x);old=ol.gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)
 lp=legal.log_softmax(1);aux=genuine+.1*(old.exp()*(old-lp)).sum(1).mean()+.1*F.mse_loss(value,ov)
 return root,replay,aux

def combine(parts,arm):
 if arm not in ARMS:raise ValueError(arm)
 a,b,c=parts
 return a+(b if arm in ('with_replay','full') else 0)+(c if arm in ('with_aux','full') else 0)

def gradient_audit(parts,params):
 vectors=[]
 for loss in parts:
  gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
  vectors.append(torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(params,gs)]))
 a,b,c=vectors
 return dict(root_norm=float(a.norm()),replay_norm=float(b.norm()),aux_norm=float(c.norm()),root_replay_cosine=float(F.cosine_similarity(a,b,dim=0)),root_aux_cosine=float(F.cosine_similarity(a,c,dim=0)))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();torch.set_num_threads(1)
 cfg=json.loads((SOURCE/'config.json').read_text());modelpath=Path(cfg['model']);registry=digest(REGISTRY)
 if registry!=cfg['registry_sha256'] or digest(modelpath)!=cfg['model_sha256']:raise ValueError('现役或初值已变化')
 args.out.mkdir(parents=True,exist_ok=False);lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic()
 def status(stage,**kw):
  atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),seconds=time.monotonic()-start,**kw))
  if time.monotonic()-start>3600 and stage not in ('complete','failed'):raise TimeoutError('一小时预算')
 def log(s):
  with (args.out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-start:.1f}秒：{s}\n')
 sources=['ablate_opening121_hard32.py','fit_opening121_hard32.py','opening121_ratios.py','train_opening121.py','az121.py','curriculum121.py','canonical121_trial.py','cycle121_targeted.py','generalization121_curve.py','opening121.py','opening121_book.json','rules121.py','unfreeze121_trial.py','checkpoint121_save.py','tests/test_opening121_loss_ablation.py','diagnostics/plot121_opening_ablation.py','diagnostics/plot_style_zh.py']
 for n in sources:
  dst=args.out/'source'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,dst)
 shutil.copytree(SOURCE/'data_sources',args.out/'data_sources');shutil.copyfile(SOURCE/'fixed32.json',args.out/'fixed32.json')
 atomic(args.out/'config.json',dict(arms=list(ARMS),source=str(SOURCE),model=str(modelpath),model_sha256=digest(modelpath),registry_sha256=registry,source_sha256={n:digest(n) for n in sources},data_sha256={p.name:digest(p) for p in (args.out/'data_sources').iterdir()},steps=2400,root_weight=.25,replay_weight=.75,lr=3e-5,root_seed=110414,ordinary_seed=110415,max_seconds=3600,scope='固定原32例；四组同初值/序列/2400更新；保护项整体作为一个因素。每100步测量，200步记梯度，最终不部署不对战。',criteria='基本拟合>=29/32；旧策略CE/旧价值MSE各<=v3+0.05；只诊断，旧留出不作为新冠军验收。'))
 try:
  data=args.out/'data_sources';fixed=torch.load(data/'fixed32.pt',weights_only=True)['samples'];old={'train':[],'validation':[]};normal={'train':[],'validation':[]};split=json.loads((data/'old_split.json').read_text())
  for line in (data/'old_teacher.jsonl').read_text().splitlines():
   row=json.loads(line);part=next((p for p in old if row['group'] in split[p]),None)
   if part:old[part].extend({**canonical_sample(s),'value_weight':1.} for s in samples_from_game(row))
  for line in (data/'normal_games.jsonl').read_text().splitlines():
   row=json.loads(line);normal[row['split']].extend(expert_samples(row))
  sets,_=book_samples(OpeningBook())
  for pool in (old,normal):pool['train'],_=remove_validation_duplicates(pool['train'],sets['validation'])
  base=Net121.load(modelpath);base.requires_grad_(False)
  def measure(m):return dict(fixed32=mastery(m,fixed),validation=mastery(m,sets['validation']),retention=metrics(m,old['validation']),normal=metrics(m,normal['validation']))
  before=measure(base);atomic(args.out/'baseline.json',before);final={};audits={};gradients=[];log('四组开始；原32训练难例及哈希核验，2400更新，不运行对战。')
  import hashlib
  for arm in ARMS:
   m=copy.deepcopy(base);m.requires_grad_(True)
   if arm in ('root_only','with_replay'):m.value.requires_grad_(False)
   opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=3e-5,weight_decay=1e-4);rr=np.random.default_rng(110414);ro=np.random.default_rng(110415);seq=hashlib.sha256();curve=[];cases=[];clipped=0;normsum=0.;losses=[]
   for step in range(2401):
    if step%100==0:
     curve.append(dict(step=step,**measure(m)));cases+=predict(m,fixed,step);atomic(args.out/(arm+'_curve.json'),curve);atomic(args.out/(arm+'_cases.json'),cases);status('损失拆分',arm=arm,step=step)
     if step%600==0:m.save(args.out/f'{arm}_step{step}.pt')
     if step==2400:break
    ri=rr.integers(32,size=4);oi=ro.integers(len(old['train']),size=6);ni=ro.integers(len(normal['train']),size=6);seq.update(ri.tobytes()+oi.tobytes()+ni.tobytes());roots=[fixed[int(i)] for i in ri];ordinary=[old['train'][int(i)] for i in oi]+[normal['train'][int(i)] for i in ni];parts=terms(m,base,roots,ordinary);loss=combine(parts,arm)
    if not torch.isfinite(loss):raise ValueError('非有限损失')
    if step%200==0:gradients.append(dict(arm=arm,step=step,**gradient_audit(parts,[p for n,p in m.named_parameters() if p.requires_grad and not n.startswith('value.')])));atomic(args.out/'gradients.json',gradients)
    if step%100==0:losses.append(dict(step=step,root=float(parts[0].detach()),replay=float(parts[1].detach()),aux=float(parts[2].detach())))
    opt.zero_grad();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1));normsum+=norm;clipped+=norm>1;opt.step()
   final[arm]=curve[-1];audits[arm]=dict(sequence_sha256=seq.hexdigest(),root_exposures=9600,ordinary_loss_exposures=28800 if arm in ('with_replay','full') else 0,ordinary_aux_exposures=28800 if arm in ('with_aux','full') else 0,clipped_updates=clipped,mean_preclip_norm=normsum/2400);atomic(args.out/'sampling_and_clipping.json',audits);atomic(args.out/(arm+'_losses.json'),losses);log(f'{arm}完成：{round(final[arm]["fixed32"]["top1_in_book"]*32)}/32；旧CE {final[arm]["retention"]["policy_ce"]:.3f}')
  if len({a['sequence_sha256'] for a in audits.values()})!=1:raise ValueError('采样序列变化')
  if digest(REGISTRY)!=registry or digest(modelpath)!=cfg['model_sha256']:raise ValueError('冠军变化')
  reproduction={a:abs(final[a]['fixed32']['top1_in_book']-json.loads((SOURCE/'result.json').read_text())['final'][b]['fixed32']['top1_in_book']) for a,b in [('root_only','root_only'),('full','mixed')]}
  atomic(args.out/'result.json',dict(baseline=before,final=final,reproduction_top1_difference=reproduction,seconds=time.monotonic()-start,promoted=False,arena_run=False));status('complete');log('四组结束；未注册模型，现役权重与注册表哈希不变。')
 except BaseException:status('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
