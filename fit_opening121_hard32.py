"""固定32个第7—10步持续难例：纯专项与旧能力保护对照。"""
import argparse,copy,fcntl,hashlib,json,os,random,shutil,subprocess,time,traceback
from collections import deque,Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import rules121 as r
from az121 import Net121,action_id,encode
from opening121 import OpeningBook,MIRROR,key
from train_opening121 import book_samples,mastery,set_ce
from opening121_ratios import mixed_loss,stage_of
from curriculum121 import batch,atomic,replay,samples_from_game
from canonical121_trial import canonical_sample
from cycle121_targeted import expert_samples
from generalization121_curve import remove_validation_duplicates
from unfreeze121_trial import metrics
from diagnostics.audit121_opening_visibility import find_pairs
from checkpoint121_save import REGISTRY,digest,verify
SOURCE=Path('nn_checkpoints/experiments/opening121_ratios_20261104_fixed')

def orbit(state):
 raw,p=state.split(':');b=[0]*121
 for i,v in enumerate(map(int,raw)):b[MIRROR[i]]=v
 return min(state,key(b,int(p)))

def prefix_map(book):
 q=deque([(r.Game(),[])]);seen={};visited=set()
 while q:
  g,prefix=q.popleft();identity=key(g.board,g.player)
  if identity in visited:continue
  visited.add(identity);seen.setdefault(identity,prefix)
  reflected=[0]*121
  for i,v in enumerate(g.board):reflected[MIRROR[i]]=v
  seen.setdefault(key(reflected,g.player),[[MIRROR[i] for i in move] for move in prefix])
  if len(prefix)>=19:continue
  for move in book.moves(g.board,g.player,g.ply,g.repetitions):q.append((replay(prefix+[move]),prefix+[move]))
 return seen

@torch.no_grad()
def predict(model,samples,step=0):
 x,*_=batch(samples);l,_=model(x);out=[]
 for i,s in enumerate(samples):
  a=s['actions'][int(l[i,s['actions']].argmax())];good=max(s['positive'],key=lambda a:float(l[i,a]));other=[a for a in s['actions'] if a not in s['positive']];margin=float(l[i,good]-l[i,other].max()) if other else None
  out.append(dict(step=step,decision_id=s['decision_id'],chosen=a,correct=a in s['positive'],book_probability=float((torch.logsumexp(l[i,s['positive']],0)-torch.logsumexp(l[i,s['actions']],0)).exp()),book_minus_nonbook_margin=margin))
 return out

def balanced_orbits(candidates,seed=110413):
 pool=sorted(candidates,key=lambda x:x['orbit']);random.Random(seed).shuffle(pool);branches=Counter();stages=Counter();chosen=[]
 for i in range(16):
  player=1+i%2;eligible=[(j,c) for j,c in enumerate(pool) if c['player']==player]
  if not eligible:raise ValueError('没有足够的双方镜像难例，保留部分数据，不复制充数')
  j,c=min(eligible,key=lambda v:(stages[v[1]['own_move']],branches[v[1]['branch']],v[0]));pool.pop(j);chosen.append(c);stages[c['own_move']]+=1;branches[c['branch']]+=1
 return chosen

def select32(base,previous,sets,book):
 samples=sets['train'];a=predict(base,samples);b=predict(previous,samples);failed={s['decision_id'] for s,x,y in zip(samples,a,b) if stage_of(s)>=7 and not x['correct'] and not y['correct']};groups={};prefixes=prefix_map(book)
 for s in samples:
  if s['decision_id'] in failed:groups.setdefault(orbit(s['decision_id']),[]).append(s)
 candidates=[]
 for identity,ss in groups.items():
  if len(ss)!=2:continue
  if any(s['decision_id'] not in prefixes for s in ss):continue
  first=ss[0];prefix=prefixes[first['decision_id']];branch=orbit(key(replay(prefix[:6]).board,replay(prefix[:6]).player));candidates.append(dict(orbit=identity,player=int(first['decision_id'].split(':')[1]),own_move=stage_of(first),branch=branch,samples=ss))
 selected=balanced_orbits(candidates);fixed=[s for c in selected for s in c['samples']];metadata=[]
 for c in selected:
  for s in c['samples']:
   prefix=prefixes[s['decision_id']];g=replay(prefix);actions=[action_id(start,path[-1],g.player,True) for start,path in r.get_all_moves(g.board,g.player)];positive={action_id(path[0],path[-1],g.player,True) for path in book.moves(g.board,g.player,g.ply,g.repetitions)}
   if actions!=s['actions'] or positive!=set(s['positive']) or not np.array_equal(s['x'],encode(g.board,g.player,g.ply,g.repetitions,canonical=True)):raise ValueError('难例输入、动作或标签审计失败')
   metadata.append(dict(decision_id=s['decision_id'],player=g.player,own_move=stage_of(s),prefix=prefix,branch=c['branch'],orbit=c['orbit'],positive_actions=s['positive']))
 if len(fixed)!=32 or len({s['decision_id'] for s in fixed})!=32:raise ValueError('样本数/唯一性错误')
 return fixed,metadata,dict(persistent_candidates=len(failed),eligible_mirror_orbits=len(candidates),branches=len({c['branch'] for c in selected}),players=dict(Counter(c['player'] for c in metadata)),own_moves=dict(Counter(c['own_move'] for c in metadata)))

def fit_loss(model,reference,roots,ordinary,arm):
 if arm=='mixed':return mixed_loss(model,reference,roots,ordinary)
 if arm!='root_only':raise ValueError('unknown arm')
 x,*_=batch(roots);l,_=model(x);return .25*set_ce(l,roots)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();torch.set_num_threads(1);saved=verify(Path('nn_checkpoints/saves/v3_before_unfreeze_20261030'))
 if digest(REGISTRY)!=saved['registry_sha256']:raise ValueError('冠军变化')
 entry=json.loads(REGISTRY.read_text());model_path=Path(entry['checkpoint']);base=Net121.load(model_path);base.requires_grad_(False);previous=Net121.load(SOURCE/'book75_step2400.pt');book=OpeningBook();sets,groups=book_samples(book);fixed,metadata,selection=select32(base,previous,sets,book)
 args.out.mkdir(parents=True,exist_ok=False);lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic()
 def status(stage,**kw):
  atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-started,**kw))
  if stage not in ('complete','failed') and time.monotonic()-started>3600:raise TimeoutError('一小时预算耗尽')
 def log(text):
  with (args.out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-started:.1f}秒：{text}\n')
 names=['fit_opening121_hard32.py','diagnostics/plot121_opening_hard32.py','diagnostics/plot_style_zh.py','tests/test_opening121_hard32.py','opening121_ratios.py','train_opening121.py','opening121.py','opening121_book.json','az121.py','rules121.py','curriculum121.py','canonical121_trial.py','cycle121_targeted.py','generalization121_curve.py','unfreeze121_trial.py','diagnostics/audit121_opening_visibility.py']
 cfg=dict(profile=r.PROFILE,model=str(model_path.resolve()),model_sha256=digest(model_path),registry_sha256=digest(REGISTRY),previous_model=str(SOURCE/'book75_step2400.pt'),previous_sha256=digest(SOURCE/'book75_step2400.pt'),seed=110413,steps=2400,lr=3e-5,root_batch=4,root_ce_weight=.25,selection=selection,
  scope='同32难例/根序列/2400更新/初值。纯专项只算4根，mixed加12普通及原真实value/KL/函数保留；根暴露相同，计算量不同。无对战无部署。',
  interpretation='高拟合>=31/32，基本拟合>=29/32；检查局部不可区分对。混合低于纯专项说明保护目标组合改变拟合；纯专项不优化value头参数但共享CNN仍影响价值输出。',max_seconds=3600,source_sha256={n:digest(n) for n in names})
 for n in names:
  dst=args.out/'source'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,dst)
 data=args.out/'data_sources';data.mkdir();source=SOURCE/'data_sources';cfg['data_sha256']={}
 for n in ('old_teacher.jsonl','old_split.json','normal_games.jsonl'):
  shutil.copyfile(source/n,data/n);cfg['data_sha256'][n]=digest(data/n)
 atomic(args.out/'config.json',cfg);atomic(args.out/'fixed32.json',dict(metadata=metadata,selection=selection));torch.save(dict(samples=[{**s,'x':torch.as_tensor(s['x'])} for s in fixed]),data/'fixed32.pt');cfg['data_sha256']['fixed32.pt']=digest(data/'fixed32.pt');atomic(args.out/'config.json',cfg)
 try:
  aliases=find_pairs(fixed);atomic(args.out/'visibility_audit.json',dict(contradictory_pairs=len(aliases),details=aliases))
  old={'train':[],'validation':[]};normal={'train':[],'validation':[]};split=json.loads((data/'old_split.json').read_text())
  for line in (data/'old_teacher.jsonl').read_text().splitlines():
   row=json.loads(line);part=next((p for p in old if row['group'] in split[p]),None)
   if part:old[part].extend({**canonical_sample(s),'value_weight':1.} for s in samples_from_game(row))
  for line in (data/'normal_games.jsonl').read_text().splitlines():
   row=json.loads(line);normal[row['split']].extend(expert_samples(row))
  for pool in (old,normal):pool['train'],_=remove_validation_duplicates(pool['train'],sets['validation'])
  def measure(m):return dict(fixed32=mastery(m,fixed),other_train=mastery(m,[s for s in sets['train'] if s['decision_id'] not in {s['decision_id'] for s in fixed}]),validation=mastery(m,sets['validation']),retention=metrics(m,old['validation']),normal=metrics(m,normal['validation']))
  before=measure(base);atomic(args.out/'baseline.json',before);result={};audits={};log('32例标签与合法走法核验通过：'+json.dumps(selection,ensure_ascii=False))
  for arm in ('root_only','mixed'):
   model=copy.deepcopy(base);model.requires_grad_(True)
   if arm=='root_only':model.value.requires_grad_(False)
   opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-5,weight_decay=1e-4);rr=np.random.default_rng(110414);ro=np.random.default_rng(110415);seq=hashlib.sha256();ordinary_seq=hashlib.sha256();curve=[];history=[];exposures=Counter()
   for step in range(2401):
    if step%100==0:
     history+=predict(model,fixed,step);curve.append(dict(step=step,**measure(model)));atomic(args.out/(arm+'_curve.json'),curve);atomic(args.out/(arm+'_cases.json'),history);status('后段32难例拟合',arm=arm,step=step,total=2400)
     if step%600==0:model.save(args.out/(arm+f'_step{step}.pt'))
     if step==2400:break
    ri=rr.integers(32,size=4);oi=ro.integers(len(old['train']),size=6);ni=ro.integers(len(normal['train']),size=6);roots=[fixed[int(i)] for i in ri];ordinary=[old['train'][int(i)] for i in oi]+[normal['train'][int(i)] for i in ni];seq.update(ri.tobytes());ordinary_seq.update(oi.tobytes()+ni.tobytes());exposures.update(s['decision_id'] for s in roots)
    loss=fit_loss(model,base,roots,ordinary,arm)
    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
   result[arm]=curve[-1];audits[arm]=dict(root_sequence_sha256=seq.hexdigest(),ordinary_sequence_sha256=ordinary_seq.hexdigest(),root_exposures=sum(exposures.values()),per_root=dict(exposures),ordinary_used=0 if arm=='root_only' else 28800);atomic(args.out/'sampling_audit.json',audits);log(arm+'完成：'+json.dumps(curve[-1],ensure_ascii=False))
  if audits['root_only']['root_sequence_sha256']!=audits['mixed']['root_sequence_sha256']:raise ValueError('根抽样序列不一致')
  if audits['root_only']['ordinary_sequence_sha256']!=audits['mixed']['ordinary_sequence_sha256']:raise ValueError('普通候选序列不一致')
  if digest(REGISTRY)!=cfg['registry_sha256'] or digest(model_path)!=cfg['model_sha256']:raise ValueError('冠军/基线变化')
  output=dict(final=result,baseline=before,selection=selection,visibility_pairs=len(aliases),promoted=False,arena_run=False,seconds=time.monotonic()-started);atomic(args.out/'result.json',output)
  p=result['root_only']['fixed32']['top1_in_book'];m=result['mixed']['fixed32']['top1_in_book'];decision='两组均达到基本拟合，可安排扩大训练集与全新分支验证。' if min(p,m)>=.9 else '纯专项可拟合、混合配方限制拟合，优先调整保护目标与训练安排。' if p>=.9 else '纯专项也未达到基本拟合，先查表示与目标/优化问题，不追加数据量。';output['decision']=decision;atomic(args.out/'result.json',output)
  plot=subprocess.run(['/Library/Frameworks/Python.framework/Versions/3.10/bin/python3','diagnostics/plot121_opening_hard32.py',str(args.out)],capture_output=True,text=True);atomic(args.out/'plot_status.json',dict(exit_code=plot.returncode,stderr=plot.stderr))
  if plot.returncode:raise RuntimeError('中文图表失败')
  lines=['# 第7—10步32难例拟合','',decision,'','|模型|固定32命中|留出128命中|旧策略CE|旧价值MSE|','|---|---:|---:|---:|---:|']
  for label,v in [('v3',before)]+list(result.items()):lines.append(f'|{label}|{v["fixed32"]["top1_in_book"]:.1%}|{v["validation"]["top1_in_book"]:.1%}|{v["retention"]["policy_ce"]:.3f}|{v["retention"]["value_mse"]:.3f}|')
  lines+=['',json.dumps(selection,ensure_ascii=False),'每组根暴露9600次、同根顺序/系数，纯专项计算量较少；纯专项价值头参数无更新，共享CNN仍会改变价值输出。每100步记录的逐例首选保留。这里只诊断训练拟合，不注册模型。'];(args.out/'REPORT.md').write_text('\n'.join(lines));log(decision);status('complete',decision=decision)
 except BaseException:status('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
