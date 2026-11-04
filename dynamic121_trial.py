"""动态实际首选纠错：160训练根、40全新开发状态、双种子固定预算。"""
import argparse,copy,fcntl,hashlib,json,os,shutil,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import rules121 as r
from az121 import Net121,Neural121,encode,action_id
from curriculum121 import atomic,batch,replay,finish,samples_from_game
from corrected121_decisions import ConsistentExpert,score_action,root_metrics
from canonical121_trial import canonical_sample
from cycle121_targeted import expert_samples
from generalization121_curve import remove_validation_duplicates
from diagnostics.fit121_objectives import objective_terms
from unfreeze121_trial import metrics
from multicandidate121_trial import witness
from confirm121_unfrozen import fresh_opening,seen_openings,start_key
from checkpoint121_save import verify,digest,REGISTRY,restore
from engine_c121 import CSearch121
from iterate121 import summary,paired_gate
PARENT=Path('nn_checkpoints/experiments/multicandidate121_20261101');SEEDS=(110401,110402)

def dynamic_rank(logits,samples,score):
 losses=[];events=[]
 for i,s in enumerate(samples):
  actual=s['actions'][int(logits[i,s['actions']].detach().argmax())];preferred=s['actions'][s['target']];gap=score(s,preferred)-score(s,actual)
  active=gap>=4
  losses.append(F.relu(.5-logits[i,preferred]+logits[i,actual]) if active else logits[i,preferred]*0)
  events.append(dict(decision_id=s['decision_id'],action=actual,gap=gap,active=active))
 return torch.stack(losses).mean(),events

def development_gate(results,baseline):
 checks={}
 for seed in SEEDS:
  old=results[f'control_{seed}'];new=results[f'dynamic_{seed}'];a=old['fresh40_actual'];b=new['fresh40_actual']
  checks[str(seed)]=dict(fewer_bad=b['bad_moves']<=a['bad_moves']-4,lower_regret=b['mean_regret']<=a['mean_regret']-1,good_moves=b['equal_best']>=a['equal_best'],retention=new['retention']['policy_ce']<=min(old['retention']['policy_ce']+.03,baseline['retention']['policy_ce']+.05),value=new['retention']['value_mse']<=old['retention']['value_mse']+.03)
 return all(all(c.values()) for c in checks.values()),checks

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out;torch.set_num_threads(1)
 saved_dir=Path('nn_checkpoints/saves/v3_before_unfreeze_20261030');saved=verify(saved_dir);pcfg=json.loads((PARENT/'config.json').read_text());model_path=Path(pcfg['model'])
 if digest(REGISTRY)!=saved['registry_sha256'] or digest(model_path)!=pcfg['model_sha256']:raise ValueError('冠军/存档变化')
 out.mkdir(parents=True,exist_ok=False);lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic();queries=0
 def status(stage,**kw):
  atomic(out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-started,new_child_queries=queries,**kw))
  if stage not in ('complete','failed') and time.monotonic()-started>10800:raise TimeoutError('三小时预算耗尽')
 def log(s):
  with (out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-started:.1f}秒：{s}\n')
 def record(name,row):
  with (out/(name+'.jsonl')).open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 import clysto121_native,ccore121
 names=['dynamic121_trial.py','diagnostics/plot121_dynamic.py','diagnostics/plot_style_zh.py','tests/test_dynamic121.py','diagnostics/fit121_objectives.py','multicandidate121_trial.py','az121.py','curriculum121.py','corrected121_decisions.py','canonical121_trial.py','cycle121_targeted.py','generalization121_curve.py','unfreeze121_trial.py','confirm121_unfrozen.py','checkpoint121_save.py','engine_c121.py','engine_clysto121.py','iterate121.py','neural121_registry.py','rules121.py',Path(clysto121_native.__file__).name,Path(ccore121.__file__).name,'clysto121_generated/core.c']
 for name in ('rules121.py','engine_clysto121.py','engine_c121.py',Path(clysto121_native.__file__).name,'clysto121_generated/core.c'):
  if digest(name)!=pcfg['source_sha256'][name]:raise ValueError('老师来源变化')
 for name in names:
  dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
 data=out/'data_sources';data.mkdir();sources={n:PARENT/'data_sources'/n for n in ('multicandidate.pt','decisions.jsonl','old_teacher.jsonl','old_split.json','normal_games.jsonl')};sources['prior_scores.jsonl']=PARENT/'score_cache.jsonl'
 for n,p in sources.items():shutil.copyfile(p,data/n)
 cfg=dict(profile=r.PROFILE,encoding=pcfg['encoding'],model=str(model_path.resolve()),model_sha256=digest(model_path),registry_sha256=digest(REGISTRY),seeds=list(SEEDS),primary_seed=SEEDS[0],steps=1600,learning_rate=3e-5,root_batch=4,ordinary_batch=12,simulations=32,
  training='CNN固定深度专家策略蒸馏，动态约束当前实际首选；老师为Clysto评估派生121。',
  difference='只替换根排序项：原固定好坏对→每次前向的当前合法首选；专家分差>=4才用.5间隔hinge，否则该根排序项0。全批均值，不重采样不提高根份额。',
  preserved='根唯一专家CE .25、排序 .25、普通CE .75、真实value 1、KL .1、value保留 .1，AdamW/学习率/clip1一致；不改动量。',
  development='40个新状态/20个新开局组；随机开局长度6/10/14/18，及v3策略走一手后的状态，P1/P2各20，不按老师分差筛选，不回灌训练。最终1600只评一次，不选择检查点或种子。',
  gate='两个种子均新40差棋至少少4例、平均专家分差至少降1分、同最优分数不降；旧CE<=control+.03且<=v3+.05，旧value MSE<=control+.03。过关后固定第一种子独立对战。',
  confirmation_games=120,fixed_c_games_per_model=40,independent_gate_p=.01,max_seconds=10800,max_new_child_queries=2048,
  source_sha256={n:digest(n) for n in names},data_sha256={n:digest(data/n) for n in sources})
 atomic(out/'config.json',cfg)
 try:
  ConsistentExpert.failure_dir=out;baseline=Net121.load(model_path);baseline.requires_grad_(False);sets=torch.load(data/'multicandidate.pt',weights_only=True)['sets'];train=sets['train'];meta={x['decision_id']:x for x in map(json.loads,(data/'decisions.jsonl').read_text().splitlines())};cache={};games={}
  for s in sets['train']+sets['validation']:
   did=s['decision_id'];games[did]=replay(meta[did]['opening'])
   for a,v in s['candidate_scores'].items():cache[did,int(a)]=v
  for line in (data/'prior_scores.jsonl').read_text().splitlines():
   row=json.loads(line);cache[row['decision_id'],row['action']]=row['score']
  def score(s,a):
   nonlocal queries
   key=s['decision_id'],int(a)
   if key not in cache:
    if queries>=2048:raise RuntimeError('老师查询预算耗尽')
    status('动态专家评分',decision_id=s['decision_id']);v=score_action(games[s['decision_id']],witness(games[s['decision_id']],a));queries+=1
    if v>meta[s['decision_id']]['expert_score']:raise ValueError('强制动作评分高于固定根最优')
    cache[key]=v;record('new_scores',dict(decision_id=s['decision_id'],action=int(a),score=v))
   return cache[key]
  # Draw fresh grouped development inputs before training; labels are never used by the optimizer.
  used,history_sources=seen_openings('nn_checkpoints/experiments');used.update((tuple(g.board),g.player) for g in games.values());atomic(out/'excluded_history.json',history_sources);fresh=[];expert=ConsistentExpert()
  for group in range(20):
   prefix=fresh_opening(110404000+group*127,6+4*(group%4),used)
   for member in range(2):
    g=replay(prefix);did=10000+2*group+member;status('准备新开局开发集',completed=len(fresh),total=40);preferred,info=expert.think(g.board,g.player,g.ply,g.repetitions)
    actions=[action_id(a,p[-1],g.player,True) for a,p in r.get_all_moves(g.board,g.player)];positive=action_id(preferred[0],preferred[-1],g.player,True)
    s=dict(decision_id=did,x=encode(g.board,g.player,g.ply,g.repetitions,canonical=True),actions=actions,target=actions.index(positive),z=0.,value_weight=0.)
    if member and start_key(prefix) in used:raise ValueError('新开发状态重复')
    used.add(start_key(prefix));games[did]=g;meta[did]=dict(expert_score=info['score'],opening=copy.deepcopy(prefix),group=group,player=g.player);cache[did,positive]=info['score'];fresh.append(s)
    if score_action(g,preferred)!=info['score']:raise ValueError('新老师根与强制子分数不一致')
    record('fresh_development',dict(decision_id=did,opening=prefix,group=group,player=g.player,expert_path=preferred,expert_score=info['score']))
    if member==0:
     with torch.no_grad():x,*_=batch([s]);l,_=baseline(x);a=actions[int(l[0,actions].argmax())]
     prefix=prefix+[witness(g,a)]
  torch.save(dict(samples=fresh),data/'fresh_development.pt');cfg['data_sha256']['fresh_development.pt']=digest(data/'fresh_development.pt');shutil.copyfile(out/'fresh_development.jsonl',data/'fresh_development.jsonl');cfg['data_sha256']['fresh_development.jsonl']=digest(data/'fresh_development.jsonl');atomic(out/'config.json',cfg)
  old={'train':[],'validation':[]};normal={'train':[],'validation':[]};split=json.loads((data/'old_split.json').read_text())
  for line in (data/'old_teacher.jsonl').read_text().splitlines():
   row=json.loads(line);part=next((k for k in old if row['group'] in split[k]),None)
   if part:old[part].extend({**canonical_sample(s),'value_weight':1.} for s in samples_from_game(row))
  for line in (data/'normal_games.jsonl').read_text().splitlines():
   row=json.loads(line);normal[row['split']].extend(expert_samples(row))
  for pool in (old,normal):pool['train'],_=remove_validation_duplicates(pool['train'],sets['validation']+fresh)
  def measure(m):return dict(train=root_metrics(m,train),retention=metrics(m,old['validation']),normal=metrics(m,normal['validation']))
  @torch.no_grad()
  def actual(m,samples,label):
   rows=[]
   for s in samples:
    x,*_=batch([s]);l,_=m(x);a=s['actions'][int(l[0,s['actions']].argmax())];regret=meta[s['decision_id']]['expert_score']-score(s,a);rows.append(dict(decision_id=s['decision_id'],action=a,regret=regret))
   atomic(out/(label+'_actions.json'),rows);return dict(n=len(rows),bad_moves=sum(r['regret']>=4 for r in rows),equal_best=sum(r['regret']==0 for r in rows),mean_regret=sum(r['regret'] for r in rows)/len(rows))
  before=measure(baseline);before['fresh40_actual']=actual(baseline,fresh,'v3_fresh40');atomic(out/'baseline.json',before);results={};audits={};log('新40状态/20组准备完成。开始双种子原排序与动态实际差棋纠错，各1600更新；保留普通回放和旧能力保护。')
  for seed in SEEDS:
   for arm in ('control','dynamic'):
    name=f'{arm}_{seed}';m=copy.deepcopy(baseline);m.requires_grad_(True);opt=torch.optim.AdamW(m.parameters(),lr=3e-5,weight_decay=1e-4);rr=np.random.default_rng(seed+1);ro=np.random.default_rng(seed+2);seq=hashlib.sha256();curve=[];active_count=0;root_count=0
    for step in range(1601):
     if step%100==0:
      curve.append(dict(step=step,**measure(m)));atomic(out/(name+'_curve.json'),curve);status('训练对照',arm=name,step=step,total=1600)
      if step%400==0:m.save(out/(name+f'_step{step}.pt'))
      if step==1600:break
     ri=rr.integers(len(train),size=4);oi=ro.integers(len(old['train']),size=6);ni=ro.integers(len(normal['train']),size=6);seq.update(ri.tobytes()+oi.tobytes()+ni.tobytes());rs=[train[int(i)] for i in ri];ordinary=[old['train'][int(i)] for i in oi]+[normal['train'][int(i)] for i in ni];loss,terms=objective_terms(m,baseline,rs,ordinary,'full_objective')
     if arm=='dynamic':
      x,*_=batch(rs);l,_=m(x);rank,events=dynamic_rank(l,rs,score);loss=loss-terms['rank']+.25*rank;active_count+=sum(e['active'] for e in events);root_count+=len(events)
      if step%100==0:record('dynamic_examples',dict(arm=name,step=step,examples=events))
     if not torch.isfinite(loss):raise ValueError('训练损失非有限数')
     opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
    results[name]=curve[-1];audits[name]=dict(sequence_sha256=seq.hexdigest(),updates=1600,root_exposures=6400,ordinary_exposures=19200);atomic(out/'sampling_audit.json',audits);atomic(out/(name+'_dynamic_audit.json'),dict(active=active_count,total=root_count));log(name+'完成')
   if audits[f'control_{seed}']!=audits[f'dynamic_{seed}']:raise ValueError('两组抽样顺序不一致')
  for name in results:
   m=Net121.load(out/(name+'_step1600.pt'));results[name]['train_actual']=actual(m,train,name+'_train');results[name]['fresh40_actual']=actual(m,fresh,name+'_fresh40')
  passed,checks=development_gate(results,before);result=dict(final=results,baseline=before,development_passed=passed,development_gate=checks,qualified=False,promoted=False,arena_run=False)
  atomic(out/'development_result.json',result);log('开发筛选结果：'+str(passed))
  if passed:
   candidate=Net121.load(out/f'dynamic_{SEEDS[0]}_step1600.pt');candidate.save(out/'candidate.pt');schedule={name:[fresh_opening(110414000+offset+i*97,4+2*(i%4),used) for i in range(n)] for name,n,offset in [('confirmation',60,0),('fixed_c',20,10000)]};atomic(out/'arena_schedule.json',schedule);fixed={};matches=[]
   for stage,m,rival in [('confirmation',candidate,baseline),('fixed_c_control',baseline,None),('fixed_c_candidate',candidate,None)]:
    rows=[]
    for pair,prefix in enumerate(schedule['confirmation' if stage=='confirmation' else 'fixed_c']):
     for side in (1,2):
      status('独立对战',comparison=stage,completed=len(rows));other=Neural121(rival,32) if rival is not None else CSearch121(seconds=.03,max_depth=10);row=finish(replay(prefix),{side:Neural121(m,32),3-side:other});row.update(pair=pair,side=side,opening=prefix);rows.append(row);record(stage,row)
    if stage=='confirmation':matches=rows
    else:fixed[stage.removeprefix('fixed_c_')]=summary(rows)
   gate=paired_gate(matches);result.update(arena_run=True,confirmation=summary(matches),paired_gate=gate,fixed_c=fixed,qualified=gate['one_sided_p']<=.01 and fixed['candidate']['points']>=fixed['control']['points'])
  if digest(REGISTRY)!=cfg['registry_sha256'] or digest(model_path)!=cfg['model_sha256']:raise ValueError('冠军/基线被并发修改')
  for n,sha in cfg['source_sha256'].items():
   if digest(n)!=sha or digest(out/'source'/n)!=sha:raise ValueError('源代码变化：'+n)
  result.update(seconds=time.monotonic()-started,new_child_queries=queries);atomic(out/'result.json',result)
  if result['qualified']:
   from neural121_registry import register,load_registered
   try:
    entry=register(out,'v4_dynamic_actual');engine=load_registered()
    for player in (1,2):
     path,info=engine.think(r.init_board(),player)
     if not r.valid_path(r.init_board(),player,path) or info['root_visits']!=32:raise ValueError('登记后烟测失败')
    result['promoted']=True;atomic(out/'adoption.json',dict(version=entry['version'],adopted=True))
   except BaseException:
    active=json.loads(REGISTRY.read_text())
    if active.get('version')=='v4_dynamic_actual' and active.get('sha256')==digest(out/'candidate.pt'):restore(saved_dir)
    raise
  atomic(out/'result.json',result);plot=subprocess.run(['/Library/Frameworks/Python.framework/Versions/3.10/bin/python3','diagnostics/plot121_dynamic.py',str(out)],capture_output=True,text=True);atomic(out/'plot_status.json',dict(exit_code=plot.returncode,stderr=plot.stderr))
  if plot.returncode:raise RuntimeError('中文图表渲染失败')
  lines=['# 动态实际差棋纠错实验','',f'开发筛选：{passed}；独立对战：{result["arena_run"]}；升级：{result["promoted"]}。','160训练根保留原分组；新40状态来自20全新随机开局组及v3走一手后的状态，双方各20，不按难度挑选。双种子/各1600步，固定最终模型比较。','|方案|训练差棋/160|新开发差棋/40|新开发同最优分/40|新开发平均分差|旧策略交叉熵|','|---|---:|---:|---:|---:|---:|']
  for name,v in results.items():
   d=v['fresh40_actual'];label=('原方案' if name.startswith('control') else '动态纠错')+'／'+name.split('_')[-1];lines.append(f'|{label}|{v["train_actual"]["bad_moves"]}|{d["bad_moves"]}|{d["equal_best"]}|{d["mean_regret"]:.3f}|{v["retention"]["policy_ce"]:.3f}|')
  lines+=['','动态排序每次检查当前合法首选，专家分差>=4就施加原0.5间隔hinge；其他损失、样本暴露和优化器不变。根评分来自固定深度5的Clysto评估派生121专家。','完整候选、实时差棋示例、动作评分缓存、代码/数据/模型哈希、中文日志、曲线和全新开发前缀保存。只有通过新开局独立对战与审计才登记。']
  (out/'REPORT.md').write_text('\n'.join(lines));log('实验完成：开发筛选'+str(passed)+'；升级'+str(result['promoted']));status('complete',development_passed=passed,promoted=result['promoted'])
 except BaseException:status('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
