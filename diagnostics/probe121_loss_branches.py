"""Bounded fixed-depth expert root checks and paired factual continuations."""
import argparse,copy,json,sys,time,hashlib,importlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rules121 as r
from diagnostics.analyze121_losses import measure

def replay(prefix):
 g=r.Game()
 for move in prefix:g.play(move)
 return g

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,required=True);a=ap.parse_args();p=a.folder;native=importlib.import_module('clysto121_native');start=time.monotonic();audits=[];branches=[];snap=p/'probe_source';snap.mkdir(exist_ok=True)
 for path in [Path(__file__),Path(native.__file__),Path('clysto121_generated/core.c'),Path('ccore121_tables.h')]:
  (snap/path.name).write_bytes(path.read_bytes())
 def status(stage,**kw):
  (p/'probe_status.json').write_text(json.dumps(dict(stage=stage,elapsed=time.monotonic()-start,**kw),ensure_ascii=False,indent=2))
  if time.monotonic()-start>3600:raise TimeoutError('一小时专家复查预算耗尽')
 def think(g,depth):
  records=[(list(b),side,count) for (b,side),count in g.repetitions.items() if count>0];s,e,value,reached,nodes,prunes,aborted=native.search(g.board,g.player,120.,depth,g.ply,records,g.max_ply)
  if s<0 or reached!=depth or aborted:raise RuntimeError('固定深度未完成，拒绝样本')
  path=next(v for v in r.legal_paths(g.board,s) if v[-1]==e);return path,value
 def forced(g,path):
  child=copy.deepcopy(g);player=g.player;child.play(path);winner,_=child.result()
  if winner is not None:return 0 if winner==0 else 100000 if winner==player else -100000
  return -think(child,4)[1]
 for seed in json.loads((p/'seeds.json').read_text()):
  status('固定深度5复查',completed=len(audits));g=replay(seed['opening']);preferred,value=think(g,5);actual_value=forced(g,seed['actual']);check=forced(g,preferred)
  if check!=value:raise ValueError('根最优与强制子分数不一致')
  audits.append(dict(seed_id=seed['seed_id'],pair=seed['pair'],side=seed['side'],ply=seed['ply'],phase=seed['phase'],expert=preferred,actual=seed['actual'],expert_score=value,actual_score=actual_value,gap=value-actual_value))
  (p/'expert_roots.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2))
 # At most two distinct opening groups per phase. Pick by verified fixed-depth gap.
 selected=[];used=set()
 for phase in ('opening','middle','endgame'):
  for row in sorted((x for x in audits if x['phase']==phase and x['gap']>=4),key=lambda x:(-x['gap'],x['seed_id'])):
   if row['pair'] in used:continue
   used.add(row['pair']);selected.append(row)
   if sum(x['phase']==phase for x in selected)==2:break
 seeds={s['seed_id']:s for s in json.loads((p/'seeds.json').read_text())}
 for row in selected:
  for role in ('actual','expert'):
   seed=seeds[row['seed_id']];g=replay(seed['opening']);g.play(row[role]);paths=[row[role]];trace=[]
   while g.result()[0] is None:
    status('双分支真实续弈',seed_id=row['seed_id'],role=role,ply=g.ply);move,_=think(g,4);g.play(move);paths.append(move)
    if len(paths)%20==0:trace.append(dict(ply=g.ply,candidate=measure(g.board,row['side']),opponent=measure(g.board,3-row['side'])))
   winner,reason=g.result();record=dict(seed_id=row['seed_id'],pair=row['pair'],side=row['side'],phase=row['phase'],role=role,opening=seed['opening'],paths=paths,winner=winner,reason=reason,plies=g.ply,final={side:measure(g.board,side) for side in (1,2)},trace=trace)
   branches.append(record);(p/'branches.json').write_text(json.dumps(branches,ensure_ascii=False,indent=2))
 # Verify complete actual paths and terminal labels independently before finishing.
 for b in branches:
  g=replay(b['opening']+b['paths'])
  if g.result()!=(b['winner'],b['reason']):raise ValueError('续弈终局审计失败')
 comparison=[]
 for row in selected:
  pair=[b for b in branches if b['seed_id']==row['seed_id']];out={b['role']:(1 if b['winner']==b['side'] else .5 if b['winner']==0 else 0) for b in pair};comparison.append(dict(seed_id=row['seed_id'],phase=row['phase'],actual_points=out['actual'],expert_points=out['expert'],improved=out['expert']>out['actual'],worse=out['expert']<out['actual']))
 result=dict(root_checks=len(audits),verified_gap4=sum(x['gap']>=4 for x in audits),by_phase={phase:dict(checked=sum(x['phase']==phase for x in audits),gap4=sum(x['phase']==phase and x['gap']>=4 for x in audits)) for phase in ('opening','middle','endgame')},paired_continuations=len(selected),branches=len(branches),comparisons=comparison,improved=sum(x['improved'] for x in comparison),worse=sum(x['worse'] for x in comparison),seconds=time.monotonic()-start,native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),scope='根固定完成深度5/强制子4；两分支第一手不同，其余双方均固定深度4专家续弈，保留真实终局。')
 (p/'expert_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));status('complete');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
