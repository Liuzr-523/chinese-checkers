"""本轮最终模型在同40新开发状态上的MCTS32实际走法核验。"""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from az121 import Net121,Neural121,action_id
from curriculum121 import replay,atomic
from corrected121_decisions import score_action,ConsistentExpert
from checkpoint121_save import digest,REGISTRY
P=Path('nn_checkpoints/experiments/dynamic121_actual_20261104');torch.set_num_threads(1);cfg=json.loads((P/'config.json').read_text());rows=list(map(json.loads,(P/'fresh_development.jsonl').read_text().splitlines()));cache={}
for r in rows:cache[r['decision_id'],action_id(r['expert_path'][0],r['expert_path'][-1],r['player'],True)]=r['expert_score']
for line in (P/'new_scores.jsonl').read_text().splitlines():
 r=json.loads(line);cache[r['decision_id'],r['action']]=r['score']
ConsistentExpert.failure_dir=P;queries=0;out={};reg=digest(REGISTRY)
for name,path in [('v3',Path(cfg['model']))]+[(f'{arm}_{seed}',P/f'{arm}_{seed}_step1600.pt') for seed in cfg['seeds'] for arm in ('control','dynamic')]:
 engine=Neural121(Net121.load(path),32);details=[];began=time.monotonic()
 for r in rows:
  g=replay(r['opening']);move,info=engine.think(g.board,g.player,g.ply,g.repetitions);a=action_id(move[0],move[-1],g.player,True);key=r['decision_id'],a
  if key not in cache:
   cache[key]=score_action(g,move);queries+=1
   with (P/'search_probe_new_scores.jsonl').open('a') as f:f.write(json.dumps(dict(decision_id=r['decision_id'],action=a,score=cache[key]))+'\n')
  regret=r['expert_score']-cache[key]
  if regret<0:raise ValueError('评分高于固定老师全局最佳')
  details.append(dict(decision_id=r['decision_id'],action=a,regret=regret,root_visits=info['root_visits']))
 out[name]=dict(bad_moves=sum(x['regret']>=4 for x in details),equal_best=sum(x['regret']==0 for x in details),mean_regret=sum(x['regret'] for x in details)/40,seconds=time.monotonic()-began,details=details)
if digest(REGISTRY)!=reg:raise ValueError('冠军变化')
atomic(P/'search_probe.json',dict(models=out,new_queries=queries,simulations=32,source_sha256=digest(__file__),registry_unchanged=True));(P/'search_probe_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({name:{k:v for k,v in x.items() if k!='details'} for name,x in out.items()},ensure_ascii=False,indent=2))
