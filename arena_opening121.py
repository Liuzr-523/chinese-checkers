"""Frozen v3 with/without book; two standard-start games plus20 paired book branches."""
import argparse,json,random,time,os
from collections import deque,Counter
from pathlib import Path
import torch
import rules121 as r
from opening121 import OpeningBook,BookEngine,key
from az121 import Neural121
from neural121_registry import load_registered
from curriculum121 import atomic,replay,finish
from iterate121 import summary

def schedule(book):
 g=r.Game();q=deque([(g,[])]);seen=set();pool=[]
 while q:
  g,prefix=q.popleft();identity=key(g.board,g.player)
  if identity in seen:continue
  seen.add(identity)
  if 4<=len(prefix)<=12:pool.append(prefix)
  if len(prefix)>=12:continue
  for path in book.moves(g.board,g.player,g.ply,g.repetitions):
   child=replay(prefix+[path]);q.append((child,prefix+[path]))
 random.Random(110405).shuffle(pool);return pool[:20]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);book=OpeningBook();prefixes=schedule(book);base=load_registered();allrows=[];started=time.monotonic();atomic(a.out/'schedule.json',dict(standard=[[]],branches=prefixes))
 for stage,ps in [('standard',[[]]),('branches',prefixes)]:
  rows=[]
  for pair,prefix in enumerate(ps):
   for side in (1,2):
    atomic(a.out/'status.json',dict(stage=stage,pid=os.getpid(),completed=len(rows),total=2*len(ps),elapsed=time.monotonic()-started));row=finish(replay(prefix),{side:BookEngine(Neural121(base.model,32)),3-side:Neural121(base.model,32)});row.update(pair=pair,side=side,opening=prefix,stage=stage);rows.append(row);allrows.append(row)
    with (a.out/(stage+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
 result={stage:summary([r for r in allrows if r['stage']==stage]) for stage in ('standard','branches')};result['seconds']=time.monotonic()-started;result['unique_game_path_sequences']=len({json.dumps(r['paths']) for r in allrows})
 for row in allrows:
  g=replay(row['opening']+row['paths'])
  if g.result()!=(row['winner'],row['reason']):raise ValueError('terminal replay failed')
 atomic(a.out/'result.json',result);atomic(a.out/'status.json',dict(stage='complete',result=result));print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
