"""Exact-board opening lookup. Raw engines remain available for fair evaluation."""
import json,random
from functools import lru_cache
from pathlib import Path
import rules121 as r
DEFAULT=Path(__file__).resolve().with_name('opening121_book.json')
MIRROR={i:r.IDS[(-q-y,y)] for i,(q,y) in enumerate(r.COORDS)}
def key(board,player):return ''.join(map(str,board))+':'+str(player)
@lru_cache(None)
def load(path=DEFAULT):
 data=json.loads(Path(path).read_text())
 if data.get('schema')!='opening121-v1' or data.get('profile')!=r.PROFILE:raise ValueError('开局库规则版本不匹配')
 return data
class OpeningBook:
 def __init__(self,path=DEFAULT):self.data=load(Path(path))
 def moves(self,board,player,ply=0,repetitions=None):
  r.validate_board(board)
  if player not in (1,2):raise ValueError('invalid player')
  if ply>=self.data['max_ply'] or r.terminal(board,player,ply,repetitions)[0] is not None:return []
  entry=self.data['nodes'].get(key(board,player));mirror=False
  if entry is None:
   reflected=[0]*121
   for i,v in enumerate(board):reflected[MIRROR[i]]=v
   entry=self.data['nodes'].get(key(reflected,player));mirror=True
  if entry is None:return []
  paths=[[MIRROR[i] for i in p] if mirror else p.copy() for p in entry['moves']]
  return [p for p in paths if r.valid_path(board,player,p)]
 def choose(self,board,player,ply=0,repetitions=None):
  moves=self.moves(board,player,ply,repetitions)
  # Favor the demonstrated three-move formation when the full position and book agree.
  for start,end in ((6,15),(1,28),(5,39)):
   if player==2:start,end=120-start,120-end
   hit=next((p for p in moves if (p[0],p[-1])==(start,end)),None)
   if hit:return hit
  return moves[0] if moves else None
class BookEngine:
 def __init__(self,engine,book=None):self.engine=engine;self.book=book or OpeningBook();self.name=getattr(engine,'name',type(engine).__name__)+'＋开局库'
 def think(self,board,player,ply=0,repetitions=None):
  path=self.book.choose(board,player,ply,repetitions)
  if path is not None:return path,dict(engine=self.name,source='opening_book',book_hit=True,book_candidates=len(self.book.moves(board,player,ply,repetitions)),rule_profile=r.PROFILE)
  path,info=self.engine.think(board,player,ply,repetitions);return path,{**info,'book_hit':False,'source':info.get('source','search')}
 def __getattr__(self,name):return getattr(self.engine,name)
