"""Extract reachable original Clysto book branches; validate each edge in strict121."""
import argparse,hashlib,json
from collections import deque,Counter
from pathlib import Path
import Chinese_Checker as old
import rules121 as r
import clysto_book as source
from engine_jumpstar import LOCAL_TO_STAR,STAR_TO_LOCAL

def key(board,player):return ''.join(map(str,board))+':'+str(player)
def export(max_ply=20):
 queue=deque([(old.init_board(),1,0)]);seen=set();nodes={};rejected=[];depths=Counter();edges=0
 while queue:
  b,p,depth=queue.popleft();identity=(tuple(b),p)
  if identity in seen or depth>=max_ply:continue
  seen.add(identity)
  if len(seen)>50000:raise RuntimeError('bounded extraction exceeded50000 states')
  board=[0]*121
  for i,v in enumerate(b):board[LOCAL_TO_STAR[i]]=v
  candidates=source.book_moves(b,p,old.get_all_moves);moves=[]
  for s,e in candidates:
   start,end=LOCAL_TO_STAR[s],LOCAL_TO_STAR[e];path=next((path for path in r.legal_paths(board,start) if path[-1]==end),None)
   if path is None:rejected.append(dict(depth=depth,player=p,start=start,end=end));continue
   if path not in moves:moves.append(path)
   child=old.apply_move(b,[s,e]);queue.append((child,3-p,depth+1))
  if moves:nodes[key(board,p)]=dict(depth=depth,moves=moves);depths[depth]+=1;edges+=len(moves)
 manifest=dict(schema='opening121-v1',profile=r.PROFILE,max_ply=max_ply,nodes=nodes,stats=dict(hit_states=len(nodes),edges=edges,visited_states=len(seen),by_depth=dict(depths),rejected_edges=rejected),source_sha256={name:hashlib.sha256((Path(source.CLYSTO_DIR)/'src'/name).read_bytes()).hexdigest() for name in ('book.hpp','constants.hpp')},scope='Original Clysto opening entries mapped to full121 states; every retained move validated under strict121. First10 moves per side=20 plies.')
 return manifest
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('opening121_book.json'));a=ap.parse_args();data=export();a.out.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n');print(json.dumps(data['stats'],ensure_ascii=False,indent=2))
