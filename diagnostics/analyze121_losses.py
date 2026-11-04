"""Replay strict121 games and locate phase deficits. Standard library only."""
import argparse,json,hashlib,sys
from collections import Counter,deque
from functools import lru_cache
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rules121 as r

@lru_cache(None)
def distances(start):
 d={start:0};q=deque([start])
 while q:
  a=q.popleft()
  for b in r.NEIGHBORS[a]:
   if b not in d:d[b]=d[a]+1;q.append(b)
 return d

@lru_cache(100000)
def position_cost(pieces,player):
 goal=r.GOAL[player];remaining=[i for i in pieces if i not in goal];targets=sorted(goal-set(pieces));n=len(remaining)
 if not n:return 0
 dp={0:0}
 for piece in remaining:
  nxt={};ds=distances(piece)
  for mask,v in dp.items():
   for j,target in enumerate(targets):
    if mask>>j&1:continue
    k=mask|1<<j;value=v+ds[target]
    if value<nxt.get(k,100000):nxt[k]=value
  dp=nxt
 return dp[(1<<n)-1]

def measure(board,player):
 pieces=tuple(i for i,p in enumerate(board) if p==player);goal=r.GOAL[player]
 return dict(goal=sum(i in goal for i in pieces),home=sum(i in r.HOME[player] for i in pieces),assignment=position_cost(pieces,player),rear=max(min(distances(i)[t] for t in goal) for i in pieces),lateral=sum(i not in r.HOME[1]|r.HOME[2] and max(abs(r.COORDS[i][0]),abs(r.COORDS[i][1]),abs(sum(r.COORDS[i])))>4 for i in pieces))

def phase(g,player):
 if measure(g.board,player)['goal']>=7:return 'endgame'
 return 'opening' if g.ply<24 else 'middle'

def replay_game(row):
 g=r.Game();snapshots={};turns=[];first8={};prefix=[]
 for move in row['opening']+row['paths']:
  player=g.player;before=measure(g.board,player);other=measure(g.board,3-player);stage=phase(g,player);at=g.ply;g.play(move);after=measure(g.board,player)
  if player==row['side'] and at>=row['start_ply']:
   turns.append(dict(pair=row['pair'],side=row['side'],ply=at,phase=stage,opening=prefix.copy(),actual=move,before=before,after=after,other=other,gain=before['assignment']-after['assignment'],deficit=before['assignment']-other['assignment']))
  prefix.append(move)
  for p in (1,2):
   if p not in first8 and measure(g.board,p)['goal']>=8:first8[p]=g.ply
  if g.ply in (20,40,60,80):snapshots[g.ply]={p:measure(g.board,p) for p in (1,2)}
 if g.result()!=(row['winner'],row['reason']) or g.ply!=row['plies']:raise ValueError('棋谱终局不一致')
 return dict(pair=row['pair'],side=row['side'],winner=row['winner'],outcome='win' if row['winner']==row['side'] else 'draw' if row['winner']==0 else 'loss',plies=g.ply,snapshots=snapshots,first8=first8,final={p:measure(g.board,p) for p in (1,2)},turns=turns)

def analyze(rows):
 games=[replay_game(row) for row in rows];pairs={p:[g for g in games if g['pair']==p] for p in sorted({g['pair'] for g in games})};double=[p for p,gs in pairs.items() if len(gs)==2 and all(g['outcome']=='loss' for g in gs)];stage_summary={}
 for outcome in ('win','loss','draw'):
  stage_summary[outcome]={}
  for ply in (20,40,60,80):
   data=[g for g in games if g['outcome']==outcome and ply in g['snapshots']]
   stage_summary[outcome][ply]=dict(n=len(data),mean_assignment_deficit=sum(g['snapshots'][ply][g['side']]['assignment']-g['snapshots'][ply][3-g['side']]['assignment'] for g in data)/max(1,len(data)),mean_goal_deficit=sum(g['snapshots'][ply][3-g['side']]['goal']-g['snapshots'][ply][g['side']]['goal'] for g in data)/max(1,len(data)))
 # One representative per game and phase, then a bounded shortlist per phase.
 pool=[]
 for g in games:
  if g['pair'] not in double:continue
  for stage in ('opening','middle','endgame'):
   candidates=[t for t in g['turns'] if t['phase']==stage]
   if candidates:pool.append(min(candidates,key=lambda t:(t['gain'], -t['deficit'],t['ply'])))
 seeds=[]
 for stage in ('opening','middle','endgame'):
  candidates=sorted((t for t in pool if t['phase']==stage),key=lambda t:(t['gain'],-t['deficit'],t['pair'],t['side']))
  seen=set()
  for t in candidates:
   if t['pair'] in seen:continue
   seen.add(t['pair']);seeds.append({**t,'seed_id':len(seeds)})
   if len(seen)>=8:break
 summary=dict(games=len(games),outcomes=dict(Counter(g['outcome'] for g in games)),double_loss_pairs=double,stage_summary=stage_summary,losing_final_goal_counts=dict(Counter(g['final'][g['side']]['goal'] for g in games if g['outcome']=='loss')),losing_reached8=sum(g['side'] in g['first8'] for g in games if g['outcome']=='loss'),selected_seeds=len(seeds),definitions='assignment为占用无关的邻接步数最小匹配，已入营棋子锁定；阶段：自身>=7子入营为收官，否则总手数<24为开局，其余中盘。正缺口表示候选落后。')
 return summary,games,seeds

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--games',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=list(map(json.loads,a.games.read_text().splitlines()));summary,games,seeds=analyze(rows)
 for name,value in [('summary',summary),('games',games),('seeds',seeds)]: (a.out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2))
 (a.out/'input_sha256.txt').write_text(hashlib.sha256(a.games.read_bytes()).hexdigest()+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
