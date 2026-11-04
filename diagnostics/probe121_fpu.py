"""Diagnose unvisited-child zero baseline using registered NN-v1 loss states."""
from collections import Counter
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from az121 import Net121,search
from engine_c121 import CSearch121
from curriculum121 import replay
import rules121 as r

def main():
    torch.set_num_threads(1);model=Net121.load('nn_checkpoints/models/checkpoints/neural121_v1_correct.pt')
    source=Path('nn_checkpoints/experiments/tactical121_guard_20261030/fixed_c_control.jsonl')
    games=[json.loads(x) for x in source.read_text().splitlines()]
    result={mode:{str(p):Counter() for p in (1,2)} for mode in ['zero','parent']}
    for game in games:
        g=replay(game['opening']);states=[]
        for path in game['paths']:
            if g.player==game['side']:states.append((g.board.copy(),g.player,g.ply,g.repetitions.copy()))
            g.play(path)
        for index in sorted(set([0,len(states)//2]+list(range(max(0,len(states)-3),len(states))))):
            board,p,ply,counts=states[index]
            best_path,best=CSearch121(seconds=5,max_depth=3).think(board,p,ply,counts)
            if best['aborted']:raise RuntimeError('root reference timed out')
            for mode in ('zero','parent'):
                path,info=search(model,board,p,ply,counts,32,fpu_mode=mode)
                actions=[(x[0],x[-1]) for x in info['paths']];teacher=(best_path[0],best_path[-1]);rank=sorted(range(len(actions)),key=lambda i:info['priors'][i],reverse=True).index(actions.index(teacher))+1
                visited=[i for i,v in enumerate(info['policy']) if v>0]
                child=r.apply_move(board,path);history=counts.copy();history[(tuple(child),3-p)]+=1
                _,value=CSearch121(seconds=5,max_depth=2).think(child,3-p,ply+1,history)
                if value['aborted']:raise RuntimeError('child reference timed out')
                s=result[mode][str(p)];s['positions']+=1;s['visited_actions']+=len(visited);s['legal_actions']+=len(actions)
                s['teacher_move_visited']+=actions.index(teacher) in visited;s['teacher_rank_sum']+=rank
                s['depth3_equal']+=best['score']==-value['score'];s['regret_clipped']+=min(100,max(0,best['score']+value['score']))
    out=Path('nn_checkpoints/experiments/probe121_fpu_20261030');out.mkdir(exist_ok=False)
    plain={mode:{p:dict(s) for p,s in players.items()} for mode,players in result.items()}
    (out/'result.json').write_text(json.dumps(plain,indent=2));print(json.dumps(plain,indent=2))
if __name__=='__main__':main()
