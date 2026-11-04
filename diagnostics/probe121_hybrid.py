"""Frozen registered model: test whether C static progress helps local decisions."""
from collections import Counter
import json
from pathlib import Path
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from az121 import Net121,search
from engine_c121 import CSearch121
from curriculum121 import replay
import rules121 as r

def main():
    torch.set_num_threads(1);began=time.monotonic()
    model=Net121.load('nn_checkpoints/models/checkpoints/neural121_v1_correct.pt')
    source=Path('nn_checkpoints/experiments/tactical121_guard_20261030')
    games=[json.loads(x) for x in (source/'fixed_c_control.jsonl').read_text().splitlines()]
    summaries={name:Counter() for name in ['learned','half_hand','hand']}
    guard_effect=Counter();trajectory_matches=0
    treated=[json.loads(x) for x in (source/'fixed_c_candidate.jsonl').read_text().splitlines()]
    for game,other in zip(games,treated):
        trajectory_matches+=game['paths']==other['paths']
        g=replay(game['opening']);states=[]
        for path in game['paths']:
            if g.player==game['side']:states.append((g.board.copy(),g.player,g.ply,g.repetitions.copy()))
            g.play(path)
        for i in sorted(set([0,len(states)//2]+list(range(max(0,len(states)-3),len(states))))):
            board,player,ply,counts=states[i]
            _,best=CSearch121(seconds=5,max_depth=3).think(board,player,ply,counts)
            if best['aborted']:raise RuntimeError('reference timeout')
            for name,mix in [('learned',0.),('half_hand',.5),('hand',1.)]:
                path,info=search(model,board,player,ply,counts,32,heuristic_mix=mix)
                child=r.apply_move(board,path);history=counts.copy();history[(tuple(child),3-player)]+=1
                _,value=CSearch121(seconds=5,max_depth=2).think(child,3-player,ply+1,history)
                if value['aborted']:raise RuntimeError('child timeout')
                regret=best['score']+value['score'];s=summaries[name]
                s['positions']+=1;s['depth3_equal']+=regret==0;s['regret_clipped']+=min(100,max(0,regret))
                s['avoidable_short_loss']+=value['score']>=100000 and best['score']>-100000
    out=Path('nn_checkpoints/experiments/probe121_hybrid_20261030');out.mkdir(exist_ok=False)
    result=dict(decision_modes={k:dict(v) for k,v in summaries.items()},c_guard_identical_trajectories=trajectory_matches,c_games=len(games),
                seconds=time.monotonic()-began,warning='Development loss-state probe, C heuristic reference not oracle. Hand-guided search is not learned value.')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
