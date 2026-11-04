"""Inspect logged losses and compare frozen decision rules on identical states."""
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rules121 as r
from az121 import Net121,encode,search,action_id
from engine_c121 import CSearch121

ROOT=Path(__file__).resolve().parents[1]
class ZeroValue:
    def __init__(self,model):self.model=model
    def eval(self):self.model.eval();return self
    def __call__(self,x):
        logits,value=self.model(x);return logits,torch.zeros_like(value)

@torch.no_grad()
def policy(model,g):
    moves=r.get_all_moves(g.board,g.player)
    ids=[action_id(s,p[-1],g.player,getattr(model,'canonical',False)) for s,p in moves]
    logits,value=model(torch.from_numpy(encode(g.board,g.player,g.ply,g.repetitions,canonical=getattr(model,'canonical',False))).unsqueeze(0))
    return moves[int(logits[0,ids].argmax())][1],float(value[0])

def main():
    torch.set_num_threads(1)
    source=ROOT/'nn_checkpoints/experiments/selfplay121_round1_20261029'
    model=Net121.load(source/'candidate.pt');teacher=CSearch121(seconds=5,max_depth=3)
    rows=[json.loads(line) for line in (source/'candidate_vs_c.jsonl').read_text().splitlines()]
    summaries={name:Counter() for name in ['policy','mcts16','mcts64','mcts64_zero_value']}
    examples=[];values=[];began=time.monotonic()
    for index,game in enumerate(rows):
        side=game['model_side'];states=[];g=r.Game()
        for step in game['steps']:
            assert g.board==step['board'] and g.player==step['player']
            if g.player==side and g.ply>=4:states.append((g.board.copy(),g.player,g.ply,g.repetitions.copy(),step['path']))
            g.play(step['path'])
        if g.result()!=(game['winner'],game['reason']):raise ValueError('log outcome mismatch')
        picks=sorted(set([0,len(states)//2]+list(range(max(0,len(states)-3),len(states)))))
        for which in picks:
            b,p,ply,history,actual=states[which]
            g=r.Game(board=b,player=p,ply=ply);g.repetitions=history.copy()
            _,best=teacher.think(b,p,ply,history)
            assert not best['aborted']
            path,val=policy(model,g);values.append(dict(value=val,winner=game['winner'],side=side,ply=ply))
            choices={'policy':path,'mcts16':search(model,b,p,ply,history,16)[0],
                     'mcts64':search(model,b,p,ply,history,64)[0],
                     'mcts64_zero_value':search(ZeroValue(model),b,p,ply,history,64)[0]}
            for name,path in choices.items():
                child=r.apply_move(b,path);counts=history.copy();counts[(tuple(child),3-p)]+=1
                _,ci=CSearch121(seconds=5,max_depth=2).think(child,3-p,ply+1,counts)
                assert not ci['aborted'];value=-ci['score'];s=summaries[name]
                s['positions']+=1;s['exact_depth3_optimal']+=value==best['score']
                s['avoidable_forced_loss']+=value<=-100000 and best['score']>-100000
                s['missed_forced_win']+=best['score']>=100000 and value<100000
                s['regret_clipped']+=min(100,max(0,best['score']-value))
                if name=='mcts16' and value<=-100000 and best['score']>-100000 and len(examples)<8:
                    examples.append(dict(game=index,ply=ply,board=b,player=p,chosen=path,root_score=best['score'],chosen_score=value))
    result=dict(logged_games=len(rows),decisions=len(values),decision_modes={k:dict(v) for k,v in summaries.items()},
                optimistic_in_c_losses=sum(x['value']>0 and x['winner'] not in (0,x['side']) for x in values),
                c_loss_positions=sum(x['winner'] not in (0,x['side']) for x in values),examples=examples,seconds=time.monotonic()-began,
                limits='Fixed-depth C heuristic is a diagnostic reference, not a game-theoretic oracle. Logged loss states are not a promotion test.')
    target=ROOT/'nn_checkpoints/experiments/diagnose121_20261030';target.mkdir(exist_ok=True)
    (target/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
