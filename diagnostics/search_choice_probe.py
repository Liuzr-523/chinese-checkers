"""Compare 64 and 512 on the same recorded states; heuristics are diagnostics only."""
import json,pickle,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
import Chinese_Checker as cc
import engine
from nn_resnet import NetResNet
from self_play_resnet_mcts import mcts_think_resnet,resnet_forward
root=Path('nn_checkpoints/experiments/search_budget_20261022')
torch.set_num_threads(1);model=NetResNet.load(root/'model.pt').eval();results=[];start=time.monotonic()
for game_id in (3,4,9,10,15,16,21,22):
 rec=pickle.loads((root/'games'/f'{game_id:03}.pkl').read_bytes());board=cc.init_board();history=set()
 for ply,step in enumerate(rec['steps']):
  player=step['player'];history.add((tuple(board),player))
  if ply in (12,32):
   choices={}
   for n in (64,512):
    (moves,pi),(_,path)=mcts_think_resnet(model,board,player,n,discount=.995,history=history)
    after=cc.apply_move(board,path);_,v=resnet_forward(model,after,3-player)
    choices[str(n)]={'path':path,'immediate_value':-v,'rule_v2':engine.evaluate_v2(after,player),'top_visit_fraction':max(pi)}
   results.append({'game':game_id,'ply':ply,'player':player,'choices':choices})
   print(f"{len(results)}/16 完成",flush=True)
  board=cc.apply_move(board,step['move_path'])
changes=[x for x in results if x['choices']['64']['path']!=x['choices']['512']['path']]
summary={'positions':len(results),'different':len(changes),'mean_v2_difference_512_minus_64':float(np.mean([x['choices']['512']['rule_v2']-x['choices']['64']['rule_v2'] for x in changes])) if changes else None,'mean_network_difference_512_minus_64':float(np.mean([x['choices']['512']['immediate_value']-x['choices']['64']['immediate_value'] for x in changes])) if changes else None,'seconds':time.monotonic()-start,'caveat':'Non-independent positions, static heuristics are not ground-truth and do not establish cause.'}
out={'summary':summary,'positions':results};(root/'diagnostic_choice_probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(summary),flush=True)
