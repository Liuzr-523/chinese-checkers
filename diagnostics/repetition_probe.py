"""Replay previous failed games and compare the actual repeated move with history-aware search."""
import sys,pickle,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import Chinese_Checker as cc
from nn_resnet import NetResNet
from self_play_resnet_mcts import mcts_think_resnet
from mcts_history import position_key

torch.set_num_threads(1)
run=Path('nn_checkpoints/experiments/selfplay_20261022_a')
models={1:NetResNet.load(run/'candidate.pt').eval(),2:NetResNet.load(run/'baseline.pt').eval()}
results=[]
for gi,rec in enumerate(pickle.loads((run/'evaluation.pkl').read_bytes())):
 if rec['reason']!='repetition':continue
 b=cc.init_board();seen={position_key(b,1)}
 for ply,st in enumerate(rec['steps']):
  p=st['player'];nxt=cc.apply_move(b,st['move_path']);key=position_key(nxt,3-p)
  if key in seen:
   model=models[1 if p==rec['candidate_side'] else 2]
   _,(_,path)=mcts_think_resnet(model,b,p,32,history=seen)
   results.append(dict(game=gi+1,ply=ply+1,old_move=st['move_path'],new_move=path,new_repeats=position_key(cc.apply_move(b,path),3-p) in seen))
   break
  seen.add(key);b=nxt
print(json.dumps(results,indent=2))
Path('diagnostics/repetition_probe.json').write_text(json.dumps(results,indent=2))
