"""Focused reproductions; patches live only inside this process."""
import sys,inspect,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np, torch
from unittest.mock import patch
import alphazero_nn as az, self_play_resnet_mcts as rz
from nn_torch import NetTorch
from audit_alphazero import audit_counts

torch.set_num_threads(1)
result={}
# Exact numerical comparison against autograd, no weight decay ambiguity.
p=az.NetParams(seed=3,filters=4)
b=az.cc.init_board();ms=az.all_moves_indexed(b,1)
smp={'s':az.encode_board(b,1),'moves':ms,'pi':np.ones(len(ms),dtype=np.float32)/len(ms),'z':1.}
loss,grad=az.compute_loss_and_grads(p,[smp],weight_decay=0)
m=NetTorch.from_numpy_params(p)
pl,v=m(torch.from_numpy(smp['s'])[None])
logits=torch.stack([pl[0,s]+pl[0,81+d] for s,d,_ in ms])
tl=-(torch.tensor(smp['pi'])*torch.log_softmax(logits,0)).sum()+(v[0]-1)**2
tl.backward()
ptgrad=[m.conv1.weight.grad,m.conv1.bias.grad,m.conv2.weight.grad,m.conv2.bias.grad,m.p1.weight.grad.T,m.p1.bias.grad,m.p2.weight.grad.T,m.p2.bias.grad,m.v1.weight.grad.T,m.v1.bias.grad,m.v2.weight.grad.T,m.v2.bias.grad]
result['numpy_vs_torch']={'loss_numpy':loss,'loss_torch':float(tl.detach()),'max_gradient_error':max(float(np.max(np.abs(g-t.detach().numpy()))) for g,t in zip(grad,ptgrad))}
# Check current production implementations; original failure remains in the audit report.
result['counting'] = {
    'mcts_think': audit_counts(az, False),
    'mcts_think_resnet': audit_counts(rz, True),
}
# Toy game: A is an immediate P1 win. B lets P2 choose P1 loss or P1 win.
def probe(module,resnet):
 def moves(b,p):
  edges={0:[1,2],2:[3,4]}.get(b[0],[])
  return [(b[0],d,[b[0],d]) for d in edges]
 winners={1:1,3:2,4:1}
 created=[];Original=module.Node
 def factory(*args):
  n=Original(*args);created.append(n);return n
 f=lambda *a,**kw:(np.zeros(162,dtype=np.float32),0.)
 with patch.object(module,'Node',factory),patch.object(module,'all_moves_indexed',moves),patch.object(module.cc,'apply_move',lambda b,p:[p[-1]]),patch.object(module.cc,'is_game_over',lambda b:winners.get(b[0])),patch.object(np.random,'dirichlet',lambda a:np.ones(len(a))/len(a)):
  if resnet:
   with patch.object(module,'resnet_forward',f):out=module.mcts_think_resnet(None,[0],1,128)
  else:
   with patch.object(module,'encode_board',lambda *a:None),patch.object(module,'forward',f):out=module.mcts_think(None,[0],1,128)
 root=created[0];opponent=root.children[1][3]
 return {'root_selected':out[-1][1][-1], 'expected_root_selected':1,'opponent_Q_for_P1_loss':opponent.children[0][3].Q,'opponent_Q_for_P1_win':opponent.children[1][3].Q}
result['perspective_numpy']=probe(az,False)
result['perspective_resnet']=probe(rz,True)
print(json.dumps(result,indent=2))
Path(__file__).with_name('az_components.json').write_text(json.dumps(result,indent=2))
