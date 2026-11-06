"""Versioned 121-cell policy/value network and history-aware PUCT.

All values are from the encoded side-to-move perspective. Bootstrap examples
are labelled explicitly as C search imitation; later MCTS games are self-play.
"""
from collections import Counter
from pathlib import Path
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import rules121 as r

FORMAT = 'az121-v1'
ENCODING = 'axial17x17-6planes-pair121x121-v1'
CANONICAL_ENCODING = 'axial17x17-6planes-own-home-top-pair121x121-v2'
ROWS = torch.tensor([y+8 for q,y in r.COORDS])
COLS = torch.tensor([q+8 for q,y in r.COORDS])

def encode(board, player, ply=0, repetitions=None, canonical=False):
    r.validate_board(board)
    if player not in (1,2): raise ValueError('invalid player')
    x = np.zeros((6,17,17), dtype=np.float32)
    repeat = (repetitions or {}).get((tuple(board),player), 1)
    for i,(q,y) in enumerate(r.COORDS):
        x[:,y+8,q+8] = (board[i]==player, board[i]==3-player, 1,
                         player==1, min(ply,240)/240, min(repeat,3)/3)
    return x[:,::-1,::-1].copy() if canonical and player==2 else x

def action_id(start,end,player,canonical=False):
    if canonical and player==2:start,end=120-start,120-end
    return start*121+end

class Net121(nn.Module):
    def __init__(self, width=32, value_head="pooled", canonical=False):
        super().__init__(); self.width=width
        if value_head not in ("pooled","spatial","spatial_norm"): raise ValueError("unknown value head")
        self.value_head=value_head
        self.canonical=bool(canonical)
        self.encoding=CANONICAL_ENCODING if self.canonical else ENCODING
        self.stem=nn.Conv2d(6,width,3,padding=1)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.ReLU(),nn.Conv2d(width,width,3,padding=1)) for _ in range(2)])
        self.src=nn.Linear(width,16); self.dst=nn.Linear(width,16)
        self.bias=nn.Linear(width,2)
        self.value=nn.Sequential(nn.Linear(width,32),nn.ReLU(),nn.Linear(32,1),nn.Tanh()) if value_head=="pooled" else nn.Sequential(nn.Linear(width*121,128),nn.ReLU(),nn.Linear(128,1),nn.Tanh())
        if value_head=="spatial_norm":
            self.value=nn.Sequential(nn.LayerNorm(width*121),nn.Linear(width*121,128),nn.ReLU(),nn.Linear(128,1),nn.Tanh())
            nn.init.zeros_(self.value[-2].weight);nn.init.zeros_(self.value[-2].bias)
        self.register_buffer('rows',ROWS.clone()); self.register_buffer('cols',COLS.clone())

    def forward(self,x):
        h=F.relu(self.stem(x))
        for block in self.blocks: h=F.relu(h+block(h))
        cells=h[:,:,self.rows,self.cols].transpose(1,2)
        bias=self.bias(cells)
        logits=torch.bmm(self.src(cells),self.dst(cells).transpose(1,2))/4
        logits=logits+bias[:,:,0].unsqueeze(2)+bias[:,:,1].unsqueeze(1)
        value_input=cells.mean(1) if self.value_head=="pooled" else cells.flatten(1)
        return logits.flatten(1),self.value(value_input).squeeze(1)

    def save(self,path):
        torch.save(dict(format=FORMAT,encoding=self.encoding,profile=r.PROFILE,width=self.width,value_head=self.value_head,
                        state_dict=self.state_dict()),path)

    @classmethod
    def load(cls,path):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data.get('encoding') not in (ENCODING,CANONICAL_ENCODING) or any(data.get(k)!=v for k,v in [('format',FORMAT),('profile',r.PROFILE)]):
            raise ValueError('checkpoint is not compatible with strict 121-cell rules')
        model=cls(data['width'],value_head=data.get('value_head','pooled'),canonical=data['encoding']==CANONICAL_ENCODING);model.load_state_dict(data['state_dict']);return model.eval()

def safe_root_actions(board,player,ply=0,repetitions=None):
    """Keep immediate wins; otherwise avoid an opponent's immediate winning reply.

    This is an optional two-ply tactical search, not a change to legal rules.
    If every move loses, retain all moves. Strict goals precede history draws.
    """
    import ccore121
    r.validate_board(board)
    moves=ccore121.moves(board,player);wins=[];safe=[];unsafe=[]
    for start,end in moves:
        child=r.apply_move(board,[start,end]);outcome=ccore121.terminal(child,3-player)
        if outcome==player:
            wins.append((start,end));continue
        if outcome==3-player:
            unsafe.append((start,end));continue
        if outcome==0 or ply+1>=240 or (repetitions or {}).get((tuple(child),3-player),0)>=2:
            safe.append((start,end));continue
        losing=False
        for src,dst in ccore121.moves(child,3-player):
            reply=r.apply_move(child,[src,dst])
            if ccore121.terminal(reply,player)==3-player:
                losing=True;break
        (unsafe if losing else safe).append((start,end))
    allowed=wins or safe or moves
    return set(allowed),dict(legal=len(moves),immediate_wins=len(wins),unsafe=len(unsafe),allowed=len(allowed))


class Node:
    def __init__(self,prior=1):
        self.prior=prior;self.visits=0;self.total=0.;self.children=None
    @property
    def q(self): return self.total/self.visits if self.visits else 0.

@torch.no_grad()
def search(model,board,player,ply=0,repetitions=None,simulations=16,rng=None,noise=False,value_scale=1.,root_safety=False,heuristic_mix=0.,fpu_mode="zero",policy_temperature=1.,time_limit=None):
    started=time.monotonic()
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit<=0):raise ValueError('invalid search time limit')
    if simulations<1: raise ValueError('simulations must be positive')
    if not math.isfinite(policy_temperature) or policy_temperature<=0: raise ValueError('invalid policy temperature')
    if fpu_mode not in ('zero','parent'): raise ValueError('unknown first-play baseline')
    if not math.isfinite(value_scale) or not 0 <= value_scale <= 1: raise ValueError('value_scale must be between 0 and 1')
    if not math.isfinite(heuristic_mix) or not 0 <= heuristic_mix <= 1: raise ValueError('heuristic_mix must be between 0 and 1')
    rng = np.random.default_rng() if rng is None else rng
    model.eval();root=Node()
    history=Counter(repetitions or {(tuple(board),player):1})
    guard=None
    allowed=None
    if root_safety:
        allowed,guard=safe_root_actions(board,player,ply,history)
    def expand(node,b,p,n,counts):
        winner,reason=r.terminal(b,p,n,counts)
        if winner is not None: return 0. if winner==0 else 1. if winner==p else -1.
        moves=r.get_all_moves(b,p)
        if node is root and allowed is not None:
            moves=[(s,path) for s,path in moves if (s,path[-1]) in allowed]
        if not moves: return 0.
        ids=[action_id(s,path[-1],p,getattr(model,'canonical',False)) for s,path in moves]
        logits,value=model(torch.from_numpy(encode(b,p,n,counts,canonical=getattr(model,'canonical',False))).unsqueeze(0))
        probabilities=(logits[0,ids]/policy_temperature).softmax(0).cpu().numpy()
        node.children=[(path,Node(float(prior))) for (_,path),prior in zip(moves,probabilities)]
        estimate=float(value[0])*value_scale
        if heuristic_mix:
            import ccore121
            heuristic=math.tanh(ccore121.evaluate(b,p)/40.)
            estimate=(1-heuristic_mix)*estimate+heuristic_mix*heuristic
        return estimate
    expand(root,board,player,ply,history)
    if root.children is None: return None,dict(nodes=0,root_visits=0,policy=[],paths=[])
    if noise:
        perturb=rng.dirichlet(np.full(len(root.children),.3))
        for (_,child),extra in zip(root.children,perturb): child.prior=.75*child.prior+.25*float(extra)
    completed=0;depths=[]
    for _ in range(simulations):
        # Always complete one simulation; a single expansion can exceed the deadline.
        if completed and time_limit is not None and time.monotonic()-started>=time_limit:break
        node=root;b=board;p=player;n=ply;counts=history.copy();trace=[root]
        while node.children:
            path,child=max(node.children,key=lambda edge:(-edge[1].q if edge[1].visits else (node.q if fpu_mode=='parent' else 0.))+1.5*edge[1].prior*math.sqrt(node.visits+1)/(1+edge[1].visits))
            b=r.apply_move(b,path);p=3-p;n+=1;counts[(tuple(b),p)]+=1
            node=child;trace.append(node)
        value=expand(node,b,p,n,counts)
        for visited in reversed(trace):
            visited.visits+=1;visited.total+=value;value=-value
        completed+=1;depths.append(len(trace)-1)
    visits=np.array([child.visits for _,child in root.children],dtype=np.float64)
    policy=visits/visits.sum()
    index=int(np.argmax(visits))
    return root.children[index][0],dict(nodes=completed,root_visits=root.visits,
        **({'elapsed_seconds':time.monotonic()-started} if time_limit is not None else {}),
        time_limit=time_limit,root_visited_actions=int((visits>0).sum()),
        max_depth=max(depths),mean_depth=float(np.mean(depths)),visits=visits.tolist(),
        child_q=[child.q for _,child in root.children],
        child_visits=int(visits.sum()),root_safety=guard,root_q=root.q,priors=[child.prior for _,child in root.children],policy=policy.tolist(),paths=[path for path,_ in root.children])

def train(model,samples,steps=100,seed=0):
    if not samples: raise ValueError('no terminal-labelled samples')
    if any(s.get('encoding',ENCODING)!=model.encoding for s in samples): raise ValueError('sample/checkpoint encoding mismatch')
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    model.train();losses=[]
    for step in range(steps):
        batch=[samples[i] for i in rng.integers(len(samples),size=min(16,len(samples)))]
        x=torch.from_numpy(np.stack([s['x'] for s in batch]));logits,value=model(x)
        ce=[]
        for i,s in enumerate(batch):
            ids=torch.tensor(s['actions'],dtype=torch.long);target=torch.tensor(s['policy'],dtype=torch.float32)
            ce.append(-(logits[i,ids].log_softmax(0)*target).sum())
        z=torch.tensor([s['z'] for s in batch],dtype=torch.float32)
        weights=torch.tensor([s.get('value_weight',1.) for s in batch],dtype=torch.float32)
        value_loss=((value-z).square()*weights).sum()/weights.sum().clamp_min(1)
        loss=torch.stack(ce).mean()+value_loss
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        losses.append(float(loss.detach()))
    model.eval();return dict(first_loss=losses[0],last_loss=losses[-1],steps=steps)

class Neural121:
    name='121格神经网络 MCTS'
    def __init__(self,model,simulations=16,value_scale=1.,root_safety=False,heuristic_mix=0.,fpu_mode="zero",policy_temperature=1.):
        if not math.isfinite(value_scale) or not 0 <= value_scale <= 1: raise ValueError("invalid value_scale")
        if fpu_mode not in ('zero','parent'): raise ValueError('unknown first-play baseline')
        if not math.isfinite(policy_temperature) or policy_temperature<=0: raise ValueError('invalid policy temperature')
        self.policy_temperature=policy_temperature
        self.fpu_mode=fpu_mode
        self.model=model.eval();self.simulations=simulations;self.value_scale=value_scale;self.root_safety=root_safety
        if not math.isfinite(heuristic_mix) or not 0 <= heuristic_mix <= 1: raise ValueError('invalid heuristic_mix')
        self.heuristic_mix=heuristic_mix
    def think(self,board,player,ply=0,repetitions=None):
        return search(self.model,board,player,ply,repetitions,self.simulations,value_scale=self.value_scale,root_safety=self.root_safety,heuristic_mix=self.heuristic_mix,fpu_mode=self.fpu_mode,policy_temperature=self.policy_temperature)
