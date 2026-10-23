"""Read-only audit; does not train or overwrite models/replay."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,pickle,collections
from unittest.mock import patch
import numpy as np
from checkpoint_paths import artifact_path, artifact_glob
import alphazero_nn as az
import self_play_resnet_mcts as rz


def audit_counts(module, resnet=False):
    nodes=[]
    Original=module.Node
    def factory(*args):
        n=Original(*args);nodes.append(n);return n
    def moves(board,player):
        n=board[0]
        return [(n,n*2+1,[n,n*2+1]),(n,n*2+2,[n,n*2+2])] if n<31 else []
    fake_forward=lambda *args,**kwargs:(np.zeros(162,dtype=np.float32),0.)
    with patch.object(module,'Node',factory), patch.object(module,'all_moves_indexed',moves), patch.object(module.cc,'apply_move',lambda b,p:[p[-1]]), patch.object(module.cc,'is_game_over',lambda b:None), patch.object(np.random,'dirichlet',lambda a:np.ones(len(a))/len(a)):
        if resnet:
            with patch.object(module,'resnet_forward',fake_forward):
                module.mcts_think_resnet(None,[0],1,n_simulations=32)
        else:
            with patch.object(module,'forward',fake_forward),patch.object(module,'encode_board',lambda *a:None):
                module.mcts_think(None,[0],1,n_simulations=32)
    root=nodes[0]
    return dict(simulations=32,root_N=root.N,root_children_N=[x[3].N for x in root.children],root_children_sum=sum(x[3].N for x in root.children),root_children_W=[x[3].W for x in root.children])


def audit_replay(path):
    with path.open('rb') as f:data=pickle.load(f)
    out={'file':path.name,'records':len(data)}
    if not data:return out
    if 'steps' not in data[0]:
        out['format']='samples';out['z']=dict(collections.Counter(str(float(s['z'])) for s in data))
        out['explicit_vw_zero']=sum(s.get('vw')==0 for s in data)
        return out
    out['format']='games';out['declared_winners']=dict(collections.Counter(str(r.get('winner')) for r in data))
    term=collections.Counter();bad_pi=0;illegal=0;steps=0;bad_turn=0;repeated_games=0
    for rec in data:
        board=az.cc.init_board();seen=set();repeated=False
        for i,st in enumerate(rec['steps']):
            player=st['player'];path=st['move_path'];steps+=1
            bad_turn+=player!=1+i%2
            key=(tuple(board),player)
            if key in seen:repeated=True
            seen.add(key)
            pi=np.asarray(st['pi'])
            bad_pi += len(pi)!=len(st['moves']) or not np.isfinite(pi).all() or abs(pi.sum()-1)>1e-4 or (pi<0).any()
            if i%20==0:
                legal={(s,p[-1]) for s,p in az.cc.get_all_moves(board,player)}
                illegal+= (path[0],path[-1]) not in legal
            board=az.cc.apply_move(board,path)
        term[str(az.cc.is_game_over(board))]+=1
        repeated_games+=repeated
    out.update(steps=steps,actual_terminal_winners=dict(term),bad_pi=int(bad_pi),sampled_illegal_moves=illegal,bad_turn=bad_turn,games_with_repeated_position=repeated_games)
    return out

if __name__=='__main__':
    out={'mcts_numpy':audit_counts(az),'mcts_resnet':audit_counts(rz,True),'replay':[]}
    base=Path(__file__).resolve().parents[1]/'nn_checkpoints'
    names=['replay.pkl','replay_nn_mcts_b.pkl','replay_nn_mcts_test.pkl','replay_clysto_merged.pkl','replay_book.pkl','replay_v2_aug.pkl','replay_v2_argmax_230.pkl']
    names += [p.name for p in sorted(artifact_glob(base, 'az_r*_*.pkl'))[:2]]
    for name in names:
        p=artifact_path(base/name)
        if p.exists():
            result=audit_replay(p);out['replay'].append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
    print(json.dumps(out,ensure_ascii=False,indent=2))
    Path(__file__).with_name('alphazero_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
