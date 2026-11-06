"""独立小树回传核验；不把有限PUCT的访问选择当作通用minimax证明。"""
import unittest
from contextlib import ExitStack
from unittest.mock import patch
from collections import Counter
import numpy as np
import torch
import az121 as az
import rules121 as r


class FixedModel:
    canonical=False
    def __init__(self,value=0.):self.value=value;self.calls=0
    def eval(self):return self
    def __call__(self,x):
        self.calls+=1
        return torch.zeros((len(x),121*121)),torch.full((len(x),),self.value)


class MCTSDiagnosticTests(unittest.TestCase):
    def tree(self,edges,winners,simulations,value=0.,ply=0,repetitions=None,time_limit=None):
        """状态编号写入单格伪棋盘；路径唯一，真实搜索和Node回传代码不变。"""
        nodes=[];original=az.Node
        class CaptureNode(original):
            def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);nodes.append(self)
        def terminal(board,player,n,counts):
            state=board[0]
            if state in winners:return winners[state],'finite_tree_terminal'
            if counts.get((tuple(board),player),0)>=3:return 0,'repetition'
            if n>=240:return 0,'max_ply'
            return None,''
        model=FixedModel(value)
        with ExitStack() as stack:
            stack.enter_context(patch.object(az,'Node',CaptureNode))
            stack.enter_context(patch.object(az,'encode',lambda *a,**k:np.zeros((1,1,1),dtype=np.float32)))
            stack.enter_context(patch.object(r,'terminal',side_effect=terminal))
            stack.enter_context(patch.object(r,'get_all_moves',side_effect=lambda b,p:[(b[0],[b[0],dst]) for dst in edges.get(b[0],[])]))
            stack.enter_context(patch.object(r,'apply_move',side_effect=lambda b,path:[path[-1]]))
            path,info=az.search(model,[0],1,ply=ply,repetitions=repetitions,simulations=simulations,time_limit=time_limit)
        return path,info,nodes,model

    def test_two_and_three_edge_terminal_backups_alternate_sign(self):
        for depth in (2,3):
            for winner in (1,2,0):
                with self.subTest(depth=depth,winner=winner):
                    edges={i:[i+1] for i in range(depth)}
                    _,info,nodes,model=self.tree(edges,{depth:winner},depth+5)
                    # 首depth-1次模拟仅扩展零值非终局，其后每次到达真终局。
                    terminal_visits=6;root_value=0 if winner==0 else 1 if winner==1 else -1
                    self.assertEqual(nodes[0].total,terminal_visits*root_value)
                    for level,node in enumerate(nodes):
                        self.assertEqual(node.total,terminal_visits*root_value*((-1)**level))
                    self.assertEqual(info['root_visits'],depth+5)
                    self.assertEqual(model.calls,depth)

    def test_opponent_choice_uses_opponent_value_and_exact_visit_accounting(self):
        _,info,nodes,_=self.tree({0:[1],1:[2,3]},{2:1,3:2},64)
        root=nodes[0];opponent=root.children[0][1];children={path[-1]:node for path,node in opponent.children}
        self.assertEqual(children[2].q,1.)  # 终局轮到P1，P1胜
        self.assertEqual(children[3].q,-1.) # 终局轮到P1，P2胜
        self.assertGreater(children[3].visits,children[2].visits)
        self.assertEqual(opponent.total,children[3].visits-children[2].visits)
        self.assertEqual(root.total,-opponent.total)
        self.assertEqual(opponent.visits,1+sum(n.visits for n in children.values()))
        self.assertEqual(info['child_visits'],64)

    def test_immediate_win_loss_draw_values_override_adversarial_network(self):
        path,info,nodes,model=self.tree({0:[1,2,3]},{1:1,2:2,3:0},96,value=.99)
        children={p[-1]:n for p,n in nodes[0].children}
        self.assertEqual({k:n.q for k,n in children.items()},{1:-1.,2:1.,3:0.})
        self.assertEqual(path,[0,1]);self.assertEqual(model.calls,1)
        self.assertEqual(nodes[0].total,children[1].visits-children[2].visits)
        self.assertEqual(info['root_visits'],sum(n.visits for n in children.values()))

    def test_repetition_and_limit_leaf_draws_are_not_network_estimates(self):
        counts=Counter({((0,),1):1,((1,),2):2})
        for options in (dict(repetitions=counts),dict(ply=239)):
            with self.subTest(options=options):
                _,info,nodes,model=self.tree({0:[1],1:[2]}, {},8,value=.99,**options)
                self.assertEqual(info['root_q'],0.)
                self.assertEqual(nodes[0].children[0][1].q,0.)
                self.assertEqual(model.calls,1)

    def test_equal_visits_use_stable_legal_move_order(self):
        for _ in range(2):
            path,info,nodes,_=self.tree({0:[2,1]},{1:0,2:0},2)
            self.assertEqual(path,[0,2]);self.assertEqual(info['policy'],[.5,.5])
            self.assertEqual(info['root_visits'],2);self.assertEqual(info['child_visits'],2)

    def test_time_limit_validation_and_soft_budget_runs_at_least_one_simulation(self):
        for limit in (0,-1,float('nan'),float('inf')):
            with self.subTest(limit=limit),self.assertRaises(ValueError):
                az.search(FixedModel(),r.init_board(),1,time_limit=limit)
        # 根展开也耗时：即使第一次边界检查已超预算，仍保证完成一次模拟。
        with patch.object(az.time,'monotonic',side_effect=[0.,2.,3.]):
            _,info,nodes,_=self.tree({0:[1]},{1:0},100,time_limit=.01)
        self.assertEqual(info['root_visits'],1);self.assertEqual(info['child_visits'],1)
        self.assertEqual(nodes[0].visits,1);self.assertGreater(info['elapsed_seconds'],.01)

    def test_absent_time_limit_preserves_fixed_simulation_count(self):
        _,info,nodes,_=self.tree({0:[1]},{1:0},17,time_limit=None)
        self.assertEqual(info['root_visits'],17);self.assertEqual(info['child_visits'],17)
        self.assertEqual(nodes[0].visits,17)

    def strict_last_move(self,player):
        # 构造双方各十子：P1将完成且P2也可一步完成时，strict优先等轮判和。
        for target in sorted(r.GOAL[player]):
            for start in r.NEIGHBORS[target]:
                if start in r.GOAL[1] or start in r.GOAL[2]:continue
                b=[0]*121
                for i in r.GOAL[player]-{target}:b[i]=player
                b[start]=player
                if player==2:
                    for i in range(40,50):b[i]=1
                else:
                    found=False
                    for other in sorted(r.GOAL[2]):
                        for outside in r.NEIGHBORS[other]:
                            if outside in r.GOAL[1] or outside in r.GOAL[2] or b[outside]:continue
                            for i in r.GOAL[2]-{other}:b[i]=2
                            b[outside]=2;found=True;break
                        if found:break
                    if not found:continue
                path=[start,target]
                if r.valid_path(b,player,path) and r.terminal(b,player)[0] is None:return b,path
        self.fail('未找到strict真实末步局面')

    def test_strict_goal_and_equal_turn_draw_precede_history_and_limit(self):
        for player,expected in ((1,(0,'equal_turn_goal_draw')),(2,(2,'all_pieces_in_goal'))):
            with self.subTest(player=player):
                board,path=self.strict_last_move(player);r.validate_board(board)
                child=r.apply_move(board,path);counts=Counter({(tuple(board),player):1,(tuple(child),3-player):2})
                # 下一状态同时触发240手与第三次重复，真实goal结果仍先处理。
                after=counts.copy();after[(tuple(child),3-player)]+=1
                self.assertEqual(r.terminal(child,3-player,240,after),expected)
                model=FixedModel(.99);original=r.get_all_moves
                with patch.object(r,'get_all_moves',side_effect=lambda b,p:[(path[0],path)] if b==board and p==player else original(b,p)):
                    chosen,info=az.search(model,board,player,ply=239,repetitions=counts,simulations=4)
                self.assertEqual(chosen,path);self.assertEqual(model.calls,1)
                self.assertEqual(info['root_q'],0. if player==1 else 1.)

if __name__=='__main__':unittest.main()
