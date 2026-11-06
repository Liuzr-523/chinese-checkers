"""叶子估值替换的独立回传测试；不验证手工估值的棋力。"""
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import torch
import az121 as az
import rules121 as r


class ConstantModel:
    canonical=False
    def __init__(self,value=.99):self.value=value;self.calls=0
    def eval(self):return self
    def __call__(self,x):
        self.calls+=1
        return torch.zeros((len(x),121*121)),torch.full((len(x),),self.value)


class LeafAblationTests(unittest.TestCase):
    def chain(self,callback,simulations=1,winners=None,value_scale=1.,root_player=1):
        """状态0→1→2→3→4；Node与PUCT/回传保持生产实现。"""
        winners=winners or {};nodes=[];original=az.Node;model=ConstantModel()
        class CaptureNode(original):
            def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);nodes.append(self)
        def terminal(board,player,ply,counts):
            return (winners[board[0]],'test_terminal') if board[0] in winners else (None,'')
        with ExitStack() as stack:
            stack.enter_context(patch.object(az,'Node',CaptureNode))
            stack.enter_context(patch.object(az,'encode',lambda *args,**kwargs:np.zeros((1,1,1),dtype=np.float32)))
            stack.enter_context(patch.object(r,'terminal',side_effect=terminal))
            stack.enter_context(patch.object(r,'get_all_moves',side_effect=lambda b,p:[(b[0],[b[0],b[0]+1])]))
            stack.enter_context(patch.object(r,'apply_move',side_effect=lambda b,path:[path[-1]]))
            path,info=az.search(model,[0],root_player,simulations=simulations,value_scale=value_scale,leaf_value_fn=callback)
        return path,info,nodes,model

    def test_callback_receives_leaf_side_and_each_backup_flips_sign(self):
        for player in (1,2):
            with self.subTest(root_player=player):
                calls=[];values={0:-.9,1:.6,2:-.2,3:.4}
                def evaluate(board,p):
                    calls.append((board[0],p));return values[board[0]]
                _,info,nodes,_=self.chain(evaluate,simulations=3,root_player=player)
                self.assertEqual(calls,[(0,player),(1,3-player),(2,player),(3,3-player)])
                # 模拟1根回传-.6；模拟2根回传-.2；模拟3根回传-.4。
                self.assertAlmostEqual(nodes[0].total,-1.2)
                self.assertAlmostEqual(info['root_q'],-.4)
                self.assertAlmostEqual(nodes[0].children[0][1].total,1.2)
                self.assertAlmostEqual(nodes[0].children[0][1].children[0][1].total,-.6)

    def test_callback_replaces_network_before_value_scale(self):
        calls=[]
        def evaluate(board,p):calls.append((board[0],p));return .6
        _,info,nodes,model=self.chain(evaluate,value_scale=.5)
        self.assertAlmostEqual(info['root_q'],-.3)
        self.assertAlmostEqual(nodes[0].children[0][1].q,.3)
        self.assertEqual(model.calls,2) # policy依然取模型；其.99 value已被替换。
        self.assertEqual(calls,[(0,1),(1,2)])

    def test_terminal_leaf_bypasses_callback_and_model(self):
        for winner,expected in ((1,1.),(2,-1.),(0,0.)):
            with self.subTest(winner=winner):
                calls=[]
                def evaluate(board,p):
                    calls.append((board[0],p))
                    if board[0]!=0:raise AssertionError('终局不得调用估值callback')
                    return -.99
                _,info,_,model=self.chain(evaluate,simulations=5,winners={1:winner},value_scale=0.)
                self.assertEqual(calls,[(0,1)]);self.assertEqual(model.calls,1)
                self.assertEqual(info['root_q'],expected) # 真终局不乘value_scale。

    def test_terminal_root_never_calls_callback_or_network(self):
        def forbidden(*args):raise AssertionError('终局根不得调用callback')
        path,info,_,model=self.chain(forbidden,winners={0:1})
        self.assertIsNone(path);self.assertEqual(info['root_visits'],0);self.assertEqual(model.calls,0)

    def test_invalid_leaf_values_rejected_even_when_value_scale_zero(self):
        for value in (float('nan'),float('inf'),-float('inf'),1.0001,-1.0001):
            with self.subTest(value=value),self.assertRaises(ValueError):
                self.chain(lambda b,p:value,value_scale=0.)
        for value in (-1.,1.):
            with self.subTest(valid_endpoint=value):
                _,info,_,_=self.chain(lambda b,p:value)
                self.assertEqual(info['root_q'],-value)

    def test_none_preserves_default_paths_visits_and_value(self):
        board=r.init_board()
        default_path,default=az.search(ConstantModel(.25),board,1,simulations=19)
        explicit_path,explicit=az.search(ConstantModel(.25),board,1,simulations=19,leaf_value_fn=None)
        constant_path,constant=az.search(ConstantModel(.99),board,1,simulations=19,leaf_value_fn=lambda b,p:.25)
        for path,info in ((explicit_path,explicit),(constant_path,constant)):
            self.assertEqual(path,default_path)
            for field in ('paths','policy','priors','root_visits','child_visits','root_q','visits','child_q'):
                self.assertEqual(info[field],default[field],field)

if __name__=='__main__':unittest.main()
