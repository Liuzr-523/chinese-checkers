"""WDL统计只观测原PUCT；候选概率统一为根走棋方视角。"""
import unittest
from contextlib import ExitStack
from unittest.mock import patch
from collections import Counter
import numpy as np
import torch
import az121 as az
import rules121 as r
from wdl121 import WDLNet121

class TreeWDL:
    canonical=False
    probabilities={0:[.3,.4,.3],1:[.6,.3,.1],2:[.2,.2,.6],3:[.1,.5,.4]}
    def __init__(self):self.forward_calls=0;self.wdl_calls=0
    def eval(self):return self
    def forward_wdl(self,x):
        self.wdl_calls+=1
        state=int(x[0,0,0,0]);p=torch.tensor([self.probabilities.get(state,[.2,.6,.2])])
        return torch.zeros((1,14641)),p.log()
    def __call__(self,x):
        self.forward_calls+=1;logits,wdl=self.forward_wdl(x);p=wdl.softmax(-1)
        return logits,p[:,0]-p[:,2]

class WDLSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def tree(self,player=1,simulations=3,terminal=None,ply=0,repetitions=None,stats=True,model=None,**options):
        model=model or TreeWDL();terminal=terminal or {}
        def outcome(board,p,n,counts):
            state=board[0]
            if state in terminal:return terminal[state]
            if counts.get((tuple(board),p),0)>=3:return 0,'repetition'
            if n>=240:return 0,'max_ply'
            return None,''
        with ExitStack() as stack:
            stack.enter_context(patch.object(az,'encode',lambda b,*a,**k:np.array([[[b[0]]]],dtype=np.float32)))
            stack.enter_context(patch.object(r,'terminal',side_effect=outcome))
            stack.enter_context(patch.object(r,'get_all_moves',side_effect=lambda b,p:[(b[0],[b[0],b[0]+1])]))
            stack.enter_context(patch.object(r,'apply_move',side_effect=lambda b,path:[path[-1]]))
            result=az.search(model,[0],player,simulations=simulations,ply=ply,repetitions=repetitions,wdl_stats=stats,**options)
        return *result,model

    def test_multilevel_wdl_swaps_win_loss_at_every_ply_for_both_players(self):
        expected=np.mean([[.1,.3,.6],[.2,.2,.6],[.4,.5,.1]],axis=0)
        for player in (1,2):
            with self.subTest(player=player):
                _,info,model=self.tree(player=player)
                np.testing.assert_allclose(info['root_wdl'],expected,atol=1e-7)
                np.testing.assert_allclose(info['child_wdl'][0],expected,atol=1e-7)
                self.assertEqual(info['child_wdl_visits'],[3]);self.assertEqual(info['visits'],[3.])
                self.assertAlmostEqual(info['root_q'],expected[0]-expected[2],places=6)
                self.assertEqual(model.forward_calls,0);self.assertEqual(model.wdl_calls,4)

    def test_real_terminal_onehot_overrides_network_and_keeps_root_perspective(self):
        for player in (1,2):
            for winner,reason in ((player,'all_pieces_in_goal'),(3-player,'all_pieces_in_goal'),(0,'repetition')):
                with self.subTest(player=player,winner=winner):
                    _,info,model=self.tree(player=player,simulations=5,terminal={1:(winner,reason)})
                    expected=[1.,0.,0.] if winner==player else [0.,0.,1.] if winner else [0.,1.,0.]
                    self.assertEqual(info['root_wdl'],expected);self.assertEqual(info['child_wdl'],[expected])
                    self.assertEqual(info['child_wdl_visits'],[5]);self.assertEqual(model.wdl_calls,1)

    def test_max_ply_is_missing_probability_not_fabricated_draw(self):
        _,info,_=self.tree(simulations=4,ply=239)
        self.assertIsNone(info['root_wdl']);self.assertEqual(info['child_wdl'],[None])
        self.assertEqual(info['child_wdl_visits'],[0]);self.assertEqual(info['visits'],[4.])
        self.assertEqual(info['root_q'],0.)
        # 第一次模拟有NN概率，其余三次均限步；均值只按一次有效WDL计数。
        _,mixed,_=self.tree(simulations=4,ply=238)
        np.testing.assert_allclose(mixed['root_wdl'],[.1,.3,.6],atol=1e-7)
        self.assertEqual(mixed['child_wdl_visits'],[1]);self.assertEqual(mixed['visits'],[4.])

    def test_repetition_is_real_draw_probability(self):
        counts=Counter({((0,),1):1,((1,),2):2})
        _,info,_=self.tree(repetitions=counts,simulations=2)
        self.assertEqual(info['child_wdl'],[[0.,1.,0.]])
        self.assertEqual(info['child_wdl_visits'],[2])

    def test_statistics_preserve_real_board_paths_visits_q_and_default_output(self):
        model=WDLNet121(width=8);board=r.init_board()
        a,default=az.search(model,board,1,simulations=21)
        b,disabled=az.search(model,board,1,simulations=21,wdl_stats=False)
        c,stats=az.search(model,board,1,simulations=21,wdl_stats=True)
        self.assertEqual(a,b);self.assertEqual(a,c);self.assertEqual(default,disabled)
        self.assertNotIn('child_wdl',default)
        for field in default:self.assertEqual(default[field],stats[field],field)
        self.assertTrue(all(n<=v for n,v in zip(stats['child_wdl_visits'],stats['visits'])))
        for visits,prob in zip(stats['child_wdl_visits'],stats['child_wdl']):
            if visits:self.assertAlmostEqual(sum(prob),1.,places=6)
            else:self.assertIsNone(prob)

    def test_statistics_do_not_change_nonzero_value_tree_search(self):
        a,default,_=self.tree(stats=False,simulations=3)
        b,stats,_=self.tree(stats=True,simulations=3)
        self.assertEqual(a,b)
        for field in default:self.assertEqual(default[field],stats[field],field)

    def test_stats_reject_non_wdl_or_modified_evaluators(self):
        class PlainModel:
            def eval(self):return self
        cases=[dict(model=PlainModel()),dict(leaf_value_fn=lambda b,p:0.),dict(heuristic_mix=.1),dict(value_scale=0.),dict(value_scale=.9)]
        for options in cases:
            with self.subTest(options=options),self.assertRaises(ValueError):self.tree(**options)

if __name__=='__main__':unittest.main()
