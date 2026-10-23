import copy
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import torch
import alphazero_nn as az
import self_play_resnet_mcts as rz
from az_longrun import samples_for, validate, metrics
from nn_resnet import NetResNet


class LongRunTests(unittest.TestCase):
    def test_repeated_leaf_is_neutral_without_network_evaluation(self):
        for module, is_torch in [(az,False),(rz,True)]:
            nodes=[]; original=module.Node
            def create(*args):
                node=original(*args);nodes.append(node);return node
            def forward(*args,**kwargs):
                return np.zeros(162), .9
            with ExitStack() as stack:
                for obj,name,value in [
                    (module,'Node',create),
                    (module,'all_moves_indexed',lambda b,p:[(0,1,[0,1])] if b==[0] else []),
                    (module.cc,'apply_move',lambda b,p:[p[-1]]),
                    (module.cc,'is_game_over',lambda b:None)]:
                    stack.enter_context(patch.object(obj,name,value))
                net=stack.enter_context(patch.object(module,'resnet_forward' if is_torch else 'forward',side_effect=forward))
                if not is_torch:stack.enter_context(patch.object(module,'encode_board',lambda *a:None))
                fn=module.mcts_think_resnet if is_torch else module.mcts_think
                fn(None,[0],1,8,history={((1,),2)})
            self.assertEqual(net.call_count,1)
            self.assertEqual(nodes[1].Q,0.)
            self.assertEqual(nodes[1].N,8)

    def test_discounted_terminal_backup(self):
        for module, is_torch in [(az,False),(rz,True)]:
            nodes=[]; original=module.Node
            def create(*args):
                node=original(*args);nodes.append(node);return node
            with ExitStack() as stack:
                for obj,name,value in [
                    (module,'Node',create),
                    (module,'all_moves_indexed',lambda b,p:[(0,1,[0,1])] if b==[0] else []),
                    (module.cc,'apply_move',lambda b,p:[p[-1]]),
                    (module.cc,'is_game_over',lambda b:1 if b==[1] else None),
                    (module,'resnet_forward' if is_torch else 'forward',lambda *a,**k:(np.zeros(162),0.))]:
                    stack.enter_context(patch.object(obj,name,value))
                if not is_torch:stack.enter_context(patch.object(module,'encode_board',lambda *a:None))
                fn=module.mcts_think_resnet if is_torch else module.mcts_think
                fn(None,[0],1,4,discount=.9,history={((1,),2)})
            # Rule win takes precedence over historical repetition.
            self.assertAlmostEqual(nodes[1].Q,.9)

    def test_completed_labels_only_and_distance_target(self):
        board=az.cc.init_board();moves=az.all_moves_indexed(board,1)
        first=dict(player=1,move_path=moves[0][2],moves=moves,pi=[1/len(moves)]*len(moves))
        board=az.cc.apply_move(board,first['move_path']);moves=az.all_moves_indexed(board,2)
        second=dict(player=2,move_path=moves[0][2],moves=moves,pi=[1/len(moves)]*len(moves))
        # Synthetic winner labels exercise transformation; real records pass validate first.
        rec=dict(winner=1,steps=[first,second])
        samples=samples_for([rec,dict(winner=None,steps=[first])],.9)
        self.assertEqual(len(samples),2)
        self.assertAlmostEqual(samples[0]['z'],.81)
        self.assertAlmostEqual(samples[1]['z'],-.9)
        with self.assertRaises(AssertionError):validate(rec)

    def test_scaling_value_head_keeps_policy_identical(self):
        torch.set_num_threads(1);torch.manual_seed(1)
        model=NetResNet(filters=4,blocks=1).eval();scaled=copy.deepcopy(model)
        with torch.no_grad():
            scaled.v_fc2.weight.mul_(.1);scaled.v_fc2.bias.mul_(.1)
            x=torch.from_numpy(az.encode_board(az.cc.init_board(),1))[None]
            p1,v1=model(x);p2,v2=scaled(x)
        self.assertTrue(torch.equal(p1,p2))
        self.assertLessEqual(float(v2.abs()),float(v1.abs())+1e-7)

if __name__=='__main__':unittest.main()
