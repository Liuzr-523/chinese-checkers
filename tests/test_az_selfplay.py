import unittest
from unittest.mock import patch
import numpy as np
import alphazero_nn as az
import self_play_resnet_mcts as rz
from diagnostics.audit_alphazero import audit_counts
from nn_torch import NNEasyEngineTorch, NetTorch

class SelfPlayTests(unittest.TestCase):
    def test_counts_conserved_both_backends(self):
        for module,resnet in [(az,False),(rz,True)]:
            counts=audit_counts(module,resnet)
            self.assertEqual(counts['root_N'],32)
            self.assertEqual(counts['root_children_sum'],32)

    def test_unknown_outcome_masks_value(self):
        board=az.cc.init_board(); moves=az.all_moves_indexed(board,1)
        rec=dict(winner=None,steps=[dict(player=1,move_path=moves[0][2],moves=moves,pi=[1/len(moves)]*len(moves))])
        sample=az.expand_game_records([rec])[0]
        self.assertEqual((sample['z'],sample['vw']),(0.,0.))

    def test_torch_engine_really_calls_search(self):
        model=NetTorch.from_numpy_params(az.NetParams(seed=1,filters=4))
        eng=NNEasyEngineTorch(model,use_mcts=True,n_simulations=17)
        with patch.object(rz,'mcts_think_resnet',return_value=(([],[]),(0,[0,1]))) as search:
            path,info=eng.think(az.cc.init_board(),1)
        self.assertEqual(path,[0,1]); self.assertTrue(info['mcts'])
        self.assertEqual(search.call_args.args[3],17)
        self.assertFalse(search.call_args.kwargs['add_root_noise'])

    def test_evaluation_cutoff_has_no_invented_winner(self):
        from evaluate_nn_vs_opponent import play_one_game
        class FirstLegal:
            def think(self, board, player):
                return az.cc.get_all_moves(board, player)[0][1], {}
        winner, plies = play_one_game(FirstLegal(), FirstLegal(), True, max_moves=1)
        self.assertIsNone(winner)
        self.assertEqual(plies, 1)

    def test_numpy_truncation_masks_value(self):
        moves = az.all_moves_indexed(az.cc.init_board(), 1)
        result = (0., (moves, [1/len(moves)]*len(moves)), (moves[0][0], moves[0][2]))
        with patch.object(az, 'mcts_think', return_value=result):
            rec = az.self_play_one_game_raw(None, 1, max_moves=1)
        self.assertIsNone(rec['winner'])
        self.assertTrue(rec['aborted'])

    def test_winning_move_and_opponent_perspective(self):
        for module, resnet in [(az, False), (rz, True)]:
            created = []
            original = module.Node
            def factory(*args):
                node = original(*args); created.append(node); return node
            def moves(board, player):
                return [(board[0], d, [board[0], d]) for d in {0:[1,2],2:[3,4]}.get(board[0], [])]
            from contextlib import ExitStack
            with ExitStack() as stack:
                for obj, name, value in [
                    (module, 'Node', factory), (module, 'all_moves_indexed', moves),
                    (module.cc, 'apply_move', lambda b,p:[p[-1]]),
                    (module.cc, 'is_game_over', lambda b:{1:1,3:2,4:1}.get(b[0])),
                    (module, 'resnet_forward' if resnet else 'forward', lambda *a,**k:(np.zeros(162),0.))]:
                    stack.enter_context(patch.object(obj, name, value))
                if not resnet:
                    stack.enter_context(patch.object(module, 'encode_board', lambda *a:None))
                fn = module.mcts_think_resnet if resnet else module.mcts_think
                result = fn(None, [0], 1, 128, c_puct=10.)
                reply = fn(None, [2], 2, 32)
            self.assertEqual(result[-1][1][-1], 1)
            self.assertEqual(reply[-1][1][-1], 3)
            self.assertEqual(created[0].children[0][3].Q, 1.)

    def test_history_avoids_repeat_but_keeps_legal_fallback(self):
        from contextlib import ExitStack
        for module, resnet in [(az,False),(rz,True)]:
            def moves(b,p):
                return [(0,d,[0,d]) for d in (1,2)] if b == [0] else []
            with ExitStack() as stack:
                for obj,name,value in [
                    (module, 'all_moves_indexed', moves),
                    (module.cc, 'apply_move', lambda b,p:[p[-1]]),
                    (module.cc, 'is_game_over', lambda b:None),
                    (module, 'resnet_forward' if resnet else 'forward', lambda *a,**k:(np.zeros(162),0.))]:
                    stack.enter_context(patch.object(obj,name,value))
                if not resnet:
                    stack.enter_context(patch.object(module,'encode_board',lambda *a:None))
                fn = module.mcts_think_resnet if resnet else module.mcts_think
                fresh = fn(None,[0],1,8,history={((1,),2)})
                fallback = fn(None,[0],1,8,history={((1,),2),((2,),2)})
                other_turn = fn(None,[0],1,8,history={((1,),1)})
            self.assertEqual(fresh[-1][1], [0,2])
            self.assertIn(fallback[-1][1], [[0,1],[0,2]])
            self.assertEqual(other_turn[-1][1], [0,1])

    def test_engine_history_is_game_scoped(self):
        model=NetTorch.from_numpy_params(az.NetParams(seed=1,filters=4))
        eng=NNEasyEngineTorch(model,use_mcts=True,n_simulations=17)
        board=az.cc.init_board()
        path=az.cc.get_all_moves(board,1)[0][1]
        with patch.object(rz,'mcts_think_resnet',return_value=(([],[]),(path[0],path))) as search:
            eng.think(board,1)
        self.assertIn((tuple(board),1),eng.history)
        self.assertIn((tuple(az.cc.apply_move(board,path)),2),eng.history)
        self.assertIs(search.call_args.kwargs['history'],eng.history)
        eng.reset_history()
        self.assertEqual(eng.history,set())

    def test_no_moves_contract(self):
        with patch.object(az,'all_moves_indexed',return_value=[]):
            self.assertEqual(az.mcts_think(None,[],1),(None,([],[]),(None,None)))
        with patch.object(rz,'all_moves_indexed',return_value=[]):
            self.assertEqual(rz.mcts_think_resnet(None,[],1),(([],[]),(None,None)))

if __name__=='__main__':unittest.main()
