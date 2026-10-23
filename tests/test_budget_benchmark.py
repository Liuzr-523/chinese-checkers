import unittest
from unittest.mock import patch
import numpy as np
import Chinese_Checker as cc
from alphazero_nn import all_moves_indexed
from az_longrun import play
from search_budget_benchmark import schedule, compatible_extension

class BudgetTests(unittest.TestCase):
    def test_balanced_schedule(self):
        tasks=schedule(8)
        self.assertEqual(len(tasks),48)
        self.assertEqual(len(set(tasks)),48)
        for h,l,seed,side in tasks:self.assertIn((h,l,seed,3-side),tasks)

    def test_extension_keeps_existing_schedule_and_settings(self):
        self.assertEqual(schedule(50)[:48],schedule(8))
        self.assertEqual(len(schedule(50)),300)
        self.assertTrue(compatible_extension(dict(openings=8,discount=.995),dict(openings=50,discount=.995)))
        self.assertFalse(compatible_extension(dict(openings=8,discount=.995),dict(openings=50,discount=1.)))
        self.assertFalse(compatible_extension(dict(openings=8),dict(openings=2)))

    def test_each_player_gets_own_budget(self):
        calls=[]
        def search(model,board,player,budget,**kwargs):
            calls.append((player,budget))
            moves=all_moves_indexed(board,player)
            return (moves,[1/len(moves)]*len(moves)),(moves[0][0],moves[0][2])
        with patch('az_longrun.mcts_think_resnet',side_effect=search):
            rec=play([None,None],1,[256,64],[.995,.995],max_moves=2)
        self.assertEqual(calls,[(1,256),(2,64)])
        self.assertEqual(rec['search_moves'],[1,1])
        self.assertIsNone(rec['winner'])
        self.assertEqual(rec['validation']['errors'],0)

if __name__=='__main__':unittest.main()
