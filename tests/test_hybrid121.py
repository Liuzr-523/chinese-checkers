import unittest
import torch
import rules121 as r
from az121 import Net121,search,Neural121

class Hybrid121Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_default_unchanged_and_mixture_legal(self):
        torch.manual_seed(301035);model=Net121();board=r.init_board()
        a,ai=search(model,board,1,simulations=16)
        b,bi=search(model,board,1,simulations=16,heuristic_mix=0.)
        self.assertEqual(a,b);self.assertEqual(ai['policy'],bi['policy'])
        for mix in (.5,1.):
            path,info=Neural121(model,16,heuristic_mix=mix).think(board,1)
            self.assertTrue(r.valid_path(board,1,path));self.assertEqual(info['root_visits'],16)
        for mix in (-1.,2.,float('nan')):
            with self.assertRaises(ValueError):search(model,board,1,heuristic_mix=mix)
    def test_real_terminal_win_is_not_replaced_by_progress_score(self):
        board=[0]*121
        for i in range(9):board[i]=2
        board[next(i for i in r.NEIGHBORS[9] if i>=10)]=2
        for i in range(40,50):board[i]=1
        for mix in (.5,1.):
            path,info=search(Net121(),board,2,simulations=48,heuristic_mix=mix)
            self.assertEqual(path[-1],9)

if __name__=='__main__':unittest.main()
