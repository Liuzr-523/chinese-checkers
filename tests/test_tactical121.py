import unittest
from collections import Counter
import torch
import rules121 as r
from az121 import safe_root_actions,Net121,search

class Tactical121Tests(unittest.TestCase):
    def test_avoids_immediate_anti_block_reply(self):
        b=[0]*121
        for i in range(9):b[i]=1
        b[90]=1
        attacker=next(i for i in r.NEIGHBORS[9] if i>=10);b[attacker]=2
        for i in range(40,49):b[i]=2
        r.validate_board(b);self.assertIsNone(r.terminal(b,1)[0])
        allowed,info=safe_root_actions(b,1)
        self.assertGreater(info['unsafe'],0);self.assertGreater(info['allowed'],0)
        self.assertTrue(any(s<10 and e>=10 for s,e in allowed))
        for s,e in allowed:
            child=r.apply_move(b,[s,e])
            self.assertFalse(any(r.terminal(r.apply_move(child,path),1)[0]==2 for _,path in r.get_all_moves(child,2)))
        torch.set_num_threads(1);path,stats=search(Net121(),b,1,simulations=8,root_safety=True)
        self.assertIn((path[0],path[-1]),allowed);self.assertEqual(stats['root_visits'],8)

    def test_all_losing_moves_remain_available(self):
        b=[0]*121
        for i in range(9):b[i]=2
        attacker=next(i for i in r.NEIGHBORS[9] if i>=10);b[attacker]=2
        for i in range(40,50):b[i]=1
        allowed,info=safe_root_actions(b,1)
        self.assertEqual(allowed,{(s,p[-1]) for s,p in r.get_all_moves(b,1)})
        self.assertEqual(info['legal'],info['unsafe'])

    def test_max_ply_draw_not_mistaken_for_losing_reply(self):
        b=r.init_board();allowed,info=safe_root_actions(b,1,ply=239)
        self.assertEqual(info['unsafe'],0)
        self.assertEqual(allowed,{(s,p[-1]) for s,p in r.get_all_moves(b,1)})

if __name__=='__main__':unittest.main()
