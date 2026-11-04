import unittest
from collections import Counter
import clysto121_native as native
import rules121 as r
from engine_clysto121 import ClystoDerived121
from verify_clysto121 import reference
from imitate_clysto121 import expert_gate

class ClystoDerivativeTests(unittest.TestCase):
    def test_strict_goal_lock_and_real_history(self):
        b=r.init_board();b[0],b[111]=b[111],b[0]
        for p in (1,2):
            self.assertEqual(set(native.moves(b,p)),{(s,path[-1]) for s,path in r.get_all_moves(b,p)})
        b=r.init_board();counts=Counter({(tuple(b),1):3})
        path,info=ClystoDerived121().think(b,1,0,counts)
        self.assertIsNone(path);self.assertEqual(info['score'],0);self.assertFalse(info['original_clysto'])
        with self.assertRaises(ValueError):native.moves([0]*81,1)

    def test_reference_eval_on_all_wings_and_origin(self):
        b=r.init_board()
        for qy in ((-8,4),(8,-4),(-4,-4),(4,4)):
            dst=r.IDS[qy];b[0],b[dst]=b[dst],b[0]
            for p in (1,2):self.assertEqual(native.evaluate(b,p),reference(b,p))
            b[0],b[dst]=b[dst],b[0]
        path,info=ClystoDerived121(seconds=.01).think(b,1)
        self.assertTrue(r.valid_path(b,1,path))

    def test_expert_gate_requires_pair_evidence_and_complete_budget(self):
        rows=[dict(pair=i//2,side=1+i%2,winner=1+i%2,reason='all_pieces_in_goal') for i in range(40)]
        self.assertTrue(expert_gate(rows));self.assertFalse(expert_gate(rows[:38]))
        # A 50% result must never authorize expert generation/training.
        for i,row in enumerate(rows):row['winner']=1 if i%4<2 else 2
        self.assertFalse(expert_gate(rows))

if __name__=='__main__':unittest.main()
