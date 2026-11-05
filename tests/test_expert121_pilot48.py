import unittest
from expert121_pilot48 import value_label,mark_exclusions,exited
import rules121 as r

class ExpertPilotTests(unittest.TestCase):
    def test_true_draw_and_truncation(self):
        self.assertEqual(value_label(0,'equal_turn_goal_draw',1),(0.,1.));self.assertEqual(value_label(0,'repetition',2),(0.,1.));self.assertEqual(value_label(0,'max_ply',1),(0.,0.));self.assertEqual(value_label(1,'all_pieces_in_goal',2),(-1.,1.))
        with self.assertRaises(ValueError):value_label(1,'max_ply',1)
        with self.assertRaises(ValueError):value_label(None,'',1)
    def test_conflicts_mask_both_splits_preserve_rows(self):
        rows=[dict(orbit=o,split=s,value_weight=1.,policy_weight=1.) for o,s in [('a','train'),('a','development'),('b','train'),('c','development')]];shared=mark_exclusions(rows,{'b'});self.assertEqual(shared,{'a'});self.assertEqual(len(rows),4)
        self.assertTrue(all(x['value_weight']==x['policy_weight']==0 for x in rows[:3]));self.assertEqual(rows[-1]['value_weight'],1.)
    def test_initial_progress_zero(self):self.assertEqual(exited(r.init_board()),0.)
if __name__=='__main__':unittest.main()
