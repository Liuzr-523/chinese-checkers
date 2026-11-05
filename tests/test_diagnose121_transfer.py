import unittest
import numpy as np
import rules121 as r
from diagnostics.diagnose121_transfer import row_metrics,geometry,compare
from opening121 import key
from opening121_split_audit import rotate_swap
class TransferTests(unittest.TestCase):
 def test_set_probability_rank_margin_and_stable_tie(self):
  x=row_metrics([3,3,1],[10,11,12],[11]);self.assertFalse(x['hit']);self.assertEqual(x['target_rank'],1);self.assertEqual(x['target_margin'],0);self.assertEqual(x['argmax_margin'],0)
  y=row_metrics([0,2,1],[10,11,12],[10,12]);self.assertEqual(y['target_rank'],2);self.assertEqual(y['target_margin'],-1);self.assertAlmostEqual(y['target_probability'],(1+np.exp(1))/(1+np.exp(1)+np.exp(2)))
 def test_geometry_turn_symmetry(self):
  state=key(r.init_board(),1);self.assertEqual(geometry(state),geometry(rotate_swap(state)));self.assertEqual(geometry(state)['exited_home'],0)
 def test_probability_up_does_not_imply_argmax_change(self):
  old=dict(decision_id='x',player=1,own_move=7,chosen=1,hit=False,target_probability=.1,target_rank=3,target_margin=-2)
  new={**old,'target_probability':.2,'target_rank':2,'target_margin':-1};c=compare([new],[old])['overall'];self.assertEqual(c['probability_up_same_choice'],1);self.assertEqual(c['probability_up_still_miss'],1);self.assertEqual(c['choice_changed'],0)
if __name__=='__main__':unittest.main()
