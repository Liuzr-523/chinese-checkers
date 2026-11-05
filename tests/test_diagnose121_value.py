import unittest
import numpy as np
from diagnostics.diagnose121_value import summarize,metadata
from az121 import encode
from cycle121_targeted import value_label
from canonical121_trial import canonical_sample
import rules121 as r
class ValueDiagnosticTests(unittest.TestCase):
 def test_zero_prediction_baseline_and_half_ties(self):
  s=summarize([0,0],[-1,1],0);self.assertEqual(s['mse'],1);self.assertEqual(s['sign_accuracy_half_ties'],.5);self.assertIsNone(s['correlation']);self.assertEqual(s['abs_prediction_below_01_fraction'],1)
 def test_constant_target_null_and_draw_excluded_from_sign(self):
  s=summarize([.2,.4],[1,1],1);self.assertIsNone(s['correlation']);self.assertEqual(s['train_mean_baseline_mse'],0)
  s=summarize([0,1,-1],[0,1,-1],0);self.assertEqual(s['sign_samples'],2);self.assertEqual(s['sign_accuracy_half_ties'],1);self.assertAlmostEqual(s['correlation'],1)
 def test_identity_and_label_survive_canonical_rotation(self):
  g=r.Game();sample={'x':encode(g.board,2,41,canonical=True)};self.assertEqual(metadata(sample),dict(player=2,ply=41,stage='40—79手'));self.assertEqual(value_label(1,'goal',2),(-1.,1.));self.assertEqual(value_label(0,'max_ply',2),(0.,0.))
 def test_old_canonical_sample_preserves_side_to_move_outcome(self):
  g=r.Game()
  for player in (1,2):
   absolute=dict(x=encode(g.board,player,20),actions=[0*121+1],target=0,z=1. if player==1 else -1.,winner=1)
   canonical=canonical_sample(absolute);self.assertEqual(canonical['z'],absolute['z']);self.assertEqual(metadata(canonical)['player'],player)
 def test_nonfinite_rejected(self):
  with self.assertRaises(ValueError):summarize([float('nan')],[1],0)
