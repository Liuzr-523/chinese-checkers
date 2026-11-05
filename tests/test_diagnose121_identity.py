import unittest
import numpy as np
import torch
import rules121 as r
from az121 import encode,action_id
from diagnostics.diagnose121_identity import paired_input,divergence,terminal_asymmetry_witness

class IdentityTests(unittest.TestCase):
    def test_actual_rotation_only_changes_identity(self):
        for player in (1,2):
            board=r.init_board();sample=dict(x=encode(board,player,12,canonical=True),actions=[action_id(s,p[-1],player,True) for s,p in r.get_all_moves(board,player)])
            twin,p=paired_input(sample);self.assertEqual(p,player);np.testing.assert_array_equal(twin[[0,1,2,4,5]],sample['x'][[0,1,2,4,5]]);np.testing.assert_array_equal(twin[3],sample['x'][2]-sample['x'][3])
    def test_js_zero_and_positive(self):
        a=torch.tensor([1.,2.,3.]);self.assertAlmostEqual(divergence(a,a,[0,2]),0.);self.assertGreater(divergence(a,-a,[0,2]),0.)
    def test_geometric_pair_is_not_full_terminal_symmetry(self):
        witness=terminal_asymmetry_witness();self.assertEqual(witness['original_terminal'],(0,'equal_turn_goal_draw'));self.assertEqual(witness['rotated_terminal'],(2,'all_pieces_in_goal'))
if __name__=='__main__':unittest.main()
