import copy,unittest
import torch
from az121 import Net121
from opening121 import OpeningBook
from train_opening121 import book_samples
from opening121_ratios import mixed_loss
from ablate_opening121_hard32 import terms,combine,gradient_audit
class AblationTests(unittest.TestCase):
 def test_full_matches_existing_loss_and_four_factor_cells(self):
  torch.set_num_threads(1);m=Net121(width=4,canonical=True);ref=copy.deepcopy(m);s=book_samples(OpeningBook())[0]['train'][0];root={**s,'value_weight':0.};ordinary={**s,'value_weight':1.,'z':1.};parts=terms(m,ref,[root]*4,[ordinary]*12)
  torch.testing.assert_close(combine(parts,'full'),mixed_loss(m,ref,[root]*4,[ordinary]*12))
  torch.testing.assert_close(combine(parts,'with_replay')+combine(parts,'with_aux')-combine(parts,'root_only'),combine(parts,'full'))
  params=[p for n,p in m.named_parameters() if not n.startswith('value.')];audit=gradient_audit(parts,params);self.assertGreater(audit['root_norm'],0);self.assertGreater(audit['aux_norm'],0)
  self.assertTrue(all(p.grad is None for p in params))
 def test_unknown_arm_rejected(self):
  with self.assertRaises(ValueError):combine([1,2,3],'unknown')
