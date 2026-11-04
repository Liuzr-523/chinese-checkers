import copy,unittest
import numpy as np
import torch
from torch.nn import functional as F
from az121 import Net121
from opening121 import OpeningBook
from train_opening121 import book_samples
from curriculum121 import batch
from opening121_ratios import mixed_loss,acceptance,stage_of
class RatioTests(unittest.TestCase):
 def test_policy_weight_matches_actual_fraction_and_root_outcome_is_ignored(self):
  torch.set_num_threads(1);m=Net121(width=4,canonical=True);ref=copy.deepcopy(m);s=book_samples(OpeningBook())[0]['train'][0];s={**s,'positive':[s['actions'][0]],'target':0};x,*_=batch([s]);logits,value=m(x);ordinary={**s,'target':1,'value_weight':1.,'z':float(value.detach()[0])};root={**s,'z':1000.,'value_weight':0.}
  ce0=F.cross_entropy(logits[:,s['actions']],torch.tensor([0]));ce1=F.cross_entropy(logits[:,s['actions']],torch.tensor([1]))
  for n in (4,8,12):
   loss=mixed_loss(m,ref,[root]*n,[ordinary]*(16-n));expected=(n/16)*ce0+(1-n/16)*ce1;torch.testing.assert_close(loss,expected,rtol=1e-5,atol=1e-6)
 def test_gate_requires_training_fit_and_old_ability(self):
  b=dict(book_validation=dict(top1_in_book=.375),retention=dict(policy_ce=2.1,value_mse=1.),normal=dict(policy_ce=2.4));r=dict(book_train=dict(top1_in_book=.91),book_validation=dict(top1_in_book=.6),standard_first6=dict(top1_in_book=1.),retention=dict(policy_ce=2.12,value_mse=1.02),normal=dict(policy_ce=2.41));self.assertTrue(acceptance(r,b)[0]);r['book_train']['top1_in_book']=.89;self.assertFalse(acceptance(r,b)[0]);r['book_train']['top1_in_book']=.91;r['retention']['policy_ce']=2.2;self.assertFalse(acceptance(r,b)[0])
 def test_stages_are_own_move_numbers(self):
  for ply,expected in [(0,1),(5,3),(6,4),(11,6),(12,7),(19,10)]:
   x=np.zeros((6,17,17),dtype=np.float32);x[4]=ply/240;self.assertEqual(stage_of(dict(x=x)),expected)
if __name__=='__main__':unittest.main()
