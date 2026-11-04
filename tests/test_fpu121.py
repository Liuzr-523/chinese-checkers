import unittest
import torch
import rules121 as r
from az121 import search

class FixedSideBelief:
    def eval(self):return self
    def __call__(self,x):
        logits=torch.zeros((len(x),121*121))
        value=torch.where(x[:,3].sum((1,2))>0,.75,-.75)
        return logits,value

class FirstPlayTests(unittest.TestCase):
    def test_parent_baseline_removes_zero_reference_exploration_bias(self):
        torch.set_num_threads(1);b=r.init_board();model=FixedSideBelief()
        a,old=search(model,b,1,simulations=32,fpu_mode='zero')
        c,new=search(model,b,1,simulations=32,fpu_mode='parent')
        self.assertTrue(r.valid_path(b,1,a));self.assertTrue(r.valid_path(b,1,c))
        self.assertGreater(sum(p>0 for p in new['policy']),sum(p>0 for p in old['policy']))
        self.assertEqual(new['root_visits'],32);self.assertEqual(new['child_visits'],32)
        self.assertAlmostEqual(sum(new['policy']),1)
        default,di=search(model,b,1,simulations=32)
        self.assertEqual(default,a);self.assertEqual(di['policy'],old['policy'])
        with self.assertRaises(ValueError):search(model,b,1,fpu_mode='unknown')

if __name__=='__main__':unittest.main()
