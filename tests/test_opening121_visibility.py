import unittest
import numpy as np
import torch
from az121 import Net121
from diagnostics.audit121_opening_visibility import support,find_pairs
class VisibilityTests(unittest.TestCase):
 def samples(self):
  x=np.zeros((6,17,17),dtype=np.float32);y=x.copy();y[0,16,4]=1
  return [dict(x=x,positive=[1],actions=[1,2],decision_id='a'),dict(x=y,positive=[2],actions=[1,2],decision_id='b')]
 def test_alias_requires_same_relevant_inputs_and_tie_order(self):
  samples=self.samples();self.assertEqual(len(find_pairs(samples)),1);samples[1]['actions']=[2,1];self.assertEqual(find_pairs(samples),[]);samples=self.samples();mask=support(1,2);i,j=np.argwhere(mask)[0];samples[1]['x'][0,i,j]=1;self.assertEqual(find_pairs(samples),[])
 def test_current_policy_local_logits_are_invariant_to_remote_change(self):
  torch.set_num_threads(1);m=Net121(width=4,canonical=True);samples=self.samples()
  with torch.no_grad():l,_=m(torch.from_numpy(np.stack([s['x'] for s in samples])))
  torch.testing.assert_close(l[0,[1,2]],l[1,[1,2]],rtol=0,atol=0)
if __name__=='__main__':unittest.main()
