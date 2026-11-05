import copy,unittest
import numpy as np
import torch
from az121 import Net121
from train121_dual_value import group_index,draw_batch,freeze_arm,nonvalue_hash,retention_kl

class DualTrainerTests(unittest.TestCase):
    def test_sampling_masks_and_sequence(self):
        samples=[dict(split='train',group=g,sample_id=f'{g}:{i}',terminal_distance=i+1,policy_weight=1.,value_weight=1.) for g in (0,1) for i in range(20)]
        vg=group_index(samples,'value_weight');lg=group_index([s for s in samples if s['terminal_distance']<=16],'value_weight');pg=group_index(samples,'policy_weight');a,v,p=draw_batch(vg,lg,pg,np.random.default_rng(42));b,_,_=draw_batch(vg,lg,pg,np.random.default_rng(42));self.assertEqual(a,b);self.assertTrue(all(s['terminal_distance']<=16 for s in v[:8]));self.assertTrue(all(s['policy_weight']==0 for s in a[:16]));self.assertTrue(all(s['value_weight']==0 for s in a[16:]));self.assertTrue(all(s['value_weight']==s['policy_weight']==1 for s in samples))
        with self.assertRaises(ValueError):group_index([{**samples[0],'split':'development'}],'value_weight')
        with self.assertRaises(ValueError):group_index([{**samples[0],'policy_weight':0}],'policy_weight')
    def test_head_only_updates_value_preserves_logits(self):
        torch.set_num_threads(1);m=Net121(width=4,canonical=True);freeze_arm(m,'head_only');x=torch.randn(2,6,17,17);before=m(x)[0].detach().clone();h=nonvalue_hash(m);old=[p.detach().clone() for p in m.value.parameters()];opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01);loss=(m(x)[1]-1).square().mean();loss.backward();opt.step();self.assertEqual(nonvalue_hash(m),h);self.assertTrue(torch.equal(m(x)[0],before));self.assertTrue(any(not torch.equal(a,b) for a,b in zip(old,m.value.parameters())))
if __name__=='__main__':unittest.main()
