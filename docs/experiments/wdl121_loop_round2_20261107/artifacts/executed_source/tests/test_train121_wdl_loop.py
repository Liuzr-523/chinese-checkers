import copy,unittest
import numpy as np
import torch
import rules121 as r
from az121 import Net121,encode
from wdl121 import WDLNet121,training_loss
from train121_wdl_loop import policy_only,policy_ce,prior_for_training

class WDLLoopIntegrationTests(unittest.TestCase):
    def test_legacy_outcomes_cannot_enter_policy_only_loss(self):
        torch.set_num_threads(1)
        old=dict(x=encode(r.init_board(),1),actions=[0,1],target=0,winner=2,reason='all_pieces_in_goal',value_weight=1.)
        before=copy.deepcopy(old);sample=policy_only(old)
        self.assertNotIn('winner',sample);self.assertNotIn('reason',sample)
        self.assertEqual(old['winner'],before['winner'])
        parent=Net121(width=4);model=WDLNet121.from_v3(parent)
        terms=training_loss(model,[sample],reference=parent,policy_weight=0,wdl_weight=0,kl_weight=.1)
        self.assertEqual(terms['wdl_n'],0);self.assertAlmostEqual(float(terms['total']),0.,places=6)
        model.train();flags=[p.requires_grad for p in model.parameters()]
        self.assertTrue(np.isfinite(policy_ce(model,[old])))
        self.assertTrue(model.training);self.assertEqual(flags,[p.requires_grad for p in model.parameters()])

    def test_prior_follows_game_uniform_not_position_count(self):
        samples=[dict(group=0,player=1,wdl_class=0)]+[dict(group=1,player=1,wdl_class=2) for _ in range(3)]
        prior=prior_for_training(samples)
        np.testing.assert_allclose(prior[1],[.4,.2,.4]);np.testing.assert_allclose(prior[2],[1/3]*3)

if __name__=='__main__':unittest.main()
