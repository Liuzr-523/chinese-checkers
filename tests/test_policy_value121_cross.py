import unittest
import torch
import rules121 as r
from az121 import Net121,encode
from diagnostics.policy_value121_cross import CrossPolicyValue,matched_effect

class CrossValueTests(unittest.TestCase):
    def test_full_old_value_and_new_policy_are_exact_and_frozen(self):
        torch.set_num_threads(1);torch.manual_seed(111051);old=Net121(canonical=True);new=Net121(canonical=True);mix=CrossPolicyValue(new,old)
        for player in (1,2):
            x=torch.from_numpy(encode(r.init_board(),player,canonical=True)).unsqueeze(0)
            logits,value=mix(x);new_logits,_=new(x);_,old_value=old(x)
            self.assertTrue(torch.equal(logits,new_logits));self.assertTrue(torch.equal(value,old_value))
        self.assertFalse(any(p.requires_grad for p in mix.parameters()))
        with self.assertRaises(ValueError):CrossPolicyValue(Net121(canonical=True),Net121(canonical=False))

    def test_matched_effect_counts_opening_groups_not_individual_positions(self):
        a=[];b=[]
        for pair in range(6):
            for side in (1,2):
                a.append(dict(pair=pair,side=side,opening=[],winner=3-side))
                b.append(dict(pair=pair,side=side,opening=[],winner=side))
        result=matched_effect(a,b)
        self.assertEqual(result['opening_groups'],6);self.assertEqual(result['mean_game_score_rate_difference'],1.)
        self.assertAlmostEqual(result['one_sided_sign_p'],1/64)
        with self.assertRaises(ValueError):matched_effect(a,b[:-1])

if __name__=='__main__':unittest.main()
