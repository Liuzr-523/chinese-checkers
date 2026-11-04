import unittest
import torch
import rules121 as r
from az121 import Net121,encode,search,Neural121
from reanalyze121_policy import features,logits_from_features,batch,decision_target,loss_on

class ReanalysisTests(unittest.TestCase):
    def test_cached_policy_matches_full_model_and_head_update_freezes_value(self):
        torch.set_num_threads(1);torch.manual_seed(301040);model=Net121(canonical=True)
        x=torch.from_numpy(encode(r.init_board(),2,canonical=True)).unsqueeze(0)
        with torch.no_grad():
            cells=features(model,x).detach();old_value=model(x)[1].clone()
            torch.testing.assert_close(logits_from_features(model,cells),model(x)[0],atol=2e-6,rtol=1e-6)
        saved={k:v.clone() for k,v in model.state_dict().items()}
        model.requires_grad_(False)
        for head in [model.src,model.dst,model.bias]:head.requires_grad_(True)
        opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
        for _ in range(2):
            logits=logits_from_features(model,cells);loss=-logits.log_softmax(1)[0,100]
            opt.zero_grad();loss.backward();opt.step()
        for k,v in model.state_dict().items():
            if not k.startswith(('src.','dst.','bias.')):self.assertTrue(torch.equal(saved[k],v))
        with torch.no_grad():torch.testing.assert_close(old_value,model(x)[1],atol=2e-6,rtol=1e-6)

    def test_decision_targets_preserve_action_order_and_normalize(self):
        self.assertEqual(decision_target([2,2,0], 'hard'), [1.,0.,0.])
        sharp=decision_target([1,2,0], 'sharp')
        self.assertAlmostEqual(sum(sharp),1.)
        self.assertAlmostEqual(sharp[1]/sharp[0],16.)
        for invalid in ([0,0],[-1,2],[float('nan'),1]):
            with self.assertRaises(ValueError):decision_target(invalid,'hard')

    def test_temperature_prior_and_default_behavior(self):
        torch.set_num_threads(1);torch.manual_seed(42);model=Net121(canonical=True)
        board=r.init_board()
        path,base=search(model,board,2,simulations=4)
        same,explicit=search(model,board,2,simulations=4,policy_temperature=1.)
        self.assertEqual(path,same);self.assertEqual(base,explicit)
        changed,info=Neural121(model,4,policy_temperature=.5).think(board,2)
        expected=torch.tensor(base['priors'],dtype=torch.float64).square();expected/=expected.sum()
        torch.testing.assert_close(torch.tensor(info['priors'],dtype=torch.float64),expected,atol=1e-7,rtol=1e-6)
        self.assertIn(changed,info['paths']);self.assertEqual(info['child_visits'],4)
        for invalid in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):Neural121(model,policy_temperature=invalid)
            with self.assertRaises(ValueError):search(model,board,2,policy_temperature=invalid)

    def test_hard_selection_loss_matches_manual_legal_ce(self):
        torch.set_num_threads(1);torch.manual_seed(43);model=Net121(canonical=True)
        x=torch.from_numpy(encode(r.init_board(),2,canonical=True)).unsqueeze(0)
        with torch.no_grad():cells=features(model,x)[0].detach()
        sample=dict(features=cells,actions=[20,41,65],clean=[.2,.6,.2])
        actual=loss_on(model,[sample],hard=True,temperature=.5)
        with torch.no_grad():expected=-logits_from_features(model,cells.unsqueeze(0))[0,[20,41,65]].div(.5).log_softmax(0)[1]
        self.assertAlmostEqual(actual,float(expected),places=5)

if __name__=='__main__':unittest.main()
