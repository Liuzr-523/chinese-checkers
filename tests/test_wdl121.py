import math,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import rules121 as r
from az121 import Net121,encode
from wdl121 import WDLNet121,terminal_class,training_loss,training_batch
from wdl121_metrics import summarize,wdl_metrics,metrics_from_probabilities

class WDLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def sample(self,label=0,**kw):
        return dict(x=encode(r.init_board(),1),actions=[1,2],policy=[.25,.75],wdl_class=label,player=1,game_id='a',ply=20,**kw)

    def test_parent_policy_preserved_exactly_and_initial_value_uniform(self):
        for canonical in (False,True):
            base=Net121(width=8,canonical=canonical);model=WDLNet121.from_v3(base,'sha',3)
            x=torch.from_numpy(np.stack([encode(r.init_board(),p,canonical=canonical) for p in (1,2)]))
            old,_=base(x);logits,expected=model(x)
            self.assertTrue(torch.equal(logits,old));self.assertTrue(torch.equal(expected,torch.zeros(2)))
            self.assertTrue(torch.allclose(model.predict_wdl(x),torch.full((2,3),1/3)))
            self.assertTrue(all(not p.requires_grad for p in model.value.parameters()))
            for key,value in base.state_dict().items():self.assertTrue(torch.equal(value,model.state_dict()[key]),key)

    def test_terminal_classes_preserve_side_and_real_draws(self):
        self.assertEqual(terminal_class(1,1,'all_pieces_in_goal'),0);self.assertEqual(terminal_class(1,2,'all_pieces_in_goal'),2)
        for reason in ('equal_turn_goal_draw','repetition','no_legal_moves'):
            for player in (1,2):self.assertEqual(terminal_class(0,player,reason),1)
        self.assertIsNone(terminal_class(0,1,'max_ply'));self.assertIsNone(terminal_class(None,2,''))
        with self.assertRaises(ValueError):terminal_class(0,1,'unknown')
        with self.assertRaises(ValueError):terminal_class(1,1,'repetition')

    def test_distinct_checkpoint_roundtrip_and_old_format_rejection(self):
        model=WDLNet121.from_v3(Net121(width=8),parent_sha256='parent',iteration=2);x=torch.from_numpy(self.sample()['x']).unsqueeze(0)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'wdl.pt';model.save(path,metadata={'note':'中文'})
            restored=WDLNet121.load(path);self.assertEqual(restored.metadata,dict(parent_sha256='parent',iteration=2,note='中文'))
            self.assertTrue(torch.equal(model.forward_wdl(x)[1],restored.forward_wdl(x)[1]))
            with self.assertRaises(ValueError):Net121.load(path)
            Net121(width=8).save(path)
            with self.assertRaises(ValueError):WDLNet121.load(path)

    def test_truncated_sample_cannot_create_wdl_gradient(self):
        model=WDLNet121(width=8)
        sample=self.sample(0,winner=0,reason='max_ply')
        result=training_loss(model,[sample],policy_weight=0.,wdl_weight=1.)
        result['total'].backward();self.assertEqual(result['wdl_n'],0);self.assertEqual(float(result['wdl_ce'].detach()),0.)
        for p in model.wdl.parameters():self.assertIsNotNone(p.grad);self.assertEqual(int(torch.count_nonzero(p.grad)),0)
        self.assertTrue(all(p.grad is None for p in model.value.parameters()))

    def test_soft_policy_wdl_ce_and_kl_definitions(self):
        base=Net121(width=8);model=WDLNet121.from_v3(base);sample=self.sample(1)
        result=training_loss(model,[sample],base,kl_weight=.5)
        logits,_=model.forward_wdl(torch.from_numpy(sample['x']).unsqueeze(0));expected=-(logits[0,[1,2]].log_softmax(0)*torch.tensor([.25,.75])).sum()
        self.assertAlmostEqual(float(result['policy_ce'].detach()),float(expected.detach()),places=6)
        self.assertAlmostEqual(float(result['wdl_ce'].detach()),math.log(3),places=6);self.assertAlmostEqual(float(result['policy_kl'].detach()),0.,places=6)
        self.assertEqual(result['wdl_n'],1)
        result['total'].backward();self.assertGreater(float(model.wdl[-1].bias.grad.abs().sum()),0)
        with self.assertRaises(ValueError):training_batch([self.sample(1,policy_weight=-1)])

    def test_probability_metrics_have_stated_definitions(self):
        rows=[dict(probabilities=[.6,.3,.1],label=0),dict(probabilities=[.2,.5,.3],label=2)]
        m=summarize(rows)
        self.assertAlmostEqual(m['brier3'],(.26+.78)/2)
        self.assertAlmostEqual(m['nll'],(-math.log(.6)-math.log(.3))/2)
        self.assertAlmostEqual(m['expectedpoint_mse'],((.75-1)**2+.45**2)/2)
        self.assertAlmostEqual(m['ece10'],(.4+.5)/2)
        self.assertEqual(m['accuracy'],.5)
        self.assertEqual(summarize([])['nll'],None)

    def test_expected_value_uses_win_minus_loss_and_constant_baselines_match(self):
        model=WDLNet121(width=8)
        with torch.no_grad():model.wdl[-1].bias.copy_(torch.tensor([.6,.3,.1]).log())
        sample=self.sample(2);x=torch.from_numpy(sample['x']).unsqueeze(0)
        self.assertAlmostEqual(float(model(x)[1].detach()[0]),.5,places=6)
        measured=wdl_metrics(model,[sample])
        direct=metrics_from_probabilities([sample],model.predict_wdl(x).detach().numpy())
        self.assertEqual(measured,direct)
        with self.assertRaises(ValueError):metrics_from_probabilities([sample],[])

    def test_metrics_mask_and_group_without_changing_model_state(self):
        model=WDLNet121(width=8);model.train();samples=[self.sample(0,terminal_ply=25),self.sample(1,value_weight=0)]
        flags=[p.requires_grad for p in model.parameters()];result=wdl_metrics(model,samples)
        self.assertEqual(result['overall']['n'],1);self.assertEqual(result['late']['n'],1)
        self.assertEqual(result['by_player']['2']['n'],0);self.assertEqual(result['game_macro']['n_games'],1)
        self.assertTrue(model.training);self.assertEqual([p.requires_grad for p in model.parameters()],flags)
        self.assertAlmostEqual(result['overall']['brier3'],2/3,places=6)

if __name__=='__main__':unittest.main()
