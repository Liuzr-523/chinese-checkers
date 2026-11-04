import unittest
import torch
from dynamic121_trial import dynamic_rank
class DynamicRankingTests(unittest.TestCase):
 def test_penalizes_actual_third_move_after_original_pair_satisfied(self):
  logits=torch.tensor([[1.,0.,3.]],requires_grad=True);s=dict(decision_id=1,actions=[0,1,2],target=0,rejected_action=1);scores={0:10,1:0,2:4}
  loss,events=dynamic_rank(logits,[s],lambda s,a:scores[a]);self.assertEqual(events[0]['action'],2);self.assertTrue(events[0]['active']);g=torch.autograd.grad(loss,logits)[0];self.assertLess(float(g[0,0]),0);self.assertGreater(float(g[0,2]),0);self.assertEqual(float(g[0,1]),0)
 def test_accepts_equal_score_or_small_gap_without_ranking_penalty(self):
  for value in (10,8):
   logits=torch.tensor([[1.,3.]],requires_grad=True);s=dict(decision_id=1,actions=[0,1],target=0);loss,events=dynamic_rank(logits,[s],lambda s,a:10 if a==0 else value);self.assertFalse(events[0]['active']);self.assertEqual(float(loss.detach()),0);self.assertEqual(float(torch.autograd.grad(loss,logits)[0].abs().sum()),0)
 def test_uses_legal_actions_and_fixed_batch_mean(self):
  logits=torch.tensor([[1.,9.,3.],[3.,9.,1.]],requires_grad=True);samples=[dict(decision_id=i,actions=[0,2],target=0) for i in range(2)];loss,events=dynamic_rank(logits,samples,lambda s,a:10 if a==0 else 0);self.assertEqual([e['action'] for e in events],[2,0]);self.assertAlmostEqual(float(loss.detach()),1.25)
if __name__=='__main__':unittest.main()
