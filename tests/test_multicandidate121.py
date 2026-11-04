import unittest
import torch
import rules121 as r
from az121 import action_id
from corrected121_decisions import preference_loss
from multicandidate121_trial import multi_loss,witness,development_gate,SEEDS

class MultiCandidateTests(unittest.TestCase):
 def test_single_positive_reduces_to_original_ce_and_pair_hinge(self):
  logits=torch.tensor([[.3,.1,-.2]],requires_grad=True);s=dict(actions=[0,1,2],target=0,rejected_action=1,candidate_scores={0:10,1:6})
  ce,rank=multi_loss(logits,[s]);torch.testing.assert_close(ce,torch.nn.functional.cross_entropy(logits,torch.tensor([0])));torch.testing.assert_close(rank,preference_loss(logits,[s]))
 def test_equal_score_good_move_counts_and_third_bad_move_is_penalized(self):
  s=dict(actions=[0,1,2,3],target=0,rejected_action=1,candidate_scores={0:10,1:5,2:10,3:0});a=torch.tensor([[1.,0.,1.,3.]],requires_grad=True)
  ce,rank=multi_loss(a,[s]);self.assertEqual(float(preference_loss(a,[s]).detach()),0.);self.assertGreater(float(rank.detach()),0.)
  b=a.detach().clone();b[0,0]-=.7;b[0,2]=torch.log(torch.exp(a.detach()[0,0])+torch.exp(a.detach()[0,2])-torch.exp(b[0,0]));ce2,_=multi_loss(b,[s]);torch.testing.assert_close(ce,ce2)
 def test_witness_canonical_player_two(self):
  g=r.Game();g.play(r.get_all_moves(g.board,g.player)[0][1]);self.assertEqual(g.player,2)
  for start,p in r.get_all_moves(g.board,g.player):self.assertEqual(witness(g,action_id(start,p[-1],g.player,True)),p)
 def test_both_seeds_and_retention_are_required(self):
  before=dict(retention=dict(policy_ce=2.));rows={}
  for seed in SEEDS:
   rows[f'old_{seed}']=dict(validation_actual=dict(mean_regret=8.,bad_fraction=.8,equal_best_rate=.2),retention=dict(policy_ce=2.02))
   rows[f'multi_{seed}']=dict(validation_actual=dict(mean_regret=6.,bad_fraction=.6,equal_best_rate=.3),retention=dict(policy_ce=2.04))
  self.assertTrue(development_gate(rows,before)[0]);rows[f'multi_{SEEDS[1]}']['retention']['policy_ce']=2.1;self.assertFalse(development_gate(rows,before)[0])
if __name__=='__main__':unittest.main()
