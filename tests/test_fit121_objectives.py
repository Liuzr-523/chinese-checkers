import copy,unittest
import numpy as np
import torch
from az121 import Net121
from corrected121_decisions import decision_sample
from diagnostics.fit121_objectives import select32,objective_terms
import rules121 as r

class FitObjectiveTests(unittest.TestCase):
 def test_fixed_subset_covers_whole_train_groups(self):
  roots=[dict(decision_id=i) for i in range(160)];metadata=[dict(decision_id=i,pair=i//4) for i in range(200)]
  chosen,groups=select32(roots,metadata,111053)
  self.assertEqual(len(chosen),32);self.assertEqual(len(groups),8);self.assertEqual(len(set(groups)),8)
  self.assertTrue(all(s['decision_id']<160 for s in chosen))
 def test_policy_loss_ignores_outcome_labels_and_preserves_root_coefficient(self):
  torch.set_num_threads(1);torch.manual_seed(111053);base=Net121(canonical=True);model=copy.deepcopy(base);g=r.Game();paths=[p for _,p in r.get_all_moves(g.board,g.player)]
  root=decision_sample(g,paths[0],paths[1]);other={**root,'value_weight':1.,'z':1.};ordinary=[other]*12
  a,terms=objective_terms(model,base,[root]*4,ordinary,'root_policy')
  b,_=objective_terms(model,base,[{**root,'z':-1.}]*4,ordinary,'root_policy')
  self.assertEqual(float(a.detach()),float(b.detach()))
  _,mixed=objective_terms(model,base,[root]*4,ordinary,'policy_replay')
  torch.testing.assert_close(terms['root_ce'],mixed['root_ce']);torch.testing.assert_close(terms['rank'],mixed['rank'])
  self.assertEqual(float(terms['value'].detach()),0.)
  _,full=objective_terms(model,base,[root]*4,ordinary,'full_objective');self.assertGreater(float(full['value'].detach()),0.)
if __name__=='__main__':unittest.main()
