import copy,unittest
import numpy as np
import torch
from az121 import Net121,encode,action_id
import rules121 as r
from curriculum121 import replay
from order121_learning import configure,value_state_hash,pair_ids,prepare_samples,order_batch,paired_policy_loss,retention_terms

class OrderLearningTests(unittest.TestCase):
 def root(self,player=1,label=1,eligible=True,split='train'):
  g=r.Game()
  if player==2:g.play(r.get_all_moves(g.board,1)[0][1])
  # Preserve exact prefix for player2 from the actual first legal move.
  opening=[] if player==1 else [r.get_all_moves(r.Game().board,1)[0][1]]
  paths=[p for _,p in r.get_all_moves(g.board,g.player)];return dict(group=0,split=split,board=g.board,player=player,ply=g.ply,opening=opening,rear_path=paths[0],front_path=next(p for p in paths if (p[0],p[-1])!=(paths[0][0],paths[0][-1])),label=label,eligible=eligible)
 def test_label_direction_and_mask_zero(self):
  row=self.root();a,b=pair_ids(row);l=torch.zeros(2,14641,requires_grad=True)
  with torch.no_grad():l[0,a]=2
  pos=paired_policy_loss(l,[row,{**row,'label':0,'eligible':False}]);neg=paired_policy_loss(l,[{**row,'label':-1},{**row,'label':0,'eligible':False}]);self.assertLess(float(pos.detach()),float(neg.detach()));pos.backward();self.assertLess(float(l.grad[0,a]),0);self.assertGreater(float(l.grad[0,b]),0);self.assertEqual(float(l.grad[1].abs().sum()),0)
 def test_mask_zero_gradient_and_resurrection_development_rejected(self):
  row=self.root(label=0,eligible=False);l=torch.zeros(1,14641,requires_grad=True);paired_policy_loss(l,[row]).backward();self.assertEqual(float(l.grad.abs().sum()),0)
  with self.assertRaisesRegex(ValueError,'resurrected'):paired_policy_loss(l,[{**row,'label':1}])
  with self.assertRaisesRegex(ValueError,'development'):paired_policy_loss(l,[{**row,'split':'development'}])
 def test_canonical_both_players_and_full_history_encoding(self):
  for p in (1,2):
   row=self.root(p);a,b=pair_ids(row);self.assertEqual(a,action_id(row['rear_path'][0],row['rear_path'][-1],p,True));self.assertEqual(b,action_id(row['front_path'][0],row['front_path'][-1],p,True));g=replay(row['opening']);prepared=prepare_samples([row]);torch.testing.assert_close(order_batch(prepared),torch.from_numpy(encode(g.board,p,g.ply,g.repetitions,canonical=True)).unsqueeze(0))
  row=self.root();row['ply']=1
  with self.assertRaisesRegex(ValueError,'prefix'):prepare_samples([row])
 def test_repeated_root_keeps_full_repetition_history(self):
  g=r.Game();a=r.get_all_moves(g.board,g.player)[0][1];g.play(a);b=r.get_all_moves(g.board,g.player)[0][1];g.play(b);opening=[a,b,list(reversed(a)),list(reversed(b))];g=replay(opening);row={**self.root(),'board':g.board,'player':g.player,'ply':g.ply,'opening':opening};x=order_batch([row]);self.assertAlmostEqual(float(x[0,5].max()),2/3,places=6);self.assertFalse(torch.equal(x,torch.from_numpy(encode(g.board,g.player,g.ply,canonical=True)).unsqueeze(0)))
 def test_common_logit_shift_cancels(self):
  row=self.root();l=torch.randn(1,14641);torch.testing.assert_close(paired_policy_loss(l,[row]),paired_policy_loss(l+3,[row]))
 def test_frozen_value_and_shared_gradient_with_retention(self):
  torch.set_num_threads(1);torch.manual_seed(34);m=configure(Net121(width=4,canonical=True));ref=copy.deepcopy(m);ref.requires_grad_(False);row=self.root();x=order_batch([row]);before=value_state_hash(m);l,v=m(x);loss=paired_policy_loss(l,[row]);loss.backward();self.assertTrue(all(p.grad is None for p in m.value.parameters()));self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum())>0 for p in m.stem.parameters()))
  opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001);opt.step();self.assertEqual(value_state_hash(m),before)
  sample=dict(x=x[0].numpy(),actions=list(pair_ids(row)),target=0,z=1000.)
  terms=retention_terms(m,ref,[sample]);opt.zero_grad();(terms['policy_kl']+terms['value_preservation']).backward();self.assertTrue(all(p.grad is None for p in m.value.parameters()));self.assertTrue(all(p.grad is None for p in ref.parameters()));self.assertGreater(float(terms['value_preservation'].detach()),0)
if __name__=='__main__':unittest.main()
