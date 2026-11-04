import copy,unittest
import torch
import rules121 as r
from az121 import Net121
from corrected121_decisions import decision_sample
from diagnostics.ablate121_supervision32 import best_rank,objective_terms
class AblationTests(unittest.TestCase):
 def test_rank_does_not_promote_inferior_moves_against_worse_moves(self):
  logits=torch.zeros((1,4),requires_grad=True);s=dict(candidate_scores={0:10,1:6,2:1,3:0});loss=best_rank(logits,[s]);grad=torch.autograd.grad(loss,logits)[0][0];self.assertLess(float(grad[0]),0);self.assertTrue(all(float(grad[i])>=0 for i in (1,2,3)))
 def test_only_named_loss_component_changes(self):
  torch.set_num_threads(1);base=Net121(canonical=True);model=copy.deepcopy(base);g=r.Game();paths=[p for _,p in r.get_all_moves(g.board,g.player)];s=decision_sample(g,paths[0],paths[1]);s['candidate_scores']={s['actions'][s['target']]:10,s['rejected_action']:0,s['actions'][2]:10};ordinary=[{**s,'value_weight':1.,'z':1.}]*12
  _,control=objective_terms(model,base,[s]*4,ordinary,'control')
  for arm,changed in [('ties_ce_only','root_ce'),('best_rank_only','rank')]:
   _,terms=objective_terms(model,base,[s]*4,ordinary,arm)
   for key in control:
    if key!=changed:torch.testing.assert_close(control[key],terms[key])
if __name__=='__main__':unittest.main()
