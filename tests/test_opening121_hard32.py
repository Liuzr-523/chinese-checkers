import copy,unittest
import numpy as np
import torch
from torch.nn import functional as F
from az121 import Net121
from opening121 import OpeningBook
from train_opening121 import book_samples,set_ce
from fit_opening121_hard32 import balanced_orbits,fit_loss
from curriculum121 import batch
class Hard32Tests(unittest.TestCase):
 def test_selection_balances_players_stages_and_keeps_orbits(self):
  candidates=[dict(orbit=f'{p}-{stage}-{j}',player=p,own_move=stage,branch=f'{stage}-{j}') for p in (1,2) for stage in (7,8,9,10) for j in range(3)]
  chosen=balanced_orbits(candidates);self.assertEqual(len(chosen),16);self.assertEqual(sum(c['player']==1 for c in chosen),8);self.assertEqual({c['own_move'] for c in chosen},{7,8,9,10});self.assertEqual(len({c['orbit'] for c in chosen}),16)
 def test_root_coefficient_and_outcome_mask(self):
  torch.set_num_threads(1);m=Net121(width=4,canonical=True);base=copy.deepcopy(m);s=book_samples(OpeningBook())[0]['train'][0];root={**s,'z':1000.,'value_weight':0.};x,*_=batch([root]);l,v=m(x);ordinary={**s,'value_weight':1.,'z':float(v.detach()[0])}
  pure=fit_loss(m,base,[root]*4,[ordinary]*12,'root_only');expected=.25*set_ce(l,[root]);torch.testing.assert_close(pure,expected)
  mixed=fit_loss(m,base,[root]*4,[ordinary]*12,'mixed');ce=F.cross_entropy(l[:,s['actions']],torch.tensor([s['target']]));torch.testing.assert_close(mixed-pure,.75*ce,rtol=1e-5,atol=1e-6)
if __name__=='__main__':unittest.main()
