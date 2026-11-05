import copy,unittest
import numpy as np
import torch
from value121_metrics import value_metrics,gate
class FakeValue(torch.nn.Module):
 def forward(self,x):return torch.zeros((len(x),1)),x[:,0,0,0]
def sample(pred,z,group='a',game='g',player=1,late=True):
 x=np.zeros((6,17,17),dtype=np.float32);x[0,0,0]=pred;x[3,:,:]=int(player==1)
 return dict(x=x,z=z,group=group,game_id=game,player=player,late16=late)
class ValueMetricsTests(unittest.TestCase):
 def train(self):return [sample(0,1,player=1),sample(0,-1,player=2)]
 def test_group_macro_uses_points_not_game_means(self):
  # group a: one long good game (3 points), one short bad game (1 point).
  ss=[sample(1,1,game='long') for _ in range(3)]+[sample(-1,1,game='short')]+[sample(0,1,group='b')]
  m=value_metrics(FakeValue(),ss,self.train());self.assertAlmostEqual(m['point']['mse'],1.);self.assertAlmostEqual(m['group_macro']['mse'],1.)
  self.assertAlmostEqual(m['per_group']['a']['point']['mse'],1.) # game average would incorrectly be2
  self.assertEqual(len(m['per_game']),3)
 def test_prior_uses_training_only_and_sign_corr(self):
  ss=[sample(0,-1,player=1),sample(0,1,player=2)]
  m=value_metrics(FakeValue(),ss,self.train());self.assertEqual(m['train_player_prior'],{'1':1.,'2':-1.});self.assertEqual(m['point']['train_player_prior_mse'],4.);self.assertEqual(m['point']['zero_mse'],1.);self.assertIsNone(m['point']['correlation']);self.assertEqual(m['point']['prediction_std'],0.)
 def test_group_macro_differs_from_point_and_restores_mode(self):
  model=FakeValue();model.train();ss=[sample(0,1,group='a') for _ in range(3)]+[sample(1,1,group='b')]
  m=value_metrics(model,ss,self.train());self.assertEqual(m['point']['mse'],.75);self.assertEqual(m['group_macro']['mse'],.5);self.assertTrue(model.training)
 def test_missing_training_player_or_group_rejected(self):
  with self.assertRaises(ValueError):value_metrics(FakeValue(),[sample(0,1)],[sample(0,1)])
  s=sample(0,1);del s['group']
  with self.assertRaises(ValueError):value_metrics(FakeValue(),[s],self.train())
 def test_gate_zero_prior_and_late_side_and_old_limits(self):
  # balanced train makes zero and training priors equal; good predictions beat both.
  tr=[sample(0,z,player=p) for p in (1,2) for z in (-1,1)];ss=[sample(.8,1,player=1),sample(-.8,-1,player=2)]
  m=value_metrics(FakeValue(),ss,tr);b=value_metrics(FakeValue(),[sample(0,1,player=1),sample(0,-1,player=2)],tr)
  def wrap(v):return dict(development=v,retention=dict(policy_ce=2.,value_mse=1.),normal=dict(policy_ce=2.))
  c,b=wrap(m),wrap(b);self.assertTrue(gate(c,b)[0]);c['retention']['policy_ce']=2.051;self.assertFalse(gate(c,b)[1]['old_policy'])
  c=wrap(m);c['development']=copy.deepcopy(m);c['development']['late16']['players']['2']['group_macro_mse']=1.1;self.assertFalse(gate(c,b)[1]['late16_player2'])
 def test_terminal_distance_and_policy_only_filtered(self):
  s=sample(.5,1);del s['late16'];s['terminal_distance']=16
  noise={**s,'z':-1,'value_weight':0};m=value_metrics(FakeValue(),[s,noise],self.train()+[noise]);self.assertEqual(m['n'],1);self.assertEqual(m['late16']['n'],1);self.assertEqual(m['train_player_prior']['1'],1.);self.assertEqual(m['group_macro_mse'],.25)
 def test_empty_late_subset_fails_gate(self):
  tr=[sample(0,z,player=p) for p in (1,2) for z in (-1,1)];m=value_metrics(FakeValue(),[sample(0,1,late=False)],tr);w=dict(development=m,retention=dict(policy_ce=1,value_mse=1),normal=dict(policy_ce=1));self.assertFalse(gate(w,w)[0])
if __name__=='__main__':unittest.main()
