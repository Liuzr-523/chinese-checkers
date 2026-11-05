import copy,json,tempfile,unittest
from pathlib import Path
import torch
from az121 import encode,CANONICAL_ENCODING
from opening121 import key
import rules121 as r
from dual121_data import dual_loss,load_pilot,digest

class DualDataTests(unittest.TestCase):
 def fixture(self,folder):
  g=r.Game();identity=key(g.board,1);sets={'train':[],'development':[]};meta=[];schedule=[];probes=[]
  for group,part in enumerate(sets):
   schedule.append(dict(group=group,split=part));sample=dict(x=torch.from_numpy(encode(g.board,1,0,canonical=True)),actions=[1,2],target=0,z=1.,value_weight=1.,policy_weight=0.,group=group,game_id=group,decision_id=identity)
   sets[part].append(sample);meta.append(dict(game_id=group,group=group,split=part,ply=0,player=1,decision_id=identity,z=1.,value_weight=1.,policy_weight=0.,eligible=True,terminal_reason='all_pieces_in_goal',terminal_distance=4,policy_source='tip5'))
   probes.append(dict(**sample,split=part,policy_source='derived5_probe'))
   probes[-1].update(z=0.,value_weight=0.,policy_weight=1.)
  torch.save(dict(profile=r.PROFILE,encoding=CANONICAL_ENCODING,sets=sets,policy_probes=probes),folder/'dataset.pt')
  for name,value in [('schedule.json',schedule),('sample_metadata.json',meta),('result.json',dict(dataset_sha256=digest(folder/'dataset.pt')))]: (folder/name).write_text(json.dumps(value))
 def test_independent_denominators_and_zero_gradient(self):
  logits=torch.zeros(3,4,requires_grad=True);v=torch.tensor([0.,2.,7.],requires_grad=True);ss=[dict(actions=[0,1],target=0,z=1.,policy_weight=1.,value_weight=0.),dict(actions=[0,1],target=1,z=1.,policy_weight=0.,value_weight=1.),dict(actions=[0,1],target=0,z=-1.,policy_weight=0.,value_weight=0.)]
  d=dual_loss(logits,v,ss);self.assertAlmostEqual(float(d['policy_ce'].detach()),.69314718,places=6);self.assertEqual(float(d['value_mse'].detach()),1.);(d['policy_ce']+d['value_mse']).backward();torch.testing.assert_close(v.grad,torch.tensor([0.,2.,0.]));self.assertEqual(float(logits.grad[1:].abs().sum()),0.);self.assertEqual(float(logits.grad[:,2:].abs().sum()),0.)
 def test_all_masked_zero_loss_and_gradient(self):
  l=torch.zeros(1,4,requires_grad=True);v=torch.ones(1,requires_grad=True);s=[dict(actions=[1,2],target=0,z=-1.,policy_weight=0.,value_weight=0.)];d=dual_loss(l,v,s);(d['policy_ce']+d['value_mse']).backward();self.assertEqual(float(l.grad.abs().sum()+v.grad.abs().sum()),0.)
 def test_probe_split_and_tip_policy_mask_preserved(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);self.fixture(p);d=load_pilot(p);self.assertEqual([s['sample_id'] for s in d['train']['policy']],['probe:0:0']);self.assertEqual([s['sample_id'] for s in d['development']['policy']],['probe:1:0']);self.assertEqual(d['train']['value'][0]['policy_weight'],0.);self.assertEqual(d['train']['policy'][0]['value_weight'],0.)
 def test_tampered_actual_mask_rejected_even_with_updated_dataset_hash(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);self.fixture(p);raw=torch.load(p/'dataset.pt',weights_only=True);raw['sets']['train'][0]['policy_weight']=1.;torch.save(raw,p/'dataset.pt');(p/'result.json').write_text(json.dumps(dict(dataset_sha256=digest(p/'dataset.pt'))))
   with self.assertRaisesRegex(ValueError,'mask/target'):load_pilot(p)
 def test_excluded_dual_zero_cannot_be_reactivated(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);self.fixture(p);meta=json.loads((p/'sample_metadata.json').read_text());meta[0].update(eligible=False,policy_weight=0.,value_weight=0.);(p/'sample_metadata.json').write_text(json.dumps(meta));raw=torch.load(p/'dataset.pt',weights_only=True);raw['sets']['train'][0].update(policy_weight=0.,value_weight=0.);raw['policy_probes']=[s for s in raw['policy_probes'] if s['split']!='train'];torch.save(raw,p/'dataset.pt');(p/'result.json').write_text(json.dumps(dict(dataset_sha256=digest(p/'dataset.pt'))));d=load_pilot(p);self.assertEqual(d['train']['value'],[]);self.assertEqual(d['train']['policy'],[])
   meta[0]['value_weight']=1.;(p/'sample_metadata.json').write_text(json.dumps(meta));raw['sets']['train'][0]['value_weight']=1.;torch.save(raw,p/'dataset.pt');(p/'result.json').write_text(json.dumps(dict(dataset_sha256=digest(p/'dataset.pt'))))
   with self.assertRaisesRegex(ValueError,'resurrected'):load_pilot(p)
 def test_fractional_weights_use_true_independent_normalizers(self):
  l=torch.zeros(1,4,requires_grad=True);v=torch.ones(1,requires_grad=True);s=[dict(actions=[0,1],target=0,z=-1.,policy_weight=.1,value_weight=.2)];d=dual_loss(l,v,s);self.assertAlmostEqual(float(d['policy_ce'].detach()),.69314718,places=6);self.assertEqual(float(d['value_mse'].detach()),4.)
 def test_probe_development_resplit_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);self.fixture(p);raw=torch.load(p/'dataset.pt',weights_only=True);raw['policy_probes'][1]['split']='train';torch.save(raw,p/'dataset.pt');(p/'result.json').write_text(json.dumps(dict(dataset_sha256=digest(p/'dataset.pt'))))
   with self.assertRaisesRegex(ValueError,'split'):load_pilot(p)
if __name__=='__main__':unittest.main()
