import copy,unittest
import numpy as np
import torch
from order121_metrics import pair_metrics,summarize,order_gate,order_metrics,prepare_rows
import rules121 as rules
class Dummy(torch.nn.Module):
 def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.zeros(3))
 def forward(self,x):return self.weight.unsqueeze(0).expand(len(x),-1),torch.zeros(len(x))
def row(gap,label=1):
 return dict(**pair_metrics(gap,0,label*4),argmax_in_pair=True,argmax_teacher_preferred=label*gap>0)
class OrderMetricsTests(unittest.TestCase):
 def test_tie_and_teacher_direction_not_rear_rule(self):
  x=pair_metrics(2,2,5);self.assertFalse(x['correct']);self.assertTrue(x['tie']);self.assertEqual(x['score_regret'],5);self.assertAlmostEqual(x['pair_ce'],np.log(2))
  self.assertTrue(pair_metrics(-1,2,-3)['correct']);self.assertEqual(pair_metrics(-1,2,-3)['score_regret'],0)
 def test_ce_stable_and_score_scale(self):
  self.assertTrue(np.isfinite(pair_metrics(-1000,1000,2)['pair_ce']));self.assertEqual(pair_metrics(-1,0,7)['score_regret'],7)
 def test_balanced_not_micro_accuracy(self):
  ss=[row(1)]*9+[row(1,-1)];m=summarize(ss);self.assertEqual(m['accuracy'],.9);self.assertEqual(m['balanced_accuracy'],.5)
 def test_missing_class_and_gate_threshold(self):
  a=summarize([row(1)]*8);self.assertIsNone(a['balanced_accuracy']);self.assertFalse(order_gate(a,a)[0])
  base=summarize([row(1)]*2+[row(-1)]*2+[row(-1,-1)]*2+[row(1,-1)]*2)
  good=summarize([row(2)]*4+[row(-2,-1)]*4);self.assertTrue(order_gate(good,base)[0]);self.assertFalse(order_gate(good,good)[1]['balanced_gain125'])
 def test_prepare_validates_frozen_labels_and_prefix(self):
  board=rules.init_board();moves=rules.get_all_moves(board,1);rear=moves[0][1];front=next(p for _,p in moves if (p[0],p[-1])!=(rear[0],rear[-1]))
  raw=dict(eligible=True,group=0,split="train",board=board,player=1,ply=0,opening=[],rear_path=rear,front_path=front,score4_rear=1,score4_front=0,score5_rear=3,score5_front=0,label=1,value_weight=0.,z=0.)
  good=prepare_rows([raw]);self.assertEqual(len(good),1);self.assertEqual(good[0]["value_weight"],0)
  with self.assertRaises(ValueError):prepare_rows([{**raw,"label":-1}])
  with self.assertRaises(ValueError):prepare_rows([{**raw,"ply":2}])
  self.assertEqual(prepare_rows([{"eligible":False}]),[])
 def test_actual_argmax_distinct_from_pair_and_state_restored(self):
  m=Dummy();m.weight.data=torch.tensor([1.,0.,3.]);m.train();m.weight.requires_grad_(True)
  s=dict(x=np.zeros((6,17,17),dtype=np.float32),eligible=True,group=0,split='development',player=1,ply=30,actions=[0,1,2],rear_action=0,front_action=1,teacher_gap=4)
  result=order_metrics(m,[s]);self.assertEqual(result['accuracy'],1);self.assertEqual(result['all_legal_argmax']['in_pair'],0);self.assertTrue(m.training);self.assertTrue(m.weight.requires_grad);self.assertIsNone(m.weight.grad)
if __name__=='__main__':unittest.main()
