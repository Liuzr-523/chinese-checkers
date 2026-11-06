import copy,unittest
from types import SimpleNamespace
import numpy as np
import torch
from relay121_targets import stable_preference
from relay121_metrics import pair_case,summarize,relay_gate,relay_metrics,_mechanism

def game(winner,turns=9):return dict(winner=winner,reason='all_pieces_in_goal' if winner else 'equal_turn_goal_draw',start_ply=20,root_player=1,root_ply=20,plies=20+turns,opening=[[]]*20,paths=[[]]*turns,own_moves=(turns+1)//2)
def preference(label=1,tempo=False):
 a,b=(game(1,5),game(1,9)) if tempo else (game(1),game(2))
 if label==-1:a,b=b,a
 return stable_preference({str(d):dict(rear=a,front=b) for d in (3,4)},1)
class Model(torch.nn.Module):
 def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor([1.,0.,3.]))
 def forward(self,x):return self.weight.expand(len(x),-1),torch.zeros(len(x))
class RelayMetricsTests(unittest.TestCase):
 def test_point_tie_is_conservative_and_not_tempo(self):
  c=pair_case(0,0,preference());self.assertFalse(c['correct']);self.assertEqual(c['per_depth']['3']['point_regret'],1);self.assertIsNone(c['per_depth']['3']['winning_tempo_regret'])
 def test_winning_tempo_separate_from_points(self):
  c=pair_case(-1,0,preference(tempo=True));self.assertEqual(c['per_depth']['3']['point_regret'],0);self.assertEqual(c['per_depth']['3']['winning_tempo_regret'],2)
 def test_empty_development_safe_and_no_model_forward(self):
  m=Model();r=relay_metrics(m,[]);self.assertEqual(r['n'],0);self.assertIsNone(r['balanced_accuracy']);self.assertFalse(relay_gate(r,r)[0]);self.assertIsNone(r['per_depth']['3']['winning_tempo_regret_mean'])
 def test_balanced_gate_and_tempo_not_applicable(self):
  good=[pair_case(sign,0,preference(sign)) for sign in (1,-1) for _ in range(4)];old=[pair_case(0,0,preference(sign)) for sign in (1,-1) for _ in range(4)];g,b=summarize(good),summarize(old);self.assertTrue(relay_gate(g,b)[0]);self.assertEqual(g['per_depth']['3']['winning_tempo_n'],0)
  g['per_depth']['3']['point_regret_mean']=2;self.assertFalse(relay_gate(g,b)[1]['depth3_point_regret_nonworse'])
 def test_winning_tempo_regret_gate_when_applicable(self):
  old=summarize([pair_case(0,0,preference(sign,True)) for sign in (1,-1) for _ in range(4)]);good=summarize([pair_case(sign,0,preference(sign,True)) for sign in (1,-1) for _ in range(4)]);self.assertTrue(relay_gate(good,old)[0]);good['per_depth']['4']['winning_tempo_regret_mean']=3;self.assertFalse(relay_gate(good,old)[1]['depth4_tempo_regret_nonworse'])
 def test_actual_argmax_can_be_outside_pair_and_flags_restored(self):
  m=Model();m.train();row=dict(x=np.zeros((6,17,17),dtype=np.float32),_relay_preference=preference(),label=1,eligible=True,group=1,split='development',player=1,ply=20,actions=[0,1,2],rear_action=0,front_action=1)
  r=relay_metrics(m,[row]);self.assertEqual(r['accuracy'],1);self.assertEqual(r['all_legal_argmax']['in_pair'],0);self.assertTrue(m.training);self.assertTrue(m.weight.requires_grad);self.assertIsNone(m.weight.grad)
 def test_mechanism_requires_new_endpoint_after_step(self):
  board=list(map(int,'2020220010000000000000000000120000000000000000001202000000000000000002000000001201000000000020000000000110000000000110100'))
  # 历史棋形作为纯规则fixture；不读取外部模型/搜索/数据。
  row=dict(rear_path=[8,16],front_path=[78,80,105],mechanism_witness=dict(best_new_chain=[16,39,58],new_chain_advance=4,rear_start_tip_distance=13))
  from relay121_metrics import _distance
  row['mechanism_witness']['rear_start_tip_distance']=_distance(8,1);_mechanism(row,SimpleNamespace(board=board,player=1))
  row['mechanism_witness']['best_new_chain']=[16,39]
  with self.assertRaises(ValueError):_mechanism(row,SimpleNamespace(board=board,player=1))
if __name__=='__main__':unittest.main()
