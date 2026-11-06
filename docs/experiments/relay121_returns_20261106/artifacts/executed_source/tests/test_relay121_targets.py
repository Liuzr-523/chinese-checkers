import unittest
from relay121_targets import compare_returns,stable_preference

def game(winner,turns=9,reason=None):
 return dict(winner=winner,reason=reason or ('all_pieces_in_goal' if winner else 'equal_turn_goal_draw'),start_ply=20,root_ply=20,root_player=1,plies=20+turns,opening=[[]]*20,paths=[[]]*turns,own_moves=(turns+1)//2)
class RelayTargetTests(unittest.TestCase):
 def test_points_override_tempo_and_draw_is_real(self):
  a=compare_returns(game(1,31),game(0,5),1);self.assertEqual((a['label'],a['kind'],a['point_difference']),(1,'point',.5))
  self.assertEqual(compare_returns(game(0),game(2),1)['label'],1)
 def test_max_ply_and_unfinished_no_preference(self):
  self.assertEqual(compare_returns(game(0,reason='max_ply'),game(2),1)['label'],0)
  a=game(1);a['winner']=None;self.assertEqual(compare_returns(a,game(2),1)['label'],0)
 def test_own_moves_includes_forced_and_two_move_threshold(self):
  a=compare_returns(game(1,5),game(1,9),1);self.assertEqual((a['label'],a['kind'],a['own_moves_rear'],a['own_moves_front']),(1,'tempo',3,5))
  self.assertEqual(compare_returns(game(1,5),game(1,7),1)['label'],0)
 def test_same_losses_or_draws_do_not_rank_tempo(self):
  for winner in (0,2):self.assertEqual(compare_returns(game(winner,5),game(winner,15),1)['label'],0)
 def test_stability_direction_not_kind(self):
  d={'3':dict(rear=game(1),front=game(2)),'4':dict(rear=game(1,5),front=game(1,9))};a=stable_preference(d,1);self.assertEqual((a['label'],a['eligible'],a['kind']),(1,True,'mixed'))
  d['4']=dict(rear=game(2),front=game(1));self.assertFalse(stable_preference(d,1)['eligible'])
 def test_invalid_metadata_not_silently_accepted(self):
  a=game(1);a['own_moves']=999
  with self.assertRaises(ValueError):compare_returns(a,game(2),1)
if __name__=='__main__':unittest.main()
