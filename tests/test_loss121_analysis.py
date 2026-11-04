import itertools,unittest
from types import SimpleNamespace
import rules121 as r
from diagnostics.analyze121_losses import position_cost,distances,measure,phase,replay_game
class LossAnalysisTests(unittest.TestCase):
 def test_locked_goal_assignment_matches_bruteforce(self):
  targets=sorted(r.GOAL[1]);pieces=tuple(targets[2:]+[0,1]);expected=min(sum(distances(s)[t] for s,t in zip([0,1],order)) for order in itertools.permutations(targets[:2]));self.assertEqual(position_cost(pieces,1),expected)
 def test_symmetry_and_finished_cost(self):
  b=r.init_board();self.assertEqual(measure(b,1)['assignment'],measure(b,2)['assignment']);self.assertEqual(position_cost(tuple(sorted(r.GOAL[1])),1),0)
 def test_phase_boundaries_and_goal_threshold(self):
  g=SimpleNamespace(board=r.init_board(),ply=23);self.assertEqual(phase(g,1),'opening');g.ply=24;self.assertEqual(phase(g,1),'middle')
  g.board=[0]*121
  for i in sorted(r.GOAL[1])[:7]:g.board[i]=1
  for i in (0,1,2):g.board[i]=1
  self.assertEqual(phase(g,1),'endgame')
 def test_incorrect_terminal_label_is_rejected(self):
  with self.assertRaises(ValueError):replay_game(dict(side=1,pair=0,winner=1,reason='all_pieces_in_goal',plies=0,start_ply=0,opening=[],paths=[]))
if __name__=='__main__':unittest.main()
