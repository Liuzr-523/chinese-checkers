import unittest
from selfplay121_round import score

class SelfPlayRoundTests(unittest.TestCase):
    def test_score_handles_swapped_sides_and_true_draws(self):
        rows=[dict(winner=1,model_side=1,reason='all_pieces_in_goal'),
              dict(winner=1,model_side=2,reason='anti_block_goal_full'),
              dict(winner=0,model_side=2,reason='repetition')]
        result=score(rows)
        self.assertEqual((result['wins'],result['losses'],result['draws']),(1,1,1))
        self.assertEqual(score([])['games'],0)
        self.assertEqual(result['reasons']['repetition'],1)

if __name__=='__main__':unittest.main()
