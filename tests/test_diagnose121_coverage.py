import unittest
from az121 import encode, action_id
import rules121 as r
from diagnostics.diagnose121_coverage import describe, summarize, stage_of


class CoverageTests(unittest.TestCase):
    def test_stage_boundaries(self):
        self.assertEqual([stage_of(n) for n in (1,3,4,6,7,10,11,20,21,40,41)],
                         ['1—3','1—3','4—6','4—6','7—10','7—10','11—20','11—20','21—40','21—40','41及以后'])

    def test_player_ply_and_value_mask_not_fake_draw(self):
        g=r.Game();g.play(r.get_all_moves(g.board,g.player)[0][1])
        moves=r.get_all_moves(g.board,g.player);actions=[action_id(s,p[-1],g.player,True) for s,p in moves]
        s=dict(x=encode(g.board,g.player,13,canonical=True),actions=actions,target=0,z=0.,value_weight=0.)
        d=describe(s);self.assertEqual((d['player'],d['ply'],d['own_move']),(2,13,7))
        result=summarize([s]);self.assertEqual(result['active_value_labels'],{});self.assertEqual(result['zero_value_weight'],1)
        self.assertEqual(result['unique_board_turn'],1)

    def test_outcome_diversity_is_counted_not_relabelled(self):
        g=r.Game();moves=r.get_all_moves(g.board,g.player);actions=[action_id(s,p[-1],g.player,True) for s,p in moves]
        s=dict(x=encode(g.board,g.player,canonical=True),actions=actions,target=0,z=1.,value_weight=1.)
        result=summarize([s,{**s,'target':1,'z':-1.}])
        self.assertEqual(result['duplicates']['differing_value_groups'],1)
        self.assertEqual(result['duplicates']['disjoint_policy_groups'],1)
        self.assertEqual(result['active_value_labels'],{'1.0':1,'-1.0':1})
