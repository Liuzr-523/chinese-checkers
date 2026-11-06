import unittest
import rules121 as r
from collect121_order_pairs import tip_distance,farthest,select_candidates,label_scores


class OrderCollectionTests(unittest.TestCase):
    def test_tip_distances(self):
        self.assertEqual(tip_distance(120,1),0);self.assertEqual(tip_distance(0,2),0)
        self.assertEqual(tip_distance(0,1),tip_distance(120,2))
        self.assertEqual(farthest(r.init_board(),1),tip_distance(0,1))

    def test_label_stability_and_threshold(self):
        self.assertEqual(label_scores(5,1,6,1),1)
        self.assertEqual(label_scores(1,5,1,3),-1)
        for scores in [(1,1,4,1),(1,3,4,1),(3,1,2,1)]:self.assertEqual(label_scores(*scores),0)

    def test_geometry_pair_never_uses_network(self):
        import random
        rng=random.Random(14);g=r.Game()
        for _ in range(50):
            if g.result()[0] is not None:break
            pair=select_candidates(g)
            if pair:
                rear,front=pair;before=farthest(g.board,g.player)
                self.assertTrue(r.valid_path(g.board,g.player,rear));self.assertTrue(r.valid_path(g.board,g.player,front))
                self.assertLess(farthest(r.apply_move(g.board,rear),g.player),before)
                self.assertGreaterEqual(farthest(r.apply_move(g.board,front),g.player),before)
                self.assertGreaterEqual(len(front),3)
                self.assertGreater(tip_distance(front[0],g.player)-tip_distance(front[-1],g.player),0)
                return
            g.play(rng.choice([p for _,p in r.get_all_moves(g.board,g.player)]))
        self.fail('固定随机合法样本未发现几何候选对')

    def test_fixed_depth_audit_uses_original_scoring_semantics_without_search(self):
        from unittest.mock import patch
        import corrected121_decisions as corrected
        from collect121_order_pairs import audited_score
        class FakeExpert:
            def __init__(self,depth):self.requested_depth=depth
            def think(self,*args,**kwargs):
                return None,dict(depth_reached=self.requested_depth,aborted=False,score=7)
        g=r.Game();path=r.get_all_moves(g.board,g.player)[0][1]
        with patch.object(corrected,'ConsistentExpert',FakeExpert):
            score,audit=audited_score(g,path,5)
            self.assertEqual(score,-7)
            self.assertEqual(audit['search'][0]['requested_depth'],4)
            self.assertEqual(audit['search'][0]['depth_reached'],4)
            self.assertIs(corrected.ConsistentExpert,FakeExpert)
