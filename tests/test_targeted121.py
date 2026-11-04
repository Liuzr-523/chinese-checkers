import copy,unittest
import rules121 as r
from az121 import Net121
from cycle121_targeted import configure_arm,value_label,select_seeds,verified_better,classify,expert_samples

class TargetedCycleTests(unittest.TestCase):
    def test_value_labels_never_fabricate_truncated_winner(self):
        self.assertEqual(value_label(0,'max_ply',1),(0.,0.))
        self.assertEqual(value_label(0,'equal_turn_goal_draw',2),(0.,1.))
        self.assertEqual(value_label(2,'all_pieces_in_goal',1),(-1.,1.))
        with self.assertRaises(ValueError):value_label(1,'max_ply',1)
        with self.assertRaises(ValueError):value_label(None,'',1)
        with self.assertRaises(ValueError):expert_samples(dict(opening=[],paths=[],winner=1,reason='all_pieces_in_goal',plies=0))

    def test_partial_vs_full_keeps_requested_freezing_boundaries(self):
        m=Net121(canonical=True);configure_arm(m,'last_block')
        self.assertFalse(any(p.requires_grad for p in m.stem.parameters()))
        self.assertFalse(any(p.requires_grad for p in m.blocks[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in m.blocks[-1].parameters()))
        self.assertTrue(all(p.requires_grad for p in m.value.parameters()))
        configure_arm(m,'full_cnn');self.assertTrue(all(p.requires_grad for p in m.parameters()))

    def test_seed_selection_preserves_game_split_and_caps_correlation(self):
        rows=[dict(game_id=g,ply=p,category='finish_delay',split='train' if g<3 else 'validation') for g in range(4) for p in range(10,20)]
        chosen=select_seeds(rows,'train',42)
        self.assertEqual(len(chosen),6);self.assertTrue(all(s['game_id']<3 for s in chosen))
        self.assertTrue(all(sum(s['game_id']==g for s in chosen)==2 for g in range(3)))

    def test_counterfactual_improvement_requires_real_outcome_or_faster_win(self):
        self.assertTrue(verified_better(dict(winner=1,plies=100),dict(winner=2,plies=90),1))
        self.assertFalse(verified_better(dict(winner=0,plies=100),dict(winner=0,plies=110),1))
        self.assertTrue(verified_better(dict(winner=1,plies=98),dict(winner=1,plies=100),1))
        self.assertFalse(verified_better(dict(winner=1,plies=100),dict(winner=1,plies=98),1))

    def test_missed_immediate_win_classification_uses_strict_rules(self):
        b=[0]*121
        for i in r.GOAL[1]-{111}:b[i]=1
        s=next(j for j in r.NEIGHBORS[111] if j not in r.GOAL[1]);b[s]=1
        for i in range(40,50):b[i]=2
        g=r.Game(board=b,player=1)
        expert=[s,111];other=next(path for _,path in r.get_all_moves(b,1) if path[-1]!=111)
        self.assertTrue(r.valid_path(b,1,expert));self.assertEqual(classify(g,other,expert),'missed_immediate_win')

if __name__=='__main__':unittest.main()
