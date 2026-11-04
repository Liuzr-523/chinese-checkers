import unittest
from unittest.mock import patch
import numpy as np
import torch
import rules121 as r
from corrected121_decisions import eligible_decision,mixed_batch,data_gate,decision_sample,preference_loss,score_action,ConsistentExpert

class CorrectedDecisionTests(unittest.TestCase):
    def test_early_window_and_spacing_do_not_defer_to_late_seeds(self):
        self.assertTrue(eligible_decision(14,None,0));self.assertTrue(eligible_decision(26,20,1))
        self.assertFalse(eligible_decision(41,None,0));self.assertFalse(eligible_decision(25,20,1));self.assertFalse(eligible_decision(30,20,2))

    def test_explicit_decisions_remain_quarter_of_every_batch(self):
        rng=np.random.default_rng(31);roots=[{'rejected_action':i} for i in range(32)];old=[{'old':i} for i in range(1000)];normal=[{'normal':i} for i in range(80)]
        for _ in range(40):
            rows=mixed_batch(roots,old,normal,rng)
            self.assertEqual(len(rows),16);self.assertEqual(sum('rejected_action' in x for x in rows),4)
        with self.assertRaises(ValueError):mixed_batch([],old,normal,rng)

    def test_many_copies_of_few_roots_do_not_pass_coverage_gate(self):
        rows=[dict(split='train',pair=g) for g in range(12) for _ in range(2)]+[dict(split='validation',pair=20+g) for g in range(4) for _ in range(2)]
        self.assertTrue(data_gate(rows)[0])
        for row in rows:
            if row['split']=='train':row['pair']=1
        self.assertFalse(data_gate(rows)[0])

    def test_p2_preference_is_legal_canonical_and_has_no_outcome_label(self):
        g=r.Game(player=2);paths=[p for _,p in r.get_all_moves(g.board,g.player)]
        s=decision_sample(g,paths[0],paths[1])
        expected=(120-paths[0][0])*121+(120-paths[0][-1])
        self.assertEqual(s['actions'][s['target']],expected);self.assertEqual(s['value_weight'],0.)
        with self.assertRaises(ValueError):decision_sample(g,paths[0],paths[0])

    def test_preference_margin_penalizes_reversed_decisions(self):
        sample=dict(actions=[0,2],target=0,rejected_action=2)
        self.assertEqual(float(preference_loss(torch.tensor([[1.,0.,0.]]),[sample])),0.)
        self.assertGreater(float(preference_loss(torch.tensor([[0.,0.,1.]]),[sample])),0.)

    def test_forced_action_respects_real_repetition_draw_before_search(self):
        g=r.Game();path=r.get_all_moves(g.board,g.player)[0][1];child=r.apply_move(g.board,path)
        g.repetitions[(tuple(child),2)]=2
        with patch('corrected121_decisions.ConsistentExpert',side_effect=AssertionError('must not search past terminal')):
            self.assertEqual(score_action(g,path),0)

    def test_expert_refuses_unfinished_fixed_depth(self):
        e=ConsistentExpert()
        with patch('engine_clysto121.ClystoDerived121.think',return_value=([0,1],dict(aborted=True,depth_reached=4))):
            with self.assertRaises(RuntimeError):e.think(r.init_board(),1)

if __name__=='__main__':unittest.main()
