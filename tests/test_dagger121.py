import unittest
import torch
import rules121 as r
from az121 import Net121,action_id
from dagger121_expert import correction_sample,teacher_qualified

class Dagger121Tests(unittest.TestCase):
    def test_p2_expert_label_uses_actual_legal_history_without_value_label(self):
        torch.set_num_threads(1);model=Net121(canonical=True);g=r.Game(player=2)
        path=r.get_all_moves(g.board,2)[0][1];before=g.board.copy()
        sample=correction_sample(model,g,path)
        self.assertEqual(g.board,before)
        self.assertEqual(set(sample),{'features','actions','clean'})
        self.assertEqual(sum(sample['clean']),1.)
        self.assertEqual(sample['actions'][sample['clean'].index(1.)],action_id(path[0],path[-1],2,True))
        self.assertEqual(sample['features'].shape,(121,model.width));self.assertFalse(sample['features'].requires_grad)
        with self.assertRaises(ValueError):correction_sample(model,g,[0,120])

    def test_no_learning_without_current_expert_evidence(self):
        rows=[dict(pair=i//2,side=i%2+1,winner=i%2+1,reason='all_pieces_in_goal') for i in range(40)]
        self.assertTrue(teacher_qualified(rows));self.assertFalse(teacher_qualified(rows[:38]))
        for row in rows:row['winner']=0;row['reason']='equal_turn_goal_draw'
        self.assertFalse(teacher_qualified(rows))

if __name__=='__main__':unittest.main()
