import unittest
import numpy as np
import torch
from az121 import Net121
from corrected121_decisions import decision_sample
import rules121 as r
from generalization121_curve import nested_subset,remove_validation_duplicates,training_batch,position_key,predict_preferences

class GeneralizationCurveTests(unittest.TestCase):
    def test_control_is_reproducible_nested_subset_of_training_groups(self):
        roots=[dict(decision_id=i) for i in range(160)];metadata=[dict(decision_id=i,pair=i//4) for i in range(200)]
        chosen=nested_subset(roots,metadata)
        self.assertEqual(len(chosen),72);self.assertEqual(len({s['decision_id'] for s in chosen}),72)
        self.assertEqual(len({metadata[s['decision_id']]['pair'] for s in chosen}),18)
        self.assertEqual(chosen,nested_subset(roots,metadata));self.assertTrue(all(s['decision_id']<160 for s in chosen))
        with self.assertRaises(ValueError):nested_subset(roots[:60],metadata)

    def test_same_ordinary_samples_despite_different_root_pool_sizes(self):
        small=[dict(root=i) for i in range(72)];large=[dict(root=i) for i in range(160)];old=[dict(old=i) for i in range(1000)];normal=[dict(normal=i) for i in range(100)]
        a,b=np.random.default_rng(1),np.random.default_rng(1);x,y=np.random.default_rng(2),np.random.default_rng(2)
        for _ in range(100):
            left=training_batch(small,old,normal,a,x);right=training_batch(large,old,normal,b,y)
            self.assertEqual(left[4:],right[4:]);self.assertEqual(sum('root' in s for s in left),4);self.assertEqual(len(right),16)

    def test_exact_heldout_input_removed_from_all_finetuning_replay(self):
        validation=[dict(x=np.zeros((6,17,17),dtype=np.float32))]
        samples=[dict(x=validation[0]['x'].copy()),dict(x=np.ones((6,17,17),dtype=np.float32))]
        kept,removed=remove_validation_duplicates(samples,validation)
        self.assertEqual(removed,1);self.assertEqual(len(kept),1);self.assertEqual(len(samples),2)

    def test_saved_model_prediction_summary_has_required_loss_import(self):
        torch.set_num_threads(1);g=r.Game();paths=[p for _,p in r.get_all_moves(g.board,g.player)]
        sample=decision_sample(g,paths[0],paths[1]);sample['decision_id']=7
        result=predict_preferences(Net121(canonical=True),[sample])
        self.assertEqual(result[0]['decision_id'],7);self.assertTrue(np.isfinite(result[0]['ce']))
        self.assertIsInstance(result[0]['preferred'],bool)

    def test_root_dedup_does_not_count_ply_repetitions_as_new_board_experience(self):
        a=r.Game();b=r.Game();b.ply=2
        self.assertEqual(position_key(a),position_key(b));b.player=2
        self.assertNotEqual(position_key(a),position_key(b))

if __name__=='__main__':unittest.main()
