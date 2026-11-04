import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from curriculum121 import split_groups,samples_from_game,metrics
from az121 import Net121

class Curriculum121Tests(unittest.TestCase):
    def test_opening_groups_are_disjoint_and_complete(self):
        splits=split_groups(50,12)
        self.assertEqual({k:len(v) for k,v in splits.items()},{'train':35,'validation':5,'test':10})
        self.assertFalse(splits['train'] & splits['test'])
        self.assertFalse(splits['train'] & splits['validation'])
        self.assertFalse(splits['test'] & splits['validation'])
        self.assertEqual(set.union(*splits.values()),set(range(50)))
        self.assertEqual(splits,split_groups(50,12))

    def test_no_draw_value_labels_and_actual_terminal_checked(self):
        self.assertEqual(samples_from_game(dict(winner=0,reason='max_ply')),[])
        # This uses a real, existing fully logged game as a local fixture when present.
        path=Path('nn_checkpoints/experiments/az121_trial_20261029_b/games.jsonl')
        if not path.exists(): self.skipTest('optional historical game fixture missing')
        game=json.loads(path.read_text().splitlines()[0])
        record=dict(group=4,winner=game['winner'],reason=game['reason'],
                    opening=[s['path'] for s in game['steps'][:4]],paths=[s['path'] for s in game['steps'][4:]])
        samples=samples_from_game(record)
        self.assertEqual(sum(s['endgame'] for s in samples),16)
        self.assertEqual({s['z'] for s in samples},{-1.,1.})
        self.assertTrue(all(s['group']==4 for s in samples))
        bad=copy.deepcopy(record);bad['winner']=3-record['winner']
        with self.assertRaises(ValueError): samples_from_game(bad)
        torch.set_num_threads(1)
        result=metrics(Net121(),samples[-4:])
        self.assertEqual(result['samples'],4)
        self.assertTrue(np.isfinite(result['policy_ce']))

if __name__=='__main__':unittest.main()
