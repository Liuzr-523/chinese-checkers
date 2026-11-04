import unittest
from unittest.mock import patch
import numpy as np
import torch
import rules121 as r
from az121 import Net121,Neural121,CANONICAL_ENCODING,action_id
from train121_trial import play
from selfplay121_retained import batch,metrics

class RetainedSelfplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_canonical_selfplay_samples_keep_real_frame_actions_and_metadata(self):
        model=Net121(canonical=True);engine=Neural121(model,1)
        original=r.Game
        with patch.object(r,'Game',side_effect=lambda:original(max_ply=2)):
            game,samples=play({1:engine,2:engine},301038)
        self.assertEqual(game['reason'],'max_ply');self.assertEqual(game['winner'],0)
        self.assertEqual(len(samples),2)
        for step,sample in zip(game['steps'],samples):
            p=step['player'];expected={action_id(s,path[-1],p,True) for s,path in r.get_all_moves(step['board'],p)}
            self.assertEqual(set(sample['actions']),expected)
            self.assertEqual(sample['encoding'],CANONICAL_ENCODING)
            self.assertEqual(sample['value_weight'],0.)
        xs,ids,mask,pi,z,w=batch(samples)
        self.assertEqual(xs.shape,(2,6,17,17));self.assertTrue(torch.allclose(pi.sum(1),torch.ones(2)))
        self.assertEqual(float(w.sum()),0.)
        report=metrics(model,samples)
        self.assertTrue(np.isfinite(report['policy_ce']));self.assertEqual(report['value_samples'],0.)

if __name__=='__main__':unittest.main()
