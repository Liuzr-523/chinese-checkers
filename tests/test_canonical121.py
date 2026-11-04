import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
import rules121 as r
from az121 import Net121,encode,action_id,CANONICAL_ENCODING,search,train
from canonical121_trial import canonical_sample

class Canonical121Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_coordinate_mapping_preserves_original_turn_identity(self):
        self.assertTrue(all(r.COORDS[120-i]==(-q,-y) for i,(q,y) in enumerate(r.COORDS)))
        b=r.init_board();x=encode(b,2,canonical=True);red=encode(b,1)
        self.assertTrue(np.array_equal(x[:3],red[:3]))
        self.assertEqual(x[3].sum(),0);self.assertEqual(red[3].sum(),121)
        for s in range(121):
            a=action_id(s,120-s,2,True)
            self.assertEqual(action_id(a//121,a%121,2,True),s*121+120-s)

    def test_sample_conversion_keeps_winner_and_target_index(self):
        b=r.init_board();paths=[p for _,p in r.get_all_moves(b,2)]
        original=dict(x=encode(b,2),actions=[p[0]*121+p[-1] for p in paths],target=3,z=-1.,winner=1,endgame=False)
        converted=canonical_sample(original)
        self.assertEqual(converted['z'],-1.);self.assertEqual(converted['winner'],1);self.assertEqual(converted['target'],3)
        self.assertEqual(converted['encoding'],CANONICAL_ENCODING)
        self.assertEqual(converted['actions'][3],action_id(paths[3][0],paths[3][-1],2,True))
        self.assertFalse(np.shares_memory(original['x'],converted['x']))
        with self.assertRaises(ValueError):canonical_sample(converted)

    def test_search_maps_canonical_action_back_to_actual_move(self):
        b=r.init_board();desired=r.get_all_moves(b,2)[-1][1]
        chosen=action_id(desired[0],desired[-1],2,True)
        class Toy:
            canonical=True
            def eval(self):return self
            def __call__(self,x):
                logits=torch.zeros((len(x),121*121));logits[:,chosen]=20
                return logits,torch.zeros(len(x))
        path,info=search(Toy(),b,2,simulations=1)
        self.assertEqual((path[0],path[-1]),(desired[0],desired[-1]));self.assertTrue(r.valid_path(b,2,path))

    def test_checkpoint_declares_encoding_and_rejects_mixed_training_samples(self):
        m=Net121(canonical=True)
        from value121_repair import make_spatial
        self.assertTrue(make_spatial(m).canonical)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.pt';m.save(p);loaded=Net121.load(p)
            self.assertTrue(loaded.canonical);self.assertEqual(loaded.encoding,CANONICAL_ENCODING)
            x=torch.from_numpy(encode(r.init_board(),2,canonical=True)).unsqueeze(0)
            self.assertTrue(torch.equal(m(x)[0],loaded(x)[0]))
        sample=dict(x=encode(r.init_board(),1),actions=[0],policy=[1.],z=0.)
        with self.assertRaises(ValueError):train(m,[sample],steps=1)

if __name__=='__main__':unittest.main()
