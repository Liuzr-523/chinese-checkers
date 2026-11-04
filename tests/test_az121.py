import unittest
from collections import Counter
from pathlib import Path
import tempfile
import numpy as np
import torch
import rules121 as r
from az121 import Net121,encode,search,train

class AZ121Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_encoding_and_checkpoint_version(self):
        b=r.init_board();x=encode(b,1);y=encode(b,2)
        self.assertEqual(x.shape,(6,17,17));self.assertEqual(x[2].sum(),121)
        self.assertTrue(np.array_equal(x[0],y[1]))
        model=Net121();logits,value=model(torch.from_numpy(x).unsqueeze(0))
        self.assertEqual(logits.shape,(1,121*121));self.assertEqual(value.shape,(1,))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt';model.save(path);loaded=Net121.load(path)
            self.assertTrue(torch.equal(logits,loaded(torch.from_numpy(x).unsqueeze(0))[0]))
            torch.save(dict(state_dict={}),path)
            with self.assertRaises(ValueError): Net121.load(path)

    def test_spatial_value_head_preserves_policy_and_roundtrips(self):
        from value121_repair import make_spatial
        torch.manual_seed(9);base=Net121();spatial=make_spatial(base)
        x=torch.from_numpy(encode(r.init_board(),1)).unsqueeze(0)
        self.assertTrue(torch.equal(base(x)[0],spatial(x)[0]))
        self.assertEqual(spatial(x)[1].shape,(1,))
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'spatial.pt';spatial.save(p);loaded=Net121.load(p)
            self.assertEqual(loaded.value_head,'spatial')
            self.assertTrue(torch.equal(loaded(x)[1],spatial(x)[1]))
            # Existing v1 checkpoints lack this architecture field.
            base.save(p);d=torch.load(p,weights_only=True);d.pop('value_head');torch.save(d,p)
            old=Net121.load(p);self.assertEqual(old.value_head,'pooled')
            self.assertTrue(torch.equal(base(x)[0],old(x)[0]))

    def test_visits_and_terminal_history(self):
        b=r.init_board();model=Net121()
        for n in (1,8,32):
            path,info=search(model,b,1,simulations=n)
            self.assertTrue(r.valid_path(b,1,path))
            self.assertEqual(info['root_visits'],n);self.assertEqual(info['child_visits'],n)
            self.assertAlmostEqual(sum(info['policy']),1)
        counts=Counter({(tuple(b),1):3})
        self.assertIsNone(search(model,b,1,repetitions=counts)[0])
        self.assertIsNone(search(model,b,1,ply=240)[0])
        # All child arrivals are the third repetition: leaf values must be zero.
        counts=Counter({(tuple(b),1):1})
        for _,p in r.get_all_moves(b,1): counts[(tuple(r.apply_move(b,p)),2)]=2
        self.assertIsNotNone(search(model,b,1,repetitions=counts,simulations=8)[0])

    def test_limit_value_targets_are_masked(self):
        import copy
        b=r.init_board(); paths=[p for _,p in r.get_all_moves(b,1)]
        sample=dict(x=encode(b,1),actions=[p[0]*121+p[-1] for p in paths],
                    policy=[1/len(paths)]*len(paths),z=1.,value_weight=0.)
        a=Net121(); other=copy.deepcopy(a)
        one=train(a,[sample],steps=2,seed=7)
        two=train(other,[{**sample,'z':-1.}],steps=2,seed=7)
        self.assertEqual(one,two)
        for left,right in zip(a.parameters(),other.parameters()):
            self.assertTrue(torch.equal(left,right))

    def test_backup_sign_and_training(self):
        b=[0]*121
        # P2 has nine pieces in its goal and can finish with one step.
        for i in range(9): b[i]=2
        outside=next(i for i in r.NEIGHBORS[9] if i not in r.GOAL[2]);b[outside]=2
        free=[i for i in range(30,100) if not b[i]]
        for i in free[:10]: b[i]=1
        model=Net121();path,info=search(model,b,2,simulations=48)
        self.assertEqual(path[-1],9)
        zero_path,_=search(model,b,2,simulations=48,value_scale=0.)
        self.assertEqual(zero_path[-1],9)
        with self.assertRaises(ValueError): search(model,b,2,value_scale=-1.)
        ids=[p[0]*121+p[-1] for p in info['paths']]
        sample=dict(x=encode(b,2),actions=ids,policy=info['policy'],z=1.)
        outcome=train(model,[sample],steps=2)
        self.assertTrue(np.isfinite(outcome['last_loss']))

if __name__=='__main__': unittest.main()
