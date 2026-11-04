import json
from pathlib import Path
import unittest
import torch
from torch.nn import functional as F
from az121 import Net121
from value121_repair import make_spatial
from curriculum121 import samples_from_game,batch

class ValueStabilityTests(unittest.TestCase):
    def test_normalized_head_has_learning_signal_and_preserves_policy(self):
        source=Path('nn_checkpoints/experiments/correct121_failures_20261030_b')
        if not (source/'candidate.pt').exists():self.skipTest('local trained feature fixture absent')
        torch.set_num_threads(1);torch.manual_seed(301033)
        base=Net121.load(source/'candidate.pt');model=make_spatial(base,normalized=True)
        games=[json.loads(x) for x in (source/'corrections.jsonl').read_text().splitlines()[:4]]
        samples=[s for g in games for s in samples_from_game(g)][:32]
        xs,*_,z=batch(samples)
        for p in model.parameters():p.requires_grad=False
        for p in model.value.parameters():p.requires_grad=True
        opt=torch.optim.AdamW(model.value.parameters(),lr=1e-4)
        initial=float(F.mse_loss(model(xs)[1],z).detach())
        for _ in range(12):
            _,v=model(xs);loss=F.mse_loss(v,z)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.value.parameters(),5);opt.step()
        value=model(xs)[1]
        self.assertLess(float(F.mse_loss(value,z).detach()),initial)
        for name,tensor in model.state_dict().items():
            if not name.startswith('value.'):
                self.assertTrue(torch.equal(base.state_dict()[name],tensor))
        with torch.no_grad():
            a,b=base(xs)[0],model(xs)[0]
            torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-6)
            self.assertTrue(torch.equal(a.argmax(1),b.argmax(1)))
        self.assertLess(float((value.detach().abs()>.999).float().mean()),1.)
        self.assertEqual(model.value_head,'spatial_norm')

if __name__=='__main__':unittest.main()
