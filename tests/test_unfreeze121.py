import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
import checkpoint121_save as slots
from az121 import Net121,encode
import rules121 as r
from unfreeze121_trial import objective

class Unfreeze121Tests(unittest.TestCase):
    def test_save_checksum_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/'model.pt').write_bytes(b'weights')
            (folder/'manifest.json').write_text(json.dumps(dict(files={'model.pt':slots.digest(folder/'model.pt')})))
            slots.verify(folder);(folder/'model.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):slots.verify(folder)

    def test_restore_preserves_previous_registry_and_restores_exact_saved_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);save=root/'save';save.mkdir();checkpoint='nn_checkpoints/models/checkpoints/test.pt'
            (save/'model.pt').write_bytes(b'original');registry=root/'registry.json';registry.write_bytes(b'newer registry')
            original=json.dumps(dict(version='v3',checkpoint=checkpoint)).encode();(save/'registry.json').write_bytes(original)
            manifest=dict(model_sha256=slots.digest(save/'model.pt'),files={p:slots.digest(save/p) for p in ['model.pt','registry.json']})
            (save/'manifest.json').write_text(json.dumps(manifest))
            with patch.object(slots,'ROOT',root),patch.object(slots,'REGISTRY',registry):slots.restore(save)
            self.assertEqual(registry.read_bytes(),original);self.assertEqual((root/checkpoint).read_bytes(),b'original')
            self.assertEqual(len(list(root.glob('neural121_before_restore_*.json'))),1)

    def test_unplayed_corrections_cannot_change_value_outcome_loss(self):
        torch.set_num_threads(1);torch.manual_seed(99);base=Net121(canonical=True);model=copy.deepcopy(base)
        xs=torch.from_numpy(encode(r.init_board(),1,canonical=True)).unsqueeze(0)
        actions=[s*121+p[-1] for s,p in r.get_all_moves(r.init_board(),1)]
        ids=torch.tensor([actions]);mask=torch.ones_like(ids,dtype=torch.bool);target=torch.tensor([0]);weight=torch.zeros(1)
        a=objective(model,base,xs,ids,mask,target,torch.tensor([1.]),weight)
        b=objective(model,base,xs,ids,mask,target,torch.tensor([-1.]),weight)
        self.assertEqual(float(a),float(b))
        old=model.stem.weight.detach().clone();opt=torch.optim.AdamW(model.parameters(),lr=.00003)
        opt.zero_grad();a.backward();opt.step()
        self.assertFalse(torch.equal(old,model.stem.weight))
        self.assertTrue(all(p.grad is None for p in base.parameters()))

if __name__=='__main__':unittest.main()
