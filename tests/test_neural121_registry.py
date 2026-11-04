import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import rules121 as r
from az121 import Net121
import neural121_registry as registry

class NeuralRegistryTests(unittest.TestCase):
    def test_load_and_hash_refusal(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'nn_checkpoints/models/checkpoints/model.pt';path.parent.mkdir(parents=True)
            Net121().save(path)
            reg=root/'registry.json'
            entry=dict(schema='neural121-registry-v1',profile=r.PROFILE,checkpoint=str(path.relative_to(root)),
                       sha256=registry.digest(path),simulations=32,value_scale=1.,version='test')
            reg.write_text(json.dumps(entry))
            with patch.object(registry,'ROOT',root),patch.object(registry,'REGISTRY',reg):
                engine=registry.load_registered()
                path_out,info=engine.think(r.init_board(),1)
                self.assertTrue(r.valid_path(r.init_board(),1,path_out));self.assertEqual(info['root_visits'],32)
                entry['sha256']='corrupt';reg.write_text(json.dumps(entry))
                with self.assertRaises(ValueError):registry.load_registered()

    def test_registration_rejects_unqualified_experiment(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'config.json').write_text(json.dumps(dict(profile=r.PROFILE)))
            (root/'result.json').write_text(json.dumps(dict(qualified=False)))
            with self.assertRaises(ValueError):registry.audit_experiment(root)

if __name__=='__main__':unittest.main()
