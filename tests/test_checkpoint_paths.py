import sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import checkpoint_paths as cp

class CheckpointPathsTests(unittest.TestCase):
    def test_routes_legacy_names_and_keeps_explicit_paths(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cp,'ROOT',Path(tmp)/'nn_checkpoints'):
            root=cp.ROOT
            for name,folder in [('params_resnet_champ.pt','models/champions'),('params_x_ckpt_0200.pkl','models/checkpoints'),('params_x_last.pt','models/final'),('replay.pkl','data/replays'),('az_r1_0.pkl','data/selfplay'),('states_a.pkl','data/positions'),('dgrad_a.pkl','data/dagger')]:
                path=cp.artifact_path(root/name)
                self.assertEqual(path,root/folder/name)
                path.write_bytes(b'test')
                self.assertEqual(cp.artifact_path(path),path)
            self.assertEqual(len(cp.artifact_glob(root,'params_*.p*')),3)
            custom=Path(tmp)/'custom.pkl'
            self.assertEqual(cp.artifact_path(custom),custom)
            self.assertEqual(cp.artifact_path(root),root)

if __name__=='__main__':unittest.main()
