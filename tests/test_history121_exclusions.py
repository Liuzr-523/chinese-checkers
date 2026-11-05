import json
import tempfile
import unittest
from pathlib import Path
import rules121 as r
from opening121_split_audit import orbit, decision_key, mirror
from history121_exclusions import collect_history


class HistoryExclusionTests(unittest.TestCase):
    def test_nested_identity_and_replay_all_states_and_sha_dedup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); folder = root / 'trial121';folder.mkdir()
            g = r.Game(); states = [orbit(decision_key(g.board, g.player))]; paths = []
            for _ in range(4):
                path = r.get_all_moves(g.board, g.player)[0][1];paths.append(path);g.play(path);states.append(orbit(decision_key(g.board, g.player)))
            identity = decision_key(g.board, g.player)
            row = dict(opening=paths[:2], paths=paths[2:], fixed_metadata=[dict(decision_id=mirror(identity))])
            (folder/'game.json').write_text(json.dumps(row))
            data=folder/'data_sources';data.mkdir();(data/'copy.json').write_text(json.dumps(row))
            source=folder/'source';source.mkdir();(source/'ignored.json').write_text('{}')
            result=collect_history(root)
            self.assertTrue(set(states).issubset(result['orbits']))
            self.assertEqual(result['coverage']['replayed_records'],1)
            self.assertEqual(result['coverage']['duplicate_content_files_skipped'],1)
            self.assertEqual(result['coverage']['source_or_cache_files_skipped'],1)

    def test_invalid_and_unknown_replays_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'trial121';a.mkdir();(a/'bad.json').write_text(json.dumps(dict(opening=[[0,120]])))
            (root/'unknown.json').write_text(json.dumps(dict(opening=[[0,120]])))
            (root/'legacy.pkl').write_bytes(b'not loaded')
            result=collect_history(root)
            self.assertEqual(result['coverage']['invalid_replay_records'],1)
            self.assertEqual(result['coverage']['unclassified_or_non121_replay_records'],1)
            self.assertEqual(result['coverage']['binary_files_not_parsed'],1)


class SafeTensorHistoryTests(unittest.TestCase):
    def test_numeric_numpy_and_tensor_samples_and_models(self):
        import torch
        from az121 import encode, CANONICAL_ENCODING, Net121
        from history121_exclusions import decode_sample_inputs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);folder=root/'trial121';folder.mkdir()
            g=r.Game();g.play(r.get_all_moves(g.board,g.player)[0][1])
            identity=decision_key(g.board,g.player)
            x=encode(g.board,g.player,g.ply,canonical=True)
            torch.save(dict(encoding=CANONICAL_ENCODING,samples=[dict(x=x)]),folder/'numpy.pt')
            torch.save(dict(samples=[dict(x=torch.tensor(x))]),folder/'tensor.pt')
            Net121(width=4,canonical=True).save(folder/'model.pt')
            result=collect_history(root)
            self.assertIn(orbit(identity),result['orbits'])
            self.assertEqual(result['coverage']['decoded_tensor_samples'],2)
            self.assertEqual(result['coverage']['model_state_dict_records'],1)
            self.assertEqual(result['incomplete'],[])
            self.assertEqual(decode_sample_inputs(x,CANONICAL_ENCODING),{identity})
            self.assertIn(identity,decode_sample_inputs(x,None))

    def test_excluded_directory_and_invalid_tensor(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'output121';out.mkdir()
            torch.save(dict(samples=[dict(x=torch.zeros(6,17,17))]),out/'bad.pt')
            result=collect_history(root,exclude=[out])
            self.assertEqual(result['orbits'],[])
            self.assertEqual(result['incomplete'],[])
            result=collect_history(root)
            self.assertEqual(len(result['incomplete']),1)

    def test_arbitrary_pickle_global_is_rejected(self):
        import torch
        from history121_exclusions import safe_tensor_load
        class Evil:
            def __reduce__(self):
                return (eval,('1+1',))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.pt';torch.save(Evil(),path)
            with self.assertRaises(Exception):safe_tensor_load(path)


class LegacyEndingResolutionTests(unittest.TestCase):
    def fixture(self, root, wrong_start=False):
        import hashlib
        base=root/'curriculum121_100_20261029';base.mkdir()
        g=r.Game();paths=[]
        for _ in range(10):
            path=r.get_all_moves(g.board,g.player)[0][1];paths.append(path);g.play(path)
        teacher=dict(group=7,winner=1,opening=paths[:1],paths=paths[1:])
        # opening(1) + paths[:-8](1) => recovered prefix length2.
        endings=[dict(model=name,group=7,side=1,start_ply=3 if wrong_start else 2,
                      plies=4,paths=paths[2:4]) for name in ('prior','candidate')]
        (base/'teacher_games.jsonl').write_text(json.dumps(teacher)+'\n')
        (base/'split.json').write_text(json.dumps(dict(test=[7])))
        (base/'endgame_evaluation.json').write_text(json.dumps(endings))
        prefix='curriculum121_100_20261029/'
        return dict(orbits=[],coverage={},source_sha256={prefix+p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in base.iterdir()},incomplete=[dict(location=prefix+'endgame_evaluation.json:0',reason='非初始suffix缺少前缀') for _ in endings])

    def test_restore_and_keep_original_evidence(self):
        from history121_exclusions import resolve_legacy_endgames
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=self.fixture(root);result=resolve_legacy_endgames(manifest,root)
            self.assertEqual(result['incomplete'],[])
            self.assertEqual(result['coverage']['legacy_endgame_records_resolved'],2)
            self.assertGreater(len(result['orbits']),0)
            self.assertEqual(len(result['resolved'][0]['original_incomplete']),2)
            self.assertEqual(len(manifest['incomplete']),2)

    def test_wrong_start_ply_rejected(self):
        from history121_exclusions import resolve_legacy_endgames
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=self.fixture(root,True)
            with self.assertRaises(ValueError):resolve_legacy_endgames(manifest,root)
