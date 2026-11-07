import json,tempfile,unittest
from pathlib import Path
import rules121 as r
from history121_exclusions import collect_history
from opening121_split_audit import orbit,decision_key
class HistorySearchMetadataTests(unittest.TestCase):
    def scan(self,record):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'search121';root.mkdir();(root/'data.json').write_text(json.dumps(record));return collect_history(root)
    def test_candidates_are_not_continuous_game_and_actual_game_is_kept(self):
        g=r.Game();opening=[]
        for _ in range(8):path=r.get_all_moves(g.board,g.player)[0][1];g.play(path);opening.append(path)
        root_state=orbit(decision_key(g.board,g.player));paths=[p for _,p in r.get_all_moves(g.board,g.player)];n=len(paths);info=dict(paths=paths,priors=[1/n]*n,policy=[1/n]*n,root_visits=32,child_visits=32)
        path=paths[0];g.play(path);child_state=orbit(decision_key(g.board,g.player));result=self.scan(dict(opening=opening,paths=[path],search_info=[dict(info=info)]))
        self.assertFalse(result['incomplete']);self.assertIn(root_state,result['orbits']);self.assertIn(child_state,result['orbits']);self.assertEqual(result['coverage']['search_candidate_records'],1);self.assertEqual(result['coverage']['replayed_records'],1)
    def test_bad_real_game_is_still_incomplete(self):
        result=self.scan(dict(opening=[],paths=[[0,120]]));self.assertTrue(result['incomplete']);self.assertEqual(result['coverage']['invalid_replay_records'],1)
    def test_unmarked_paths_are_not_silently_ignored(self):
        result=self.scan(dict(paths=[[100,101],[102,103]]));self.assertTrue(result['incomplete'])
    def test_malformed_candidate_shape_is_incomplete(self):
        result=self.scan(dict(paths=[[0,10]],priors=[1.],policy=[.5],root_visits=32,child_visits=32));self.assertTrue(result['incomplete']);self.assertEqual(result['coverage']['invalid_search_candidate_records'],1)
if __name__=='__main__':unittest.main()
