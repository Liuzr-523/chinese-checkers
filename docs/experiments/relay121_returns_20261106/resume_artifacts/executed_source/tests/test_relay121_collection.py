import copy,json,unittest
from pathlib import Path
import rules121 as r
from curriculum121 import replay
from opening121_split_audit import orbit,decision_key
from collect121_order_pairs import tip_distance,farthest
from collect121_relay_returns import relay_witness,relay_candidates,readiness,mask_cross_inputs


class RelayCollectionTests(unittest.TestCase):
    def motivating_root(self):
        path=Path(__file__).resolve().parents[1]/'docs/experiments/skill121_review_20261105/inputs/targeted/seeds.json'
        return replay(json.loads(path.read_text())[13]['opening'])

    def test_original_small_step_has_new_forward_chain(self):
        g=self.motivating_root();self.assertEqual((g.ply,g.player),(60,1));w=relay_witness(g,[8,16])
        self.assertIsNotNone(w);self.assertIn([16,39,58],w['new_forward_chains'])
        self.assertNotIn(58,{p[-1] for p in w['original_forward_chains']})
        self.assertGreaterEqual(len(w['best_new_chain']),3)
        self.assertGreater(w['new_chain_advance'],0)

    def test_selection_is_deterministic_and_mechanistic(self):
        g=self.motivating_root();pair=relay_candidates(g);self.assertIsNotNone(pair)
        rear,front,w=pair;self.assertEqual(pair,relay_candidates(g));self.assertEqual(len(rear),2)
        self.assertIn(rear[-1],r.NEIGHBORS[rear[0]])
        self.assertEqual(tip_distance(rear[0],g.player),farthest(g.board,g.player))
        self.assertEqual(w,relay_witness(g,rear));self.assertLess(tip_distance(front[0],g.player),farthest(g.board,g.player));self.assertGreaterEqual(len(front),3)
        self.assertGreater(tip_distance(front[0],g.player)-tip_distance(front[-1],g.player),0)
        for path in w['new_forward_chains']:self.assertTrue(r.valid_path(r.apply_move(g.board,rear),g.player,path))
        self.assertIsNone(relay_witness(g,front))

    def test_train_and_development_readiness_are_independent(self):
        rows=[dict(split='train',eligible=True,label=1 if i<2 else -1) for i in range(8)]
        rows += [dict(split='development',eligible=True,label=1 if i<1 else -1) for i in range(8)]
        ready,gates,counts=readiness(rows);self.assertTrue(ready['training']);self.assertFalse(ready['development'])
        self.assertEqual(counts['development']['1'],1)

    def test_cross_input_masks_both_groups_without_deleting_returns(self):
        identity=orbit(decision_key(r.init_board(),1));game=dict(opening=[],paths=[])
        rows=[dict(group=i,split=part,orbit=identity,child_orbits=[f'child{i}a',f'child{i}b'],returns={'3':{'rear':copy.deepcopy(game),'front':copy.deepcopy(game)}},label=1 if i==0 else -1,eligible=True) for i,part in enumerate(('train','development'))]
        old=copy.deepcopy([s['returns'] for s in rows]);audit=mask_cross_inputs(rows,set())
        self.assertEqual(audit['masked_groups'],[0,1]);self.assertTrue(all(s['label']==0 and not s['eligible'] for s in rows))
        self.assertEqual([s['raw_label'] for s in rows],[1,-1]);self.assertEqual([s['returns'] for s in rows],old)

    def test_old_single_jump_endpoint_is_not_new_relay(self):
        from unittest.mock import patch
        g=self.motivating_root();old_path=[8,58];new_path=[16,39,58]
        def mock_paths(board,start):
            return [old_path] if start==8 else [new_path]
        with patch('collect121_relay_returns.r.legal_paths',side_effect=mock_paths):
            self.assertIsNone(relay_witness(g,[8,16]))
