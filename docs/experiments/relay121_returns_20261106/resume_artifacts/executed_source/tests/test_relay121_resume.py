import copy,json,tempfile,unittest
from pathlib import Path
from collect121_relay_returns import resume_inputs,relay_candidates
from curriculum121 import replay
from opening121_split_audit import orbit,decision_key
from checkpoint121_save import digest
import rules121 as r

class RelayResumeTests(unittest.TestCase):
    def fixture(self,folder):
        source=Path(folder);(source/'source').mkdir();(source/'run.lock').write_text('')
        cfg=dict(profile=r.PROFILE,seed=111091,groups=48,train=24,development=24,root_plies=[60]*4,max_attempts_per_group=128,max_seconds=3600,advance_depth=3,continuation_depths=[3,4],query_safety_seconds=120,max_ply=240,labels='fixed',scope='fixed')
        for name in ('collect121_relay_returns.py','dependency.py'):(source/'source'/name).write_text(name)
        cfg['source_sha256']={name:digest(source/'source'/name) for name in ('collect121_relay_returns.py','dependency.py')}
        seed=Path(__file__).resolve().parents[1]/'docs/experiments/skill121_review_20261105/inputs/targeted/seeds.json'
        opening=json.loads(seed.read_text())[13]['opening'];g=replay(opening);rear,front,w=relay_candidates(g);children=[]
        for path in (rear,front):
            x=copy.deepcopy(g);x.play(path);children.append(orbit(decision_key(x.board,x.player)))
        anchor=replay(opening[:8]);identity=decision_key(g.board,g.player)
        row=dict(group=0,split='train',bucket='ply56_57',board=g.board,player=g.player,ply=g.ply,opening=opening,root_id=identity,decision_id=identity,orbit=orbit(identity),anchor=orbit(decision_key(anchor.board,anchor.player)),child_orbits=children,rear_path=rear,front_path=front,mechanism_witness=w)
        for name,value in [('config.json',cfg),('status.json',dict(stage='failed',seconds=211.1236)),('history_exclusions.json',dict(orbits=[])),('schedule_partial.json',[row])]:
            (source/name).write_text(json.dumps(value))
        return source,cfg,row

    def test_reuses_partial_and_budget_without_writing_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source,cfg,row=self.fixture(folder);before={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
            cfg['source_sha256']['collect121_relay_returns.py']='changed collector permitted'
            history,rows,used,elapsed,provenance=resume_inputs(source,cfg)
            self.assertEqual(rows,[row]);self.assertEqual(elapsed,211.1236);self.assertEqual(len(used),4)
            self.assertEqual(provenance['restart_group'],1);self.assertEqual(provenance['restart_attempt'],0)
            self.assertEqual(before,{p.name:p.read_bytes() for p in source.iterdir() if p.is_file()})

    def test_changed_dependency_or_budget_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source,cfg,_=self.fixture(folder);changed=copy.deepcopy(cfg);changed['query_safety_seconds']=121
            with self.assertRaises(ValueError):resume_inputs(source,changed)
            changed=copy.deepcopy(cfg);changed['source_sha256']['dependency.py']='changed'
            with self.assertRaises(ValueError):resume_inputs(source,changed)

    def test_already_frozen_or_running_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source,cfg,_=self.fixture(folder);(source/'schedule_freeze.json').write_text('{}')
            with self.assertRaises(ValueError):resume_inputs(source,cfg)
            (source/'schedule_freeze.json').unlink();(source/'status.json').write_text(json.dumps(dict(stage='running',seconds=1)))
            with self.assertRaises(ValueError):resume_inputs(source,cfg)

    def test_history_collision_and_changed_mechanism_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source,cfg,row=self.fixture(folder);(source/'history_exclusions.json').write_text(json.dumps(dict(orbits=[row['orbit']])))
            with self.assertRaises(ValueError):resume_inputs(source,cfg)
            (source/'history_exclusions.json').write_text(json.dumps(dict(orbits=[])));row['mechanism_witness']['new_chain_advance']+=1
            (source/'schedule_partial.json').write_text(json.dumps([row]))
            with self.assertRaises(ValueError):resume_inputs(source,cfg)

if __name__=='__main__':unittest.main()
