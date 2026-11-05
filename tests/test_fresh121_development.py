import unittest
from fresh121_projection_trial import generate_schedule
from curriculum121 import replay
from opening121 import key
from opening121_split_audit import orbit,rotate_swap,mirror

class FreshDevelopmentTests(unittest.TestCase):
    def test_schedule_balanced_deterministic_and_all_symmetries_excluded(self):
        initial,_=generate_schedule(set(),count=4);excluded={orbit(initial[0]['cases'][0]['decision_id'])}
        rows,audit=generate_schedule(excluded,count=4);again,_=generate_schedule(excluded,count=4);self.assertEqual(rows,again)
        identities=[]
        for group in rows:
            self.assertEqual([r['player'] for r in group['cases']],[1,2]);self.assertNotIn(group['anchor'],excluded)
            for case in group['cases']:
                g=replay(case['prefix']);self.assertEqual(key(g.board,g.player),case['decision_id']);self.assertNotIn(orbit(case['decision_id']),excluded);identities.append(orbit(case['decision_id']))
                self.assertNotIn(orbit(mirror(rotate_swap(case['decision_id']))),excluded)
        self.assertEqual(len(identities),len(set(identities)));self.assertEqual({g['own_move'] for g in rows},{7,8,9,10})

if __name__=='__main__':unittest.main()
