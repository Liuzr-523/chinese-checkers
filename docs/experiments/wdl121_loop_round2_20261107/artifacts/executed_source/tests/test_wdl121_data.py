import copy,unittest
from unittest.mock import patch
import numpy as np
import torch
import rules121 as r
import wdl121_data as d
from wdl121 import terminal_class
class Dummy:
    canonical=True
    def eval(self):return self
    def predict_wdl(self,x):return torch.tensor([[.2,.3,.5]]*len(x))
def fixture(group=0,split='train',truncated=False):
    if truncated:g=r.Game();g.ply=239;path=r.get_all_moves(g.board,g.player)[0][1]
    else:
        b=[0]*121
        for i in range(9):b[i]=2
        outside=next(i for i in r.NEIGHBORS[9] if i not in r.GOAL[2]);b[outside]=2
        for i in [i for i in range(30,100) if not b[i]][:10]:b[i]=1
        g=r.Game(board=b,player=2,ply=239);path=next(p for _,p in r.get_all_moves(b,2) if p[0]==outside and p[-1]==9)
    legal=[p for _,p in r.get_all_moves(g.board,g.player)];index=legal.index(path);policy=[float(i==index) for i in range(len(legal))];info=dict(paths=legal,visits=policy,policy=policy,root_visits=1,child_visits=1,nodes=1)
    row=dict(group=group,split=split,opening=[],board=g.board,player=g.player,ply=g.ply)
    end=copy.deepcopy(g);end.play(path);winner,reason=end.result();game=dict(group=group,split=split,opening=[],start_ply=239,paths=[path],search_info=[dict(player=g.player,ply=g.ply,path=path,visit_target=policy,info=info)],winner=winner,reason=reason,plies=240)
    return g,row,game,info
class WDLDataTests(unittest.TestCase):
    def test_schedule_fixed_prefix_and_probe(self):
        schedule=d.make_schedule(Dummy(),set(),count=56)
        self.assertEqual([sum(x['split']==s for x in schedule) for s in ('train','development','probe')],[32,16,8])
        self.assertEqual([x['ply'] for x in schedule[:4]],[8,9,16,17])
        self.assertEqual(schedule,d.make_schedule(Dummy(),set(),count=56))
        for row in schedule:
            g=d.replay(row['opening']);self.assertEqual(g.board,row['board']);self.assertEqual(d.identity(g),row['orbit'])
        first=schedule[0]['orbit'];other=d.make_schedule(Dummy(),{first},count=56);self.assertNotEqual(other[0]['orbit'],first)
    def test_terminal_classes_and_truncation(self):
        self.assertEqual(terminal_class(1,1,'all_pieces_in_goal'),0);self.assertEqual(terminal_class(1,2,'all_pieces_in_goal'),2);self.assertEqual(terminal_class(0,1,'equal_turn_goal_draw'),1);self.assertIsNone(terminal_class(0,1,'max_ply'))
        for truncated in (False,True):
            g,row,game,_=fixture(truncated=truncated)
            with patch.object(d,'replay',side_effect=lambda _:copy.deepcopy(g)):samples,audit=d.build_samples([game],[row],set())
            self.assertEqual(len(samples['train']),1);sample=samples['train'][0];self.assertEqual(sample['ply'],239);self.assertEqual(sample['wdl_class'],None if truncated else 0);self.assertEqual(audit['prefix_samples'],0)
    def test_crosssplit_and_history_exclude_both(self):
        g,a,ga,_=fixture();_,b,gb,_=fixture(1,'development')
        with patch.object(d,'replay',side_effect=lambda _:copy.deepcopy(g)):samples,audit=d.build_samples([ga,gb],[a,b],set())
        self.assertEqual(samples,dict(train=[],development=[]));self.assertEqual(audit['counts']['cross_split_samples'],2)
        with patch.object(d,'replay',side_effect=lambda _:copy.deepcopy(g)):samples,audit=d.build_samples([ga],[a],{d.identity(g)})
        self.assertFalse(samples['train']);self.assertEqual(audit['counts']['historical_samples'],1)
    def test_corrupt_policy_and_path_rejected(self):
        g,row,game,_=fixture();game['search_info'][0]['visit_target']=[0.]*len(game['search_info'][0]['visit_target'])
        with patch.object(d,'replay',side_effect=lambda _:copy.deepcopy(g)),self.assertRaises(AssertionError):d.build_samples([game],[row],set())
        _,row,game,_=fixture();game['paths'][0]=[0,120]
        with patch.object(d,'replay',side_effect=lambda _:copy.deepcopy(g)),self.assertRaises(AssertionError):d.build_samples([game],[row],set())
    def test_selfplay_records_noise_and_probabilities(self):
        g,row,game,info=fixture()
        with patch.object(d,'replay',side_effect=lambda _:copy.deepcopy(g)),patch.object(d,'search',return_value=(game['paths'][0],info)) as search:
            actual=d.run_selfplay(Dummy(),row,simulations=1)
        record=actual['search_info'][0];self.assertTrue(record['noise']);self.assertEqual(record['temperature'],1);np.testing.assert_allclose(record['wdl_probabilities'],[.2,.3,.5]);self.assertEqual(actual['winner'],2);self.assertTrue(search.call_args.kwargs['noise'])
if __name__=='__main__':unittest.main()
