import copy,hashlib,json,tempfile,unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch
import rules121 as r
from wdl121_display import WDLAnalysis,ModelPredictor,format_probabilities,position_key,terminal_text,top_candidates

class ManualExecutor:
    def __init__(self):self.jobs=[]
    def submit(self,fn,*args):
        f=Future();self.jobs.append((f,fn,args));return f
    def shutdown(self,**kwargs):pass

class WDLDisplayTests(unittest.TestCase):
    def test_player_two_mapping_and_invalid_probabilities(self):
        text=format_probabilities([.6,.1,.3],2)
        self.assertIn('P1胜 30.0%',text);self.assertIn('P2胜 60.0%',text);self.assertIn('未校准',text)
        for probabilities in ([.7,.7,-.4],[float('nan'),0,1],[0,0,0]):
            with self.assertRaises(ValueError):format_probabilities(probabilities,1)

    def test_missing_checkpoint_and_bad_hash_do_not_fabricate_winrate(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict('os.environ',{},clear=True):
            pointer=Path(folder)/'pointer.json';predict=ModelPredictor(pointer)
            self.assertEqual(predict(r.init_board(),1,0,{}),'胜率模型尚未生成')
            checkpoint=Path(folder)/'model.pt';checkpoint.write_bytes(b'bad')
            pointer.write_text(json.dumps(dict(checkpoint=str(checkpoint),sha256='wrong')))
            with self.assertRaises(ValueError):predict(r.init_board(),1,0,{})

    def test_real_wdl_checkpoint_is_loaded_as_a_separate_analysis_model(self):
        import torch
        from az121 import Net121
        from wdl121 import WDLNet121
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder,patch.dict('os.environ',{},clear=True):
            path=Path(folder)/'candidate.pt';base=Net121(width=4,canonical=True)
            WDLNet121.from_v3(base).save(path)
            pointer=Path(folder)/'display.json';pointer.write_text(json.dumps(dict(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),role='experimental',calibrated=False)))
            predictor=ModelPredictor(pointer);result=predictor(r.init_board(),2,0,{});text=result['text']
            self.assertEqual(len(result['candidates']),5)
            self.assertEqual(result['simulations'],128)
            self.assertIn('P1胜 33.3%',text);self.assertIn('未校准',text)
            self.assertIsNot(predictor.model,base)
            legacy=Path(folder)/'legacy.pt';base.save(legacy)
            with patch.dict('os.environ',{'CC121_WDL_MODEL':str(legacy)}):
                with self.assertRaises(ValueError):predictor(r.init_board(),1,0,{})

    def test_top_candidates_use_root_win_probability_and_keep_distinct_paths(self):
        info=dict(paths=[[1,4],[2,4],[3,5],[4,6]],visits=[3,7,2,0],child_wdl=[[.2,.6,.2],[.6,.1,.3],None,[1,0,0]],child_wdl_visits=[3,6,0,0])
        cs=top_candidates(info)
        self.assertEqual(cs[0]['path'],[2,4]);self.assertEqual(cs[1]['path'],[1,4])
        self.assertEqual(cs[0]['wdl'][0],.6);self.assertTrue(cs[0]['partial'])
        self.assertIsNone(cs[2]['wdl']);self.assertIsNone(cs[3]['wdl'])
        with self.assertRaises(ValueError):top_candidates(info,2)

    def test_board_percentage_badges_distinguish_same_endpoint_and_use_pwin(self):
        import gui121
        candidates=[dict(rank=i+1,path=[i+1,16],visits=10,wdl=[.6,.1,.3]) for i in range(5)]
        badges=gui121.candidate_badges(candidates)
        self.assertEqual(len(badges),5)
        self.assertEqual([b['label'] for b in badges],[f'{i+1} · 60.0%' for i in range(5)])
        for i,b in enumerate(badges):
            self.assertTrue(gui121.pygame.Rect(8,112,854,718).contains(b['rect']))
            for other in badges[:i]:self.assertFalse(b['rect'].colliderect(other['rect']))
        self.assertEqual(len(gui121.candidate_badges(candidates,selected=2)),1)
        candidates[0]['wdl']=None
        self.assertIn('未估计',gui121.candidate_badges(candidates)[0]['label'])

    def test_stale_move_undo_generation_and_history_are_rejected(self):
        executor=ManualExecutor();a=WDLAnalysis(executor=executor);g=r.Game()
        a.update(g,0);old=executor.jobs[0][0]
        g.play(r.get_all_moves(g.board,g.player)[0][1]);a.update(g,0)
        old.set_result('旧局面预测不得显示')
        self.assertNotIn('旧局面',a.update(g,0));self.assertEqual(len(executor.jobs),2)
        executor.jobs[1][0].set_result('当前估计');self.assertEqual(a.update(g,0),'当前估计')
        g.undo();self.assertNotEqual(a.update(g,1),'当前估计')
        key=position_key(g,1);g.repetitions[(tuple(g.board),g.player)]+=1
        self.assertNotEqual(key,position_key(g,1))

    def test_one_request_per_position_animation_and_errors(self):
        executor=ManualExecutor();a=WDLAnalysis(executor=executor);g=r.Game()
        self.assertIn('动画',a.update(g,0,True));self.assertEqual(len(executor.jobs),0)
        a.update(g,0);a.update(g,0);self.assertEqual(len(executor.jobs),1)
        executor.jobs[0][0].set_exception(ValueError('坏模型'))
        self.assertIn('坏模型',a.update(g,0));a.update(g,0);self.assertEqual(len(executor.jobs),1)

    def test_terminal_rules_are_not_model_estimates_and_limit_is_not_true_draw(self):
        g=r.Game();g.ply=240
        self.assertIn('截断',terminal_text(g));self.assertNotIn('100%',terminal_text(g))
        g=r.Game();g.repetitions[(tuple(g.board),g.player)]=3
        executor=ManualExecutor();a=WDLAnalysis(executor=executor)
        self.assertEqual(a.update(g,0),'规则确定结果：和棋 100%');self.assertEqual(len(executor.jobs),0)
        b=[0]*121
        for i in r.GOAL[2]:b[i]=2
        for i in range(40,50):b[i]=1
        g=r.Game(board=b,player=1)
        self.assertEqual(terminal_text(g),'规则确定结果：P2胜 100%')

if __name__=='__main__':unittest.main()
