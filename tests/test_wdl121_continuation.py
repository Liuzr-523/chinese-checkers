import tempfile,unittest
from pathlib import Path
import torch
from az121 import Net121
from wdl121 import WDLNet121
from train121_wdl_loop import initial_model,decisive_metrics,decisive_checks

class ContinuationTests(unittest.TestCase):
    def test_parent_state_inherited_exactly_without_reinitializing_head(self):
        parent=WDLNet121(width=8)
        with torch.no_grad():parent.wdl[-1].bias.copy_(torch.tensor([.9,-.7,.3]));parent.stem.weight.add_(.1)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'parent.pt';parent.save(path)
            child=initial_model(Net121(width=8),path,2)
            for key,value in parent.state_dict().items():self.assertTrue(torch.equal(value,child.state_dict()[key]),key)
            self.assertEqual(child.metadata['iteration'],1)
            self.assertTrue(all(not p.requires_grad for p in child.value.parameters()))

    def test_iteration_two_requires_parent(self):
        with self.assertRaises(ValueError):initial_model(Net121(width=8),iteration=2)
        with self.assertRaises(ValueError):initial_model(Net121(width=8),iteration=0)
        first=initial_model(Net121(width=8));self.assertEqual(float(first.wdl[-1].weight.abs().sum()),0.)

    def records(self,quality=.8,winners=(1,1,2,2)):
        rows=[]
        for game,winner in enumerate(winners):
            for player in (1,2):
                label=0 if winner==player else 2
                prob=[quality*.8,.2,(1-quality)*.8] if label==0 else [(1-quality)*.8,.2,quality*.8]
                rows.append(dict(game_id=str(game),player=player,label=label,probabilities=prob))
        return dict(records=rows)

    def test_conditional_binary_brier_is_game_macro(self):
        measured=decisive_metrics(self.records(.8))
        self.assertAlmostEqual(measured['overall']['game_macro_brier'],.04)
        self.assertEqual(measured['overall']['n_games'],4)
        for side in ('1','2'):self.assertAlmostEqual(measured['by_player'][side]['game_macro_brier'],.04)
        checks=decisive_checks(self.records(.8),self.records(.7),self.records(.6))
        self.assertTrue(all(checks.values()))

    def test_coverage_requires_each_actual_winner_twice_not_side_labels(self):
        for winners in ((1,1,1,1),(1,1,1,2),()):
            with self.subTest(winners=winners):
                metrics=self.records(.8,winners)
                checks=decisive_checks(metrics,metrics,metrics)
                self.assertFalse(checks['decisive_coverage']);self.assertFalse(checks['decisive_parent_gain005'])

    def test_no_gain_fails_even_with_full_coverage_and_draws_excluded(self):
        f=self.records(.8);f['records'].append(dict(game_id='draw',player=1,label=1,probabilities=[.2,.6,.2]))
        self.assertEqual(decisive_metrics(f)['overall']['n'],8)
        checks=decisive_checks(f,f,f);self.assertTrue(checks['decisive_coverage']);self.assertFalse(checks['decisive_parent_gain005'])

if __name__=='__main__':unittest.main()
