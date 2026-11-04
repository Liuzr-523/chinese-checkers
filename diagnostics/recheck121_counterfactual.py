"""One bounded deterministic depth-4 repeat of changed outcomes plus same-root controls.
No training. Forced root choices remain the original recorded choices.
"""
import json,sys,time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from curriculum121 import replay,finish,atomic
from engine_clysto121 import ClystoDerived121
from cycle121_targeted import verified_better
ROOT=Path('nn_checkpoints/experiments/targeted121_complete_cycle_20261030')
OUT=Path('nn_checkpoints/experiments/recheck121_counterfactual_20261031')
class FixedExpert(ClystoDerived121):
    def __init__(self):super().__init__(seconds=10.,max_depth=4)
    def think(self,*args,**kw):
        path,info=super().think(*args,**kw)
        if info['aborted']:raise ValueError('fixed-depth probe hit timeout; do not pretend deterministic')
        return path,info

def main():
    began=time.monotonic();OUT.mkdir(exist_ok=False)
    seeds=json.loads((ROOT/'seeds.json').read_text());records=[json.loads(s) for s in (ROOT/'focused_games.jsonl').read_text().splitlines()];pairs=defaultdict(dict)
    for row in records:pairs[row['seed_id']][row['role']]=row
    selected=[];controls=[]
    for i,seed in enumerate(seeds):
        a,b=pairs[i]['expert'],pairs[i]['played'];first=a['paths'][0]
        if a['winner']!=b['winner']:selected.append(i)
        elif (first[0],first[-1])==(seed['played_path'][0],seed['played_path'][-1]):controls.append(i)
    selected+=controls[:2];atomic(OUT/'config.json',dict(seed_ids=selected,changed_outcomes=len(selected)-min(2,len(controls)),same_root_controls=controls[:2],depth=4,repeats=2,maximum_games=4*len(selected),scope='Fixed forced root moves, new constant-depth continuation policy. Diagnostic robustness only; not a new label of optimal play.'))
    teacher=FixedExpert();results=[]
    for offset,i in enumerate(selected):
        seed=seeds[i];p=seed['player'];original=pairs[i];repeats=[]
        for rep in range(2):
            branches={}
            for role,forced in [('expert',original['expert']['paths'][0]),('played',seed['played_path'])]:
                atomic(OUT/'status.json',dict(stage='fixed_depth_recheck',seed=offset,total=len(selected),repeat=rep,role=role,elapsed=time.monotonic()-began))
                if time.monotonic()-began>600:raise TimeoutError('ten-minute diagnosis budget exhausted')
                prefix=seed['opening']+[forced];row=finish(replay(prefix),{1:teacher,2:teacher});row.update(opening=prefix,seed_id=i,repeat=rep,role=role)
                branches[role]=row
                with (OUT/'games.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            repeats.append(branches)
        for role in ('expert','played'):
            if repeats[0][role]['winner']!=repeats[1][role]['winner'] or repeats[0][role]['paths']!=repeats[1][role]['paths']:raise ValueError('fixed-depth repeated trajectories differed')
        def difference(a,b):
            def points(row):return .5 if row['winner']==0 else float(row['winner']==p)
            return points(a)-points(b)
        prior=difference(original['expert'],original['played']);current=difference(repeats[0]['expert'],repeats[0]['played'])
        results.append(dict(seed_id=i,player=p,prior_point_difference=prior,fixed_point_difference=current,
                            original_expert_winner=original['expert']['winner'],original_played_winner=original['played']['winner'],
                            fixed_expert_winner=repeats[0]['expert']['winner'],fixed_played_winner=repeats[0]['played']['winner'],repeated_paths_identical=True))
    result=dict(seeds=len(selected),games=len(selected)*4,prior_positive=sum(x['prior_point_difference']>0 for x in results),
                fixed_positive=sum(x['fixed_point_difference']>0 for x in results),fixed_negative=sum(x['fixed_point_difference']<0 for x in results),
                sign_unchanged=sum((x['prior_point_difference']>0)-(x['prior_point_difference']<0)==(x['fixed_point_difference']>0)-(x['fixed_point_difference']<0) for x in results),
                details=results,seconds=time.monotonic()-began,note='Changing continuation policy/depth changes outcomes; this cannot alone isolate timing noise in prior run or prove an optimal move.')
    atomic(OUT/'result.json',result);atomic(OUT/'status.json',dict(stage='complete',elapsed=time.monotonic()-began));print(json.dumps(result))
if __name__=='__main__':main()
