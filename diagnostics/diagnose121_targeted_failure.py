"""Audit targeted-cycle seed timing, factual branch contrast and training exposure.
No training, no reuse of arena outcomes to make training samples.
"""
import hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import rules121 as r
from az121 import Net121,Neural121,encode,action_id
from curriculum121 import replay,atomic
from cycle121_targeted import expert_samples,verified_better

RUN=Path('nn_checkpoints/experiments/targeted121_complete_cycle_20261030')
OUT=Path('nn_checkpoints/experiments/diagnose121_targeted_failure_20261031')

def rows(path):return [json.loads(s) for s in path.read_text().splitlines()]
def describe(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),minimum=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),maximum=float(a.max())) if len(a) else dict(n=0)

def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=False);cfg=json.loads((RUN/'config.json').read_text())
    for name,sha in cfg['source_sha256'].items():assert hashlib.sha256((RUN/'source'/name).read_bytes()).hexdigest()==sha
    for name,sha in cfg['data_sha256'].items():assert hashlib.sha256((RUN/'data_sources'/name).read_bytes()).hexdigest()==sha
    seeds=json.loads((RUN/'seeds.json').read_text());games=rows(RUN/'focused_games.jsonl');comparison=rows(RUN/'counterfactual_comparison.jsonl')
    source_games=rows(RUN/'data_sources/dev_games.jsonl');annotations=rows(RUN/'data_sources/dev_annotations.jsonl')
    by_seed=defaultdict(dict)
    for game in games:by_seed[game['seed_id']][game['role']]=game
    focus={'train':[],'validation':[]};verified={'train':[],'validation':[]};metadata=[];groups=Counter();same=0;changed_old=0
    timing=[];first_disagreements=[];outcomes=Counter();root_checks=[]
    for i,seed in enumerate(seeds):
        a=by_seed[i]['expert'];b=by_seed[i]['played'];p=seed['player'];part=seed['split'];origin=source_games[seed['game_id']]
        assert verified_better(a,b,p)==comparison[i]['verified_better'];is_better=comparison[i]['verified_better']
        expert_path=a['paths'][0];same_action=(expert_path[0],expert_path[-1])==(seed['played_path'][0],seed['played_path'][-1])
        old_changed=(expert_path[0],expert_path[-1])!=(seed['original_expert_path'][0],seed['original_expert_path'][-1])
        same+=same_action;changed_old+=old_changed
        def tag(game):return 'win' if game['winner']==p else 'draw' if game['winner']==0 else 'loss'
        outcomes['expert_'+tag(a)+'__played_'+tag(b)]+=1
        original_lost=origin['winner']==3-p
        if original_lost and a['winner']==b['winner']==3-p:groups['both_expert_continuations_still_lose_from_original_loss']+=1
        if original_lost and (a['winner']==p or b['winner']==p):groups['teacher_continuation_recovers_original_loss']+=1
        if same_action and is_better:groups['reported_better_despite_same_root_action']+=1
        timing.append(dict(seed_id=i,source_game=seed['game_id'],part=part,category=seed['category'],seed_ply=seed['ply'],source_end_ply=origin['plies'],
                           source_remaining=origin['plies']-seed['ply'],source_fraction=seed['ply']/origin['plies'],original_nn_lost=original_lost,
                           same_root_action=same_action,expert_budget_changed_action=old_changed,verified_better=is_better))
        for role,game in [('expert',a),('played',b)]:
            samples=expert_samples(game)
            for j,_ in enumerate(samples):
                meta=dict(seed_id=i,role=role,root=(role=='expert' and j==0),verified=is_better,root_action_distinct=not same_action)
                focus[part].append(meta)
                if is_better:verified[part].append(meta)
        root_checks.append(dict(seed=seed,expert_path=expert_path,verified=is_better,same=same_action))
    for game,ann in zip(source_games,annotations):
        if ann['split'] not in ('train','validation'):continue
        differing=[a['ply'] for a in ann['annotations'] if (a['played_path'][0],a['played_path'][-1])!=(a['expert_path'][0],a['expert_path'][-1])]
        if differing:first_disagreements.append(min(differing))
    declared=json.loads((RUN/'data.json').read_text())
    for part in focus:assert len(focus[part])==declared['focus'][part] and len(verified[part])==declared['verified_focus'][part]
    # Replay EXACT sampler schedule for each selected checkpoint. This is item exposure,
    # not gradient influence: samples can have different gradient magnitudes.
    exposures={}
    for arm in cfg['arms']:
        history=json.loads((RUN/(arm+'_training.json')).read_text());chosen=min(history,key=lambda h:h['validation']['criterion']);steps=chosen['step']
        rng=np.random.default_rng(cfg['seed']);count=Counter();per_root=Counter()
        pools=[verified['train'] or focus['train'],focus['train']]
        for _ in range(steps):
            for pool in pools:
                for index in rng.integers(len(pool),size=4):
                    sample=pool[int(index)];count['focused_items']+=1
                    if sample['verified']:count['verified_trajectory_items']+=1
                    if sample['root']:
                        count['expert_root_items']+=1;per_root[sample['seed_id']]+=1
                        if sample['verified']:count['verified_root_items']+=1
                        if sample['verified'] and sample['root_action_distinct']:count['verified_distinct_root_items']+=1
            rng.integers(declared['normal']['train'],size=4);rng.integers(declared['retention']['train'],size=4)
        total=steps*16
        exposures[arm]=dict(best_step=steps,total_training_items=total,counts=dict(count),
            verified_root_fraction=count['verified_root_items']/total,verified_distinct_root_fraction=count['verified_distinct_root_items']/total,
            expert_root_fraction=count['expert_root_items']/total,verified_trajectory_fraction=count['verified_trajectory_items']/total,
            verified_training_seeds=len({s['seed_id'] for s in verified['train']}),per_root=dict(per_root))
    models={'v3':Net121.load(cfg['model']),'candidate':Net121.load(RUN/'candidate.pt')};root_policy={}
    with torch.no_grad():
        for name,model in models.items():
            buckets=defaultdict(list)
            for item in root_checks:
                seed=item['seed'];g=replay(seed['opening']);teacher=item['expert_path']
                x=torch.from_numpy(encode(g.board,g.player,g.ply,g.repetitions,canonical=True)).unsqueeze(0)
                logits,_=model(x);moves=r.get_all_moves(g.board,g.player);ids=[action_id(s,p[-1],g.player,True) for s,p in moves]
                chosen=ids.index(action_id(teacher[0],teacher[-1],g.player,True));lp=logits[0,ids].log_softmax(0)
                outcome=(int(lp.argmax())==chosen,float(lp[chosen].exp()))
                buckets['all'].append(outcome);buckets[seed['split']].append(outcome)
                if item['verified'] and not item['same']:buckets['verified_distinct'].append(outcome)
            root_policy[name]={k:dict(n=len(v),agreement=sum(x[0] for x in v)/len(v),expert_probability=sum(x[1] for x in v)/len(v)) for k,v in buckets.items()}
    # Value labels are factual outcomes, but the first forced choice is not the
    # same continuation policy in both branches. Only expert moves are targets.
    result=dict(audit=dict(source_and_data_hashes_match=True,expert_games_replayed=len(games),training_sampler_reproduced=True),
                seeds=dict(total=len(seeds),selected_ply=describe([s['seed_ply'] for s in timing]),source_remaining=describe([s['source_remaining'] for s in timing]),
                           source_fraction=describe([s['source_fraction'] for s in timing]),first_development_disagreement_ply=describe(first_disagreements),
                           selected_nn_original_losses=sum(s['original_nn_lost'] for s in timing),same_actual_root_action=same,new_expert_differs_from_old_label=changed_old,
                           continuation_outcomes=dict(outcomes),continuation_interpretation=dict(groups),details=timing),
                exposure=exposures,root_policy=root_policy,scope='Development-data/label audit; no new training, no arena-score tuning. Exposure is not gradient attribution.')
    atomic(OUT/'result.json',result)
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    atomic(OUT/'provenance.json',dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_run=str(RUN),model_sha256={k:hashlib.sha256(Path(p).read_bytes()).hexdigest() for k,p in [('v3',cfg['model']),('candidate',RUN/'candidate.pt')]}))
    print(json.dumps({k:v for k,v in result.items() if k not in ('seeds','exposure')}))
    print(json.dumps(dict(seeds={k:v for k,v in result['seeds'].items() if k!='details'},exposure={k:{a:b for a,b in v.items() if a!='per_root'} for k,v in exposures.items()})))
if __name__=='__main__':main()
