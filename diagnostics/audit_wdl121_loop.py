"""WDL闭环独立重放：不重复自弈或训练。"""
import argparse,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch,math
from az121 import encode,action_id
import rules121 as r
from curriculum121 import replay
from opening121_split_audit import orbit,decision_key

def identity(g):return orbit(decision_key(g.board,g.player))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit_games(games,schedule,history):
    assert len(schedule)==56 and len(games)==48;used=set(history);roots={};steps=0;labels=Counter();reasons=Counter();trajectory=defaultdict(set);states=[];owners=defaultdict(set)
    for i,row in enumerate(schedule):
        assert row['group']==i and row['split']==('train' if i<32 else 'development' if i<48 else 'probe') and row['ply']==(8,9,16,17)[i%4]
        g=replay(row['opening']);assert (g.board,g.player,g.ply)==(row['board'],row['player'],row['ply']);anchor=replay(row['opening'][:8]);ids={identity(g),identity(anchor)};assert not ids&used;used.update(ids);assert identity(g)==row['orbit'] and identity(anchor)==row['anchor'];roots[i]=g
    for game,row in zip(games,schedule):
        group=row['group'];assert game['group']==group and game['opening']==row['opening'] and game['start_ply']==row['ply'];g=replay(game['opening']);assert len(game['paths'])==len(game['search_info']);rng=np.random.default_rng(111122+group)
        for offset,(path,record) in enumerate(zip(game['paths'],game['search_info'])):
            assert g.result()[0] is None and record['ply']==g.ply and record['player']==g.player and record['path']==path and record['noise'] is True
            info=record['info'];legal=[p for _,p in r.get_all_moves(g.board,g.player)];assert info['paths']==legal and path in legal
            visits=np.array(info['visits']);policy=np.array(record['visit_target']);assert len(policy)==len(legal) and np.isfinite(policy).all() and min(policy)>=0 and np.isclose(policy.sum(),1);assert sum(visits)==info['nodes']==info['root_visits']==info['child_visits']==32;assert np.allclose(policy,visits/32) and np.allclose(policy,info['policy'])
            assert record['temperature']==(1 if offset<12 else 0);selection=np.array(record['selection_probabilities']);assert np.isclose(selection.sum(),1) and min(selection)>=0
            rng.dirichlet(np.full(len(legal),.3))
            if offset<12:assert np.allclose(selection,policy) and policy[legal.index(path)]>0 and legal[int(rng.choice(len(legal),p=policy))]==path
            else:assert legal[int(policy.argmax())]==path and selection[legal.index(path)]==1
            predictions=np.array(record['wdl_probabilities']);assert np.allclose(predictions,[1/3]*3,atol=1e-7) and predictions.shape==(3,) and np.isfinite(predictions).all() and min(predictions)>=0 and np.isclose(predictions.sum(),1)
            label=None if game['reason']=='max_ply' else 1 if game['winner']==0 else 0 if game['winner']==g.player else 2
            oid=identity(g);owners[oid].add(row['split']);states.append((oid,row['split'],label));trajectory[group].add(oid);labels[str(label)]+=1;steps+=1;g.play(path)
        assert g.ply==game['plies'] and g.result()==(game['winner'],game['reason']);reasons[game['reason']]+=1
    cross={oid for oid,parts in owners.items() if len(parts)>1};retained=Counter();excluded=Counter()
    for oid,part,label in states:
        if oid in cross:excluded['cross_split_samples']+=1
        if oid in history:excluded['historical_samples']+=1
        if oid in cross or oid in history:excluded['excluded_samples']+=1
        else:retained[part]+=1
    probe_hits={str(i):sorted(group for group,orbits in trajectory.items() if identity(roots[i]) in orbits) for i in range(48,56)}
    return dict(roots=56,selfplay_games=48,replayed_steps=steps,terminal_reasons=dict(reasons),raw_class_counts=dict(labels),cross_split_orbits=len(cross),retained={part:retained[part] for part in ('train','development')},excluded=dict(excluded),probe_root_trajectory_hits=probe_hits)

def audit_training(e,schedule,games,history,out):
    load=lambda name:json.loads((e/name).read_text());cfg=load('config.json');freeze=load('dataset_freeze.json');pools=torch.load(e/'samples.pt',weights_only=True)
    assert digest(e/'schedule.json')==freeze['schedule_sha256'] and digest(e/'selfplay_games.jsonl')==freeze['games_sha256'] and digest(e/'samples.pt')==freeze['samples_sha256']
    samples=[s for ss in pools.values() for s in ss];lookup={(s['group'],s['ply']):s for s in samples};assert len(lookup)==len(samples)
    for game in games:
        g=replay(game['opening'])
        for path,record in zip(game['paths'],game['search_info']):
            sample=lookup.get((game['group'],g.ply))
            if sample is not None:
                expected=None if game['reason']=='max_ply' else 1 if game['winner']==0 else 0 if game['winner']==g.player else 2
                assert sample['wdl_class']==expected and sample['player']==g.player and sample['terminal_ply']==game['plies'];assert np.array_equal(sample['x'].numpy(),encode(g.board,g.player,g.ply,g.repetitions,canonical=True));assert sample['actions']==[action_id(p[0],p[-1],g.player,True) for p in record['info']['paths']];assert np.allclose(sample['policy'],record['visit_target']);assert identity(g)==sample['orbit']
            g.play(path)
    assert {k:len(v) for k,v in pools.items()}==out['retained'];data=load('data_audit.json');assert data['raw_samples']==out['replayed_steps'] and data['cross_split_orbits']==out['cross_split_orbits']
    hashes=0
    drift=[]
    for name,sha in cfg['source_sha256'].items():
        assert digest(e/'source'/name)==sha;hashes+=1
        if digest(name)!=sha:drift.append(name)
    assert set(drift)<=set(('gui121.py','tests/test_wdl121_display.py')),drift
    out['post_execution_source_drift']=drift
    for field in ('native_sha256','old_data_sha256'):
        for name,sha in cfg[field].items():assert digest(name)==sha;hashes+=1
    assert digest(cfg['model'])==cfg['model_sha256'] and digest('nn_checkpoints/models/champions/neural121.json')==cfg['registry_sha256'];out['hash_entries']=hashes;out['model_registry_unchanged']=True
    result=load('result.json');assert not result['promoted'];out['training_started']=result['training_started']
    if not result['training_started']:return
    from cap_opening121_replay import load_data
    _,old,normal,_=load_data(Path('nn_checkpoints/experiments/fresh121_projection_20261105/data_sources'));groups=defaultdict(list)
    for s in pools['train']:groups[s['group']].append(s)
    ids=sorted(groups);rr=np.random.default_rng(111123);ro=np.random.default_rng(111124);trace=load('sample_trace.json');assert len(trace)==2000
    for actual in trace:
        selected=[ids[int(i)] for i in rr.integers(len(ids),size=16)];indices=[int(rr.integers(len(groups[g]))) for g in selected];oi=ro.integers(len(old['train']),size=8).tolist();ni=ro.integers(len(normal['train']),size=8).tolist();assert actual==dict(groups=selected,samples=indices,old=oi,normal=ni)
    parent=torch.load(e/'initial_wdl.pt',weights_only=True)['state_dict'];v3=torch.load(cfg['model'],weights_only=True)['state_dict'];assert all(torch.equal(parent[k],v) for k,v in v3.items());assert not torch.count_nonzero(parent['wdl.2.weight']) and not torch.count_nonzero(parent['wdl.2.bias']);assert digest(e/'candidate_step2000.pt')==result['checkpoint_sha256'];valuekeys=[k for k in parent if k.startswith('value.')]
    for step in (0,1000,2000):
        state=torch.load(e/f'candidate_step{step}.pt',weights_only=True)['state_dict'];assert all(torch.equal(state[k],parent[k]) for k in valuekeys)
        if step==0:assert all(torch.equal(state[k],parent[k]) for k in state)
    baseline=load('baseline.json');final=load('final.json')
    prior={p:np.ones(3) for p in (1,2)}
    for ss in groups.values():
        for sample in ss:
            if sample['wdl_class'] is not None:prior[sample['player']][sample['wdl_class']]+=1/len(ss)
    for player,vector in prior.items():assert np.allclose(vector/vector.sum(),baseline['train_player_prior'][str(player)])
    dev_lookup={(str(s['game_id']),s['ply']):s for s in pools['development']}
    for archive in (baseline['wdl'],baseline['prior_metrics'],final['wdl']):
        game_metrics=defaultdict(list);subsets=defaultdict(list)
        for row in archive['records']:
            sample=dev_lookup[(row['game_id'],row['ply'])];assert sample['wdl_class']==row['label'] and sample['player']==row['player'];assert row['late']==(0<=sample['terminal_ply']-sample['ply']<=16)
            p=np.array(row['probabilities']);label=row['label'];assert p.shape==(3,) and min(p)>=0 and np.isclose(p.sum(),1);brier=float(((p-np.eye(3)[label])**2).sum());nll=-math.log(max(p[label],1e-300));game_metrics[row['game_id']].append((brier,nll));subsets[str(row['player'])].append(brier)
            if row['late']:subsets['late'].append(brier)
        for key,ss in subsets.items():
            summary=archive['late'] if key=='late' else archive['by_player'][key];assert summary['n']==len(ss) and np.isclose(summary['brier3'],np.mean(ss))
        means=[np.mean(v,axis=0) for v in game_metrics.values()];macro=archive['game_macro'];assert macro['n_games']==len(means);assert np.isclose(macro['brier3'],np.mean([v[0] for v in means])) and np.isclose(macro['nll'],np.mean([v[1] for v in means]))
    decisive={}
    for name,archive in [('candidate',final['wdl']),('prior',baseline['prior_metrics'])]:
        values=[];conditional=[];points=[];pergame=defaultdict(list)
        for row in archive['records']:
            if row['label']==1:continue
            p=np.asarray(row['probabilities']);q=float(p[0]/(p[0]+p[2]));target=float(row['label']==0);values.append((q-target)**2);conditional.append(q);points.append((p[0]+.5*p[1]-target)**2);pergame[row['game_id']].append(points[-1])
        decisive[name]=dict(n=len(values),binary_brier=float(np.mean(values)),conditional_pwin_min=min(conditional),conditional_pwin_max=max(conditional),expectedpoint_mse_sample=float(np.mean(points)),expectedpoint_mse_game_macro=float(np.mean([np.mean(v) for v in pergame.values()])))
    out['decisive_diagnostic']=decisive
    out['limitations']=['正式开发门槛通过主要受降低和棋概率影响；候选二类胜负Brier劣于训练先验，不能证明学会胜负区分。','开发16局没有真实和棋样本，不具备和棋校准验收覆盖；不晋升。']
    f,b,p=final['wdl'],baseline['wdl'],baseline['prior_metrics'];fm,bm,pm=[a['game_macro'] for a in (f,b,p)]
    checks=dict(development_coverage=fm['n_games']>=8,nll_uniform=fm['nll']<=bm['nll']-.02,nll_prior=fm['nll']<=pm['nll']-.02,brier_uniform=fm['brier3']<=bm['brier3']-.02,brier_prior=fm['brier3']<=pm['brier3'],new_policy=final['new_policy_ce']<=baseline['new_policy_ce']+.05,old_policy=final['old_policy']['old']<=baseline['old_policy']['old']+.05,normal_policy=final['old_policy']['normal']<=baseline['old_policy']['normal']+.05,value_parameters_unchanged=True)
    for player in ('1','2'):checks['player'+player+'_brier']=f['by_player'][player]['n']>0 and f['by_player'][player]['brier3']<=b['by_player'][player]['brier3']
    checks['late_brier']=f['late']['n']==0 or f['late']['brier3']<=b['late']['brier3'];assert checks==final['checks']==result['checks'];assert all(checks.values())==result['development_pass'];assert not result['calibrated']
    probes=[json.loads(line) for line in (e/'probe_games.jsonl').read_text().splitlines()];assert len(probes)==32 and {(g['group'],g['nn_side'],g['opponent']) for g in probes}=={(i,p,o) for i in range(48,56) for p in (1,2) for o in ('WDL0','v3')};counts={}
    train_orbits={s['orbit'] for s in pools['train']}
    for game in probes:
        row=schedule[game['group']];assert game['opening']==row['opening'];assert game['input_isolated']==(not ({row['orbit'],row['anchor']}&train_orbits));g=replay(game['opening']);assert len(game['paths'])==len(game['search_info'])
        for path,record in zip(game['paths'],game['search_info']):
            assert record['ply']==g.ply and record['player']==g.player and g.result()[0] is None;info=record['info'];assert info['nodes']==info['root_visits']==info['child_visits']==sum(info['visits'])==32;assert info['paths'][int(np.argmax(info['visits']))]==path;g.play(path)
        assert g.ply==game['plies'] and g.result()==(game['winner'],game['reason'])
    for opponent in ('WDL0','v3'):
        ss=[g for g in probes if g['opponent']==opponent];counts[opponent]=dict(games=len(ss),wins=sum(g['winner']==g['nn_side'] for g in ss),losses=sum(g['winner'] not in (0,g['nn_side']) for g in ss),true_draws=sum(g['winner']==0 and g['reason']!='max_ply' for g in ss),truncated=sum(g['reason']=='max_ply' for g in ss))
    assert counts==result['probe_counts'];out.update(initial_selfplay_predictions_uniform=True,initial_policy_scalar_exact_v3=True,trace_exact=True,updates=2000,value_parameters_frozen=True,development_checks=checks,probe_counts=counts)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',type=Path,required=True);args=ap.parse_args();e=args.experiment;load=lambda name:json.loads((e/name).read_text())
    schedule=load('schedule.json');games=[json.loads(line) for line in (e/'selfplay_games.jsonl').read_text().splitlines()];history=load('history_exclusions.json');history=set(history['orbits']);out=audit_games(games,schedule,history)
    audit_training(e,schedule,games,history,out)
    out.update(roadmap_step=3,passed=True,scope='全部56根及48自弈逐手重放、概率与真实标签/隔离独立复核；未重复搜索或训练。')
    (e/'independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
