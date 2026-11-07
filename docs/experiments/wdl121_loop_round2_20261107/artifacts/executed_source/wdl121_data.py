"""固定模型完整自弈与真实终局WDL标签；前缀仅溯源，几何轨道只用于隔离。"""
import random
from collections import Counter,defaultdict
import numpy as np
import torch
import rules121 as r
from az121 import search,encode,action_id
from curriculum121 import replay
from opening121_split_audit import orbit,decision_key

def identity(g):return orbit(decision_key(g.board,g.player))
def make_schedule(model,history_orbits,count=56,train=32,development=16,seed=111121,max_attempts=128,status=lambda *a,**k:None):
    if not 0<train<train+development<=count:raise ValueError('invalid split')
    used=set(history_orbits);schedule=[]
    for group in range(count):
        target=(8,9,16,17)[group%4]
        for attempt in range(max_attempts):
            status('冻结WDL开局',group=group,attempt=attempt);rng=random.Random(seed+group*100003+attempt*10000019);g=r.Game();opening=[];anchor=None
            while g.ply<target and g.result()[0] is None:
                path=rng.choice(r.get_all_moves(g.board,g.player))[1];g.play(path);opening.append(path)
                if g.ply==8:anchor=identity(g)
            if g.ply!=target or g.result()[0] is not None:continue
            rid=identity(g);ids={rid,anchor}
            if ids&used:continue
            row=dict(group=group,split='train' if group<train else 'development' if group<train+development else 'probe',opening=opening,board=g.board,player=g.player,ply=g.ply,orbit=rid,anchor=anchor,attempt=attempt);schedule.append(row);used.update(ids);break
        else:raise ValueError(f'组{group}候选耗尽，不补根或改阈值')
    return schedule

@torch.no_grad()
def _probabilities(model,g):
    x=torch.from_numpy(encode(g.board,g.player,g.ply,g.repetitions,canonical=model.canonical)).unsqueeze(0)
    probabilities=model.predict_wdl(x)[0].detach().cpu().numpy().tolist()
    if len(probabilities)!=3 or not np.isfinite(probabilities).all() or min(probabilities)<0 or not np.isclose(sum(probabilities),1):raise ValueError('invalid WDL probabilities')
    return probabilities

def run_selfplay(model,row,simulations=32,seed=None,status=lambda *a,**k:None):
    model.eval();g=replay(row['opening']);assert (g.board,g.player,g.ply)==(row['board'],row['player'],row['ply']);rng=np.random.default_rng(111122+row['group'] if seed is None else seed);paths=[];infos=[]
    while g.result()[0] is None:
        status('固定模型WDL自弈',group=row['group'],ply=g.ply);prediction=_probabilities(model,g)
        _,info=search(model,g.board,g.player,g.ply,g.repetitions,simulations=simulations,rng=rng,noise=True)
        probabilities=np.asarray(info['policy'],dtype=float)
        if len(probabilities)!=len(info['paths']) or not np.isfinite(probabilities).all() or min(probabilities)<0 or not np.isclose(probabilities.sum(),1):raise ValueError('invalid visit policy')
        stochastic=len(paths)<12;index=int(rng.choice(len(probabilities),p=probabilities)) if stochastic else int(probabilities.argmax());path=info['paths'][index]
        infos.append(dict(player=g.player,ply=g.ply,path=path,noise=True,temperature=1 if stochastic else 0,selection_probabilities=probabilities.tolist() if stochastic else [float(i==index) for i in range(len(probabilities))],wdl_probabilities=prediction,visit_target=probabilities.tolist(),info=info));paths.append(path);g.play(path)
    winner,reason=g.result();return dict(group=row['group'],game_id=row['group'],split=row['split'],opening=row['opening'],start_ply=row['ply'],paths=paths,search_info=infos,winner=winner,reason=reason,plies=g.ply,seed=111122+row['group'] if seed is None else seed)

def build_samples(games,schedule,history_orbits):
    from wdl121 import terminal_class
    indexed={row['group']:row for row in schedule if row['split']!='probe'};assert len(indexed)==len(games);samples=[];seen=set();owners=defaultdict(set)
    for game in games:
        group=game['group'];assert group not in seen;seen.add(group);row=indexed[group];assert game['opening']==row['opening'];g=replay(game['opening']);assert g.ply==game['start_ply']==row['ply'];assert len(game['paths'])==len(game['search_info'])
        for path,record in zip(game['paths'],game['search_info']):
            assert g.result()[0] is None and record['player']==g.player and record['ply']==g.ply and record['path']==path
            info=record['info'];legal=[p for _,p in r.get_all_moves(g.board,g.player)];assert info['paths']==legal and path in legal
            policy=np.asarray(record['visit_target'],dtype=float);assert len(policy)==len(legal) and np.isfinite(policy).all() and min(policy)>=0 and np.isclose(policy.sum(),1);assert np.allclose(policy,info['policy']);assert sum(info['visits'])==info['root_visits']==info['child_visits']==info['nodes'];assert np.allclose(policy,np.asarray(info['visits'])/sum(info['visits']))
            oid=identity(g);owners[oid].add(row['split']);sample=dict(x=encode(g.board,g.player,g.ply,g.repetitions,canonical=True),actions=[action_id(p[0],p[-1],g.player,True) for p in legal],policy=policy.tolist(),player=g.player,game_id=group,group=group,ply=g.ply,split=row['split'],orbit=oid,winner=game['winner'],reason=game['reason'],terminal_ply=game['plies'],wdl_class=terminal_class(game['winner'],g.player,game['reason']));samples.append(sample);g.play(path)
        assert g.ply==game['plies'] and g.result()==(game['winner'],game['reason'])
    cross={oid for oid,parts in owners.items() if len(parts)>1};historical=set(history_orbits);result={part:[] for part in ('train','development')};counts=Counter();labels=Counter()
    for sample in samples:
        cross_hit=sample['orbit'] in cross;history_hit=sample['orbit'] in historical;counts['cross_split_samples']+=cross_hit;counts['historical_samples']+=history_hit
        if cross_hit or history_hit:counts['excluded_samples']+=1;continue
        result[sample['split']].append(sample);labels[str(sample['wdl_class'])]+=1
    audit=dict(raw_samples=len(samples),retained={part:len(rows) for part,rows in result.items()},cross_split_orbits=len(cross),counts=dict(counts),class_counts=dict(labels),prefix_samples=0,scope='按整盘固定split；跨组保守轨道冲突双方样本均排除，再排历史输入；截断wdl_class=None，无旋转换色标签增强。')
    return result,audit
