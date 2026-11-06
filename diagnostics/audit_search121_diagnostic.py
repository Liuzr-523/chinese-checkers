"""搜索诊断独立归档复核：只重算48次纯策略，不重复MCTS/实战。"""
import sys,json,math,hashlib
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
import rules121 as r
from az121 import Net121,encode,action_id
from curriculum121 import replay
E=Path('nn_checkpoints/experiments/search121_diagnostic_20261106')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def equal(a,b):
    if isinstance(a,dict):assert a.keys()==b.keys();[equal(a[k],b[k]) for k in a]
    elif isinstance(a,list):assert len(a)==len(b);[equal(x,y) for x,y in zip(a,b)]
    elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-7),(a,b)
    else:assert a==b,(a,b)
def match(a,b):return (a[0],a[-1])==(b[0],b[-1])
def quality(row,path):
    if not row['label']:return '无稳定标签'
    pref=row['rear_path'] if row['label']==1 else row['front_path']
    if match(path,pref):return '稳定偏好方'
    if any(match(path,row[k+'_path']) for k in ('rear','front')):return '稳定非偏好方'
    return '候选外未知'
def visit_check(info,n,board,player):
    paths=info['paths'];visits=info['visits'];policy=info['policy'];assert len(paths)==len(visits)==len(policy)==len(info['priors'])==len(info['child_q'])
    assert paths==[path for _,path in r.get_all_moves(board,player)];assert all(r.valid_path(board,player,p) for p in paths)
    assert len({(p[0],p[-1]) for p in paths})==len(paths)
    assert all(v>=0 and int(v)==v for v in visits);assert sum(visits)==info['nodes']==info['root_visits']==info['child_visits']==n
    assert info['root_visited_actions']==sum(v>0 for v in visits)
    equal(sum(policy),1.);equal(sum(info['priors']),1.)
    for v,p in zip(visits,policy):equal(p,v/n)
    assert 1<=info['mean_depth']<=info['max_depth']<=n
    assert all(-1.000001<=q<=1.000001 for q in info['child_q']);assert -1.000001<=info['root_q']<=1.000001
    return paths[int(np.argmax(visits))]
def main():
    torch.set_num_threads(1);load=lambda n:json.loads((E/n).read_text());lines=lambda n:[json.loads(x) for x in (E/n).read_text().splitlines()]
    cfg=load('config.json');assert load('status.json')['stage']=='complete';result=load('result.json');assert result['completed'] and not result['training_started'] and not result['promoted'];source=Path(next(iter(cfg['input_sha256']))).parent;rows=json.loads((source/'dataset.json').read_text())['rows'];schedule=json.loads((source/'schedule.json').read_text());assert len(rows)==len(schedule)==48 and sum(bool(x['label']) for x in rows)==11
    for row in rows:
        labels=[]
        for d in ('3','4'):
            a,b=[row['returns'][d][s] for s in ('rear','front')];p=row['player']
            if any(g['reason']=='max_ply' or g['winner'] is None for g in (a,b)):label=0
            else:
                points=[.5 if g['winner']==0 else float(g['winner']==p) for g in (a,b)]
                if points[0]!=points[1]:label=1 if points[0]>points[1] else -1
                elif a['winner']==b['winner']==p and abs(a['own_moves']-b['own_moves'])>=2:label=1 if a['own_moves']<b['own_moves'] else -1
                else:label=0
            labels.append(label)
        raw=labels[0] if labels[0] and labels[0]==labels[1] else 0
        assert raw==row['raw_label']
        assert row['label']==(0 if row['exclusion_reasons'] else raw)
    model=Net121.load(cfg['model']);model.requires_grad_(False);pure=lines('pure_policy.jsonl');cases=lines('cases.jsonl');assert len(pure)==48 and len(cases)==288;assert {(x['group'],x['simulations'],x['value_scale']) for x in cases}=={(g,n,s) for g in range(48) for n in (32,128,512) for s in (1.,0.)}
    roots={};logits_by={}
    for row,scheduled,base in zip(rows,schedule,pure):
        group=row['group'];g=replay(row['opening']);assert group==base['group'] and scheduled['opening']==row['opening'];assert (g.board,g.player,g.ply)==(row['board'],row['player'],row['ply']);roots[group]=g
        moves=r.get_all_moves(g.board,g.player);ids=[action_id(s,p[-1],g.player,model.canonical) for s,p in moves]
        with torch.no_grad():logits,v=model(torch.from_numpy(encode(g.board,g.player,g.ply,g.repetitions,canonical=model.canonical)).unsqueeze(0))
        best=moves[int(logits[0,ids].argmax())][1];assert match(best,base['pure_path']);equal(float(v[0]),base['root_value']);assert base['pure_quality']==quality(row,best);logits_by[group]=(logits,ids)
    for case in cases:
        group=case['group'];row=rows[group];g=roots[group];info=case['info'];assert case['pure_path']==pure[group]['pure_path'] and case['split']==row['split'] and case['player']==g.player and case['ply']==g.ply;chosen=visit_check(info,case['simulations'],g.board,g.player);assert match(chosen,case['path']);assert case['legal']==len(info['paths']) and case['label']==row['label'];assert case['quality']==quality(row,chosen);assert case['pure_quality']==quality(row,case['pure_path']);assert case['changed_from_policy']==(not match(chosen,case['pure_path']))
        ci=next(i for i,p in enumerate(info['paths']) if match(p,chosen));a,b=[next(i for i,p in enumerate(info['paths']) if match(p,row[k+'_path'])) for k in ('rear','front')];equal(case['pair_visits'],[info['visits'][a],info['visits'][b]]);equal(case['pair_q_root'],[-info['child_q'][a],-info['child_q'][b]])
        logits,ids=logits_by[group];assert case['chosen_policy_rank']==1+sum(float(logits[0,i])>float(logits[0,ids[ci]]) for i in ids)
    summaries=[]
    for n in (32,128,512):
        for scale in (1.,0.):
            for split in ('train','development','all'):
                ss=[x for x in cases if x['simulations']==n and x['value_scale']==scale and (split=='all' or x['split']==split)];eligible=[x for x in ss if x['label']];keys=('稳定偏好方','稳定非偏好方','候选外未知')
                summaries.append(dict(simulations=n,value_scale=scale,split=split,n=len(ss),changed_from_policy=sum(x['changed_from_policy'] for x in ss),eligible=len(eligible),quality={k:sum(x['quality']==k for x in eligible) for k in keys},pure_quality={k:sum(x['pure_quality']==k for x in eligible) for k in keys},mean_seconds=float(np.mean([x['elapsed'] for x in ss])),mean_coverage=float(np.mean([x['info']['root_visited_actions']/x['legal'] for x in ss])),mean_depth=float(np.mean([x['info']['mean_depth'] for x in ss])),max_depth=max(x['info']['max_depth'] for x in ss),pair_visit_correct=sum((x['pair_visits'][0]-x['pair_visits'][1])*x['label']>0 for x in eligible)))
    equal(summaries,load('case_summary.json'));equal(summaries,result['cases'])
    games=lines('games.jsonl');assert len(games)==12 and {(x['group'],x['nn_side']) for x in games}=={(g,p) for g in range(6) for p in (1,2)};steps=0
    for game in games:
        assert game['opening']==schedule[game['group']]['opening'];g=replay(game['opening']);assert g.ply==game['start_ply'];assert len(game['paths'])==len(game['search_info'])
        for path,info in zip(game['paths'],game['search_info']):
            assert g.result()[0] is None and info['ply']==g.ply and info['player']==g.player;assert info['engine']==('NN' if g.player==game['nn_side'] else 'Clysto派生');assert info['elapsed']>=0 and info['cpu_seconds']>=0
            if info['engine']=='NN':
                inner=info['info'];assert inner['time_limit']==.1;assert match(visit_check(inner,inner['nodes'],g.board,g.player),path)
            g.play(path);steps+=1
        assert g.result()==(game['winner'],game['reason']) and g.ply==game['plies']
    arena=dict(games=12,nn_wins=sum(g['winner']==g['nn_side'] for g in games),nn_losses=sum(g['winner'] not in (0,g['nn_side']) for g in games),true_draws=sum(g['winner']==0 and g['reason']!='max_ply' for g in games),truncated=sum(g['reason']=='max_ply' for g in games),timings={})
    for engine in ('NN','Clysto派生'):
        ss=[i for g in games for i in g['search_info'] if i['engine']==engine];arena['timings'][engine]=dict(moves=len(ss),mean_seconds=float(np.mean([i['elapsed'] for i in ss])),max_seconds=max(i['elapsed'] for i in ss),mean_cpu_seconds=float(np.mean([i['cpu_seconds'] for i in ss])),over_110ms=sum(i['elapsed']>.11 for i in ss))
    equal(arena,result['arena']);hashes=0
    for field in ('source_sha256','input_sha256','native_sha256'):
        for name,digest in cfg[field].items():assert sha(name)==digest;hashes+=1
    assert sha(cfg['model'])==cfg['model_sha256'];assert sha('nn_checkpoints/models/champions/neural121.json')==cfg['registry_sha256']
    out=dict(roadmap_step=3,passed=True,cases=288,roots=48,pure_forward_rechecked=48,mcts_searches_repeated=0,stable_pair_labels=11,games=12,replayed_game_steps=steps,hash_entries_verified=hashes,registry_model_unchanged=True,arena=arena,scope='独立重算纯策略与归档访问/选择/覆盖/摘要、逐步重放12局；不重复MCTS，不将候选外动作贴质量标签。最大深度依据归档范围核验，未重新搜索证明内部树深。')
    (E/'independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
