"""独立归档核验叶值对照；不重复搜索或实战。"""
import sys,json,math
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch,clysto121_native
import rules121 as r
from az121 import Net121,encode,action_id
from curriculum121 import replay
from diagnostics.audit_search121_diagnostic import sha,equal,match,quality,visit_check
E=Path('nn_checkpoints/experiments/leaf121_ablation_20261107')
ARMS=('nn','zero','hand')
@torch.no_grad()
def pure(model,g):
    moves=r.get_all_moves(g.board,g.player);ids=[action_id(s,p[-1],g.player,model.canonical) for s,p in moves];logits,v=model(torch.from_numpy(encode(g.board,g.player,g.ply,g.repetitions,canonical=model.canonical)).unsqueeze(0));return moves[int(logits[0,ids].argmax())][1],float(v[0])
def arena_summary(games):
    out=dict(games=len(games),nn_wins=sum(g['winner']==g['nn_side'] for g in games),nn_losses=sum(g['winner'] not in (0,g['nn_side']) for g in games),true_draws=sum(g['winner']==0 and g['reason']!='max_ply' for g in games),truncated=sum(g['reason']=='max_ply' for g in games),timings={})
    for engine in ('NN','Clysto派生'):
        ss=[i for g in games for i in g['search_info'] if i['engine']==engine];out['timings'][engine]=dict(moves=len(ss),mean_seconds=float(np.mean([i['elapsed'] for i in ss])),max_seconds=max(i['elapsed'] for i in ss),mean_cpu_seconds=float(np.mean([i['cpu_seconds'] for i in ss])),over_110ms=sum(i['elapsed']>.11 for i in ss))
    return out
def main():
    torch.set_num_threads(1);load=lambda n:json.loads((E/n).read_text());lines=lambda n:[json.loads(x) for x in (E/n).read_text().splitlines()];assert load('status.json')['stage']=='complete';cfg=load('config.json');result=load('result.json');assert result['completed'] and not result['training_started'] and not result['promoted'];source=Path(next(iter(cfg['input_sha256']))).parent;rows=json.loads((source/'dataset.json').read_text())['rows'];schedule=json.loads((source/'schedule.json').read_text());model=Net121.load(cfg['model']);model.requires_grad_(False)
    bases=lines('pure_policy.jsonl');cases=lines('cases.jsonl');assert len(bases)==48 and len(cases)==432;assert {(x['group'],x['arm'],x['simulations']) for x in cases}=={(g,a,n) for g in range(48) for a in ARMS for n in (32,128,512)};roots={}
    for group,(row,base,scheduled) in enumerate(zip(rows,bases,schedule)):
        assert group==row['group']==base['group'] and row['opening']==scheduled['opening'];g=replay(row['opening']);roots[group]=g;assert (g.board,g.player,g.ply)==(row['board'],row['player'],row['ply']);path,value=pure(model,g);assert match(path,base['pure_path']);equal(value,base['root_nn_value']);equal(math.tanh(clysto121_native.evaluate(g.board,g.player)/40.),base['root_hand_value'])
    for case in cases:
        group=case['group'];row=rows[group];g=roots[group];assert case['split']==row['split'] and case['label']==row['label'];assert case['pure_path']==bases[group]['pure_path'];assert case['quality']==quality(row,case['path']);assert case['legal']==len(r.get_all_moves(g.board,g.player));assert match(visit_check(case['info'],case['simulations'],g.board,g.player),case['path'])
        assert case['info']['time_limit'] is None and case['elapsed']>=0 and case['cpu_seconds']>=0
    summaries=[]
    for arm in ARMS:
        for n in (32,128,512):
            for split in ('all','train','development'):
                ss=[x for x in cases if x['arm']==arm and x['simulations']==n and (split=='all' or x['split']==split)];es=[x for x in ss if x['label']]
                summaries.append(dict(arm=arm,simulations=n,split=split,n=len(ss),changed_from_policy=sum(not match(x['path'],x['pure_path']) for x in ss),eligible=len(es),quality={k:sum(x['quality']==k for x in es) for k in ('稳定偏好方','稳定非偏好方','候选外未知')},mean_coverage=float(np.mean([x['info']['root_visited_actions']/x['legal'] for x in ss])),mean_seconds=float(np.mean([x['elapsed'] for x in ss])),mean_depth=float(np.mean([x['info']['mean_depth'] for x in ss]))))
    equal(summaries,load('case_summary.json'));equal(summaries,result['cases']);games=lines('games.jsonl');assert len(games)==48 and {(x['group'],x['nn_side'],x['arm']) for x in games}=={(g,p,a) for g in range(6) for p in (1,2) for a in (*ARMS,'pure')};steps=0;pure_moves=0;terminal=Counter();budget={a:[] for a in ARMS}
    for game in games:
        assert game['opening']==schedule[game['group']]['opening'];g=replay(game['opening']);assert g.ply==game['start_ply'];assert len(game['paths'])==len(game['search_info'])
        for path,info in zip(game['paths'],game['search_info']):
            assert g.result()[0] is None and info['player']==g.player and info['ply']==g.ply;assert info['engine']==('NN' if g.player==game['nn_side'] else 'Clysto派生');assert info['elapsed']>=0 and info['cpu_seconds']>=0
            if info['engine']=='NN':
                inner=info['info']
                if game['arm']=='pure':
                    assert inner==dict(mode='pure_policy',nodes=0);assert match(pure(model,g)[0],path);pure_moves+=1
                else:
                    assert inner['time_limit']==.1 and 1<=inner['nodes']<=100000;assert match(visit_check(inner,inner['nodes'],g.board,g.player),path);budget[game['arm']].append(inner['nodes'])
            g.play(path);steps+=1
        assert g.result()==(game['winner'],game['reason']) and g.ply==game['plies'];terminal[game['reason']]+=1
    arena={a:arena_summary([g for g in games if g['arm']==a]) for a in (*ARMS,'pure')};equal(arena,result['arena']);hashes=0
    for field in ('source_sha256','native_sha256','input_sha256'):
        for name,digest in cfg[field].items():assert sha(name)==digest;hashes+=1
    assert sha(cfg['model'])==cfg['model_sha256'] and sha('nn_checkpoints/models/champions/neural121.json')==cfg['registry_sha256']
    out=dict(roadmap_step=3,passed=True,roots=48,cases=432,games=48,replayed_steps=steps,pure_root_forwards=48,pure_game_forwards=pure_moves,repeated_mcts_searches=0,hash_entries=hashes,model_registry_unchanged=True,terminal_reasons=dict(terminal),arena=arena,actual_simulations={a:dict(moves=len(v),mean=float(np.mean(v)),min=min(v),max=max(v)) for a,v in budget.items()},scope='独立重放全部归档、重算根纯策略及pure实战选招、静态根值、访问与摘要；不重复MCTS，不证明内部树所有叶值，具体叶值使用由冻结源码和单元测试约束。')
    (E/'independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
