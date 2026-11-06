"""固定现役v3，诊断PUCT选择与实际时间；不训练或晋升。"""
import sys,json,time,hashlib,shutil,fcntl,traceback,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import rules121 as r
from az121 import Net121,search,encode,action_id
from curriculum121 import replay,atomic
from engine_clysto121 import ClystoDerived121
from checkpoint121_save import digest,REGISTRY

SOURCE=Path('nn_checkpoints/experiments/relay121_returns_20261106_resume1/collection')
OUT=Path('nn_checkpoints/experiments/search121_diagnostic_20261106')
PROTOCOL=Path('docs/experiments/search121_diagnostic_20261106/protocol.md')

def main():
    OUT.mkdir(exist_ok=False);lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(111101)
    registry=json.loads(REGISTRY.read_text());model=Net121.load(registry['checkpoint']);model.requires_grad_(False)
    assert registry['version']=='v3_expert_imitation' and digest(registry['checkpoint'])==registry['sha256']
    rows=json.loads((SOURCE/'dataset.json').read_text())['rows'];schedule=json.loads((SOURCE/'schedule.json').read_text());assert len(rows)==len(schedule)==48
    names=['tests/test_mcts121_diagnostics.py','diagnostics/search121_diagnostic.py','az121.py','rules121.py','curriculum121.py','engine_c121.py','engine_clysto121.py',str(PROTOCOL)]
    import ccore121,clysto121_native
    cfg=dict(native_sha256={str(Path(m.__file__)):digest(m.__file__) for m in (ccore121,clysto121_native)},roadmap_step=3,seed=111101,model=registry['checkpoint'],model_sha256=registry['sha256'],registry_sha256=digest(REGISTRY),source_sha256={n:digest(n) for n in names},input_sha256={str(SOURCE/n):digest(SOURCE/n) for n in ('dataset.json','schedule.json')},simulations=[32,128,512],value_scales=[1.,0.],arena_groups=list(range(6)),arena_seconds=.1,arena_games=12,max_seconds=1800,scope='已看48根诊断和12局软时间预算，不是全新独立验收，无训练/晋升。')
    for n in names:
        dest=OUT/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,dest)
    atomic(OUT/'config.json',cfg)
    def status(stage,**kw):
        atomic(OUT/'status.json',dict(roadmap_step=3,stage=stage,seconds=time.monotonic()-started,**kw))
        if time.monotonic()-started>1800:raise TimeoutError('固定30分钟诊断预算耗尽')
    def record(name,row):
        with (OUT/name).open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    try:
        model(torch.from_numpy(encode(rows[0]['board'],rows[0]['player'],canonical=model.canonical)).unsqueeze(0))
        cases=[]
        for row in rows:
            g=replay(row['opening']);moves=r.get_all_moves(g.board,g.player);ids=[action_id(s,p[-1],g.player,model.canonical) for s,p in moves]
            with torch.no_grad():logits,v=model(torch.from_numpy(encode(g.board,g.player,g.ply,g.repetitions,canonical=model.canonical)).unsqueeze(0))
            best=int(logits[0,ids].argmax());pure=moves[best][1];a=row['rear_path'];b=row['front_path']
            pref=a if row['label']==1 else b if row['label']==-1 else None
            def match(p,q):return q is not None and (p[0],p[-1])==(q[0],q[-1])
            def quality(path):
                if pref is None:return '无稳定标签'
                if match(path,pref):return '稳定偏好方'
                if match(path,a) or match(path,b):return '稳定非偏好方'
                return '候选外未知'
            base=dict(group=row['group'],split=row['split'],label=row['label'],player=g.player,ply=g.ply,legal=len(moves),pure_path=pure,pure_quality=quality(pure),root_value=float(v[0]),rear_path=a,front_path=b)
            record('pure_policy.jsonl',base)
            for simulations in (32,128,512):
                for scale in (1.,0.):
                    status('固定局面搜索诊断',group=row['group'],simulations=simulations,value_scale=scale)
                    wall=time.monotonic();cpu=time.process_time();path,info=search(model,g.board,g.player,g.ply,g.repetitions,simulations=simulations,value_scale=scale)
                    ci=next(i for i,p in enumerate(info['paths']) if match(p,path));ai=next(i for i,p in enumerate(info['paths']) if match(p,a));bi=next(i for i,p in enumerate(info['paths']) if match(p,b))
                    out=dict(**base,simulations=simulations,value_scale=scale,path=path,quality=quality(path),changed_from_policy=not match(path,pure),elapsed=time.monotonic()-wall,cpu_seconds=time.process_time()-cpu,chosen_policy_rank=1+sum(float(logits[0,i])>float(logits[0,ids[ci]]) for i in ids),pair_visits=[info['visits'][ai],info['visits'][bi]],pair_q_root=[-info['child_q'][ai],-info['child_q'][bi]],info=info)
                    cases.append(out);record('cases.jsonl',out)
        atomic(OUT/'case_summary.json',summarize_cases(cases))
        games=[]
        for group in range(6):
            for side in (1,2):
                g=replay(schedule[group]['opening']);paths=[];infos=[];opponent=ClystoDerived121(seconds=.1,max_depth=62)
                while g.result()[0] is None:
                    status('固定软时间预算对局',group=group,nn_side=side,ply=g.ply)
                    wall=time.monotonic();cpu=time.process_time();player=g.player
                    if player==side:path,info=search(model,g.board,g.player,g.ply,g.repetitions,simulations=100000,time_limit=.1)
                    else:path,info=opponent.think(g.board,g.player,g.ply,g.repetitions)
                    if path is None:raise ValueError('未终局却无走法')
                    infos.append(dict(player=player,ply=g.ply,engine='NN' if player==side else 'Clysto派生',elapsed=time.monotonic()-wall,cpu_seconds=time.process_time()-cpu,info=info));g.play(path);paths.append(path)
                    atomic(OUT/'current_game.json',dict(group=group,nn_side=side,opening=schedule[group]['opening'],paths=paths,search_info=infos))
                winner,reason=g.result();game=dict(group=group,nn_side=side,opening=schedule[group]['opening'],start_ply=schedule[group]['ply'],paths=paths,search_info=infos,winner=winner,reason=reason,plies=g.ply);games.append(game);record('games.jsonl',game)
        if digest(REGISTRY)!=cfg['registry_sha256'] or digest(cfg['model'])!=cfg['model_sha256']:raise ValueError('现役模型或注册表变化')
        for hashes in (cfg['source_sha256'],cfg['input_sha256'],cfg['native_sha256']):
            for name,sha in hashes.items():
                if digest(name)!=sha:raise ValueError('冻结源码或输入变化：'+name)
        atomic(OUT/'result.json',dict(roadmap_step=3,completed=True,training_started=False,promoted=False,cases=summarize_cases(cases),arena=summarize_games(games),seconds=time.monotonic()-started));status('complete')
    except BaseException:
        atomic(OUT/'status.json',dict(stage='failed',error=traceback.format_exc(),seconds=time.monotonic()-started));raise

def summarize_cases(cases):
    out=[]
    for n in (32,128,512):
        for scale in (1.,0.):
            for split in ('train','development','all'):
                ss=[x for x in cases if x['simulations']==n and x['value_scale']==scale and (split=='all' or x['split']==split)];eligible=[x for x in ss if x['label']];q={k:sum(x['quality']==k for x in eligible) for k in ('稳定偏好方','稳定非偏好方','候选外未知')}
                out.append(dict(simulations=n,value_scale=scale,split=split,n=len(ss),changed_from_policy=sum(x['changed_from_policy'] for x in ss),eligible=len(eligible),quality=q,pure_quality={k:sum(x['pure_quality']==k for x in eligible) for k in q},mean_seconds=float(np.mean([x['elapsed'] for x in ss])),mean_coverage=float(np.mean([x['info']['root_visited_actions']/x['legal'] for x in ss])),mean_depth=float(np.mean([x['info']['mean_depth'] for x in ss])),max_depth=max(x['info']['max_depth'] for x in ss),pair_visit_correct=sum((x['pair_visits'][0]-x['pair_visits'][1])*x['label']>0 for x in eligible)))
    return out

def summarize_games(games):
    moves={s:[x for g in games for x in g['search_info'] if x['engine']==s] for s in ('NN','Clysto派生')}
    return dict(games=len(games),nn_wins=sum(g['winner']==g['nn_side'] for g in games),nn_losses=sum(g['winner'] not in (0,g['nn_side']) for g in games),true_draws=sum(g['winner']==0 and g['reason']!='max_ply' for g in games),truncated=sum(g['reason']=='max_ply' for g in games),timings={s:dict(moves=len(xs),mean_seconds=float(np.mean([x['elapsed'] for x in xs])),max_seconds=max(x['elapsed'] for x in xs),mean_cpu_seconds=float(np.mean([x['cpu_seconds'] for x in xs])),over_110ms=sum(x['elapsed']>.11 for x in xs)) for s,xs in moves.items()})

if __name__=='__main__':main()
