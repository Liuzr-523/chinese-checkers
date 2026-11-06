"""固定NN先验，只替换非终局叶值；无训练或晋升。"""
import sys,json,time,shutil,fcntl,traceback,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import ccore121,clysto121_native
import rules121 as r
from az121 import Net121,search,encode,action_id
from curriculum121 import replay,atomic
from engine_clysto121 import ClystoDerived121
from checkpoint121_save import REGISTRY,digest
from diagnostics.search121_diagnostic import summarize_games
SOURCE=Path('nn_checkpoints/experiments/relay121_returns_20261106_resume1/collection')
OUT=Path('nn_checkpoints/experiments/leaf121_ablation_20261107')
PROTOCOL='docs/experiments/leaf121_ablation_20261107/protocol.md'
ARMS=('nn','zero','hand')

def hand_value(board,player):return math.tanh(clysto121_native.evaluate(board,player)/40.)
def options(arm):return dict(value_scale=0.) if arm=='zero' else dict(leaf_value_fn=hand_value) if arm=='hand' else {}
def match(a,b):return a is not None and b is not None and (a[0],a[-1])==(b[0],b[-1])
@torch.no_grad()
def pure(model,g):
    moves=r.get_all_moves(g.board,g.player);ids=[action_id(s,p[-1],g.player,model.canonical) for s,p in moves]
    logits,value=model(torch.from_numpy(encode(g.board,g.player,g.ply,g.repetitions,canonical=model.canonical)).unsqueeze(0));return moves[int(logits[0,ids].argmax())][1],float(value[0])
def summary(cases):
    out=[]
    for arm in ARMS:
        for n in (32,128,512):
            for split in ('all','train','development'):
                xs=[x for x in cases if x['arm']==arm and x['simulations']==n and (split=='all' or x['split']==split)];es=[x for x in xs if x['label']]
                out.append(dict(arm=arm,simulations=n,split=split,n=len(xs),changed_from_policy=sum(not match(x['path'],x['pure_path']) for x in xs),eligible=len(es),quality={k:sum(x['quality']==k for x in es) for k in ('稳定偏好方','稳定非偏好方','候选外未知')},mean_coverage=float(np.mean([x['info']['root_visited_actions']/x['legal'] for x in xs])),mean_seconds=float(np.mean([x['elapsed'] for x in xs])),mean_depth=float(np.mean([x['info']['mean_depth'] for x in xs]))))
    return out

def main():
    OUT.mkdir(exist_ok=False);lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(111102)
    reg=json.loads(REGISTRY.read_text());assert reg['version']=='v3_expert_imitation' and digest(reg['checkpoint'])==reg['sha256'];model=Net121.load(reg['checkpoint']);model.requires_grad_(False)
    raw=json.loads((SOURCE/'dataset.json').read_text())['rows'];schedule=json.loads((SOURCE/'schedule.json').read_text());assert len(raw)==len(schedule)==48
    names=['diagnostics/leaf121_ablation.py','diagnostics/search121_diagnostic.py','tests/test_leaf121_ablation.py','az121.py','rules121.py','curriculum121.py','engine_clysto121.py','engine_c121.py',PROTOCOL]
    cfg=dict(roadmap_step=3,seed=111102,arms=list(ARMS),arena_arms=list(ARMS)+['pure'],simulations=[32,128,512],hand_scale=40,arena_groups=list(range(6)),arena_games_per_arm=12,arena_seconds=.1,max_seconds=1800,model=reg['checkpoint'],model_sha256=reg['sha256'],registry_sha256=digest(REGISTRY),source_sha256={n:digest(n) for n in names},native_sha256={str(Path(m.__file__)):digest(m.__file__) for m in (ccore121,clysto121_native)},input_sha256={str(SOURCE/n):digest(SOURCE/n) for n in ('dataset.json','schedule.json')},scope='旧诊断局面，纯策略不限时对照，其余同软预算；无训练/晋升。')
    for n in names:
        dest=OUT/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,dest)
    atomic(OUT/'config.json',cfg)
    def status(stage,**kw):
        atomic(OUT/'status.json',dict(stage=stage,roadmap_step=3,seconds=time.monotonic()-started,**kw))
        if time.monotonic()-started>1800:raise TimeoutError('固定30分钟诊断预算耗尽')
    def record(n,row):
        with (OUT/n).open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    try:
        cases=[];pure(model,replay(schedule[0]['opening']))
        for row in raw:
            g=replay(row['opening']);pure_path,value=pure(model,g)
            base=dict(group=row['group'],split=row['split'],label=row['label'],player=g.player,ply=g.ply,legal=len(r.get_all_moves(g.board,g.player)),pure_path=pure_path,root_nn_value=value,root_hand_value=hand_value(g.board,g.player),rear_path=row['rear_path'],front_path=row['front_path'])
            record('pure_policy.jsonl',base)
            for n in (32,128,512):
                for arm in ARMS:
                    status('固定叶值局面对照',group=row['group'],arm=arm,simulations=n);wall=time.monotonic();cpu=time.process_time();path,info=search(model,g.board,g.player,g.ply,g.repetitions,simulations=n,**options(arm))
                    pref=row['rear_path'] if row['label']==1 else row['front_path'] if row['label']==-1 else None
                    quality='无稳定标签' if pref is None else '稳定偏好方' if match(path,pref) else '稳定非偏好方' if any(match(path,row[k+'_path']) for k in ('rear','front')) else '候选外未知'
                    item=dict(**base,arm=arm,simulations=n,path=path,quality=quality,elapsed=time.monotonic()-wall,cpu_seconds=time.process_time()-cpu,info=info);cases.append(item);record('cases.jsonl',item)
        atomic(OUT/'case_summary.json',summary(cases));games=[]
        for group in range(6):
            for side in (1,2):
                for arm in (*ARMS,'pure'):
                    g=replay(schedule[group]['opening']);paths=[];infos=[];opponent=ClystoDerived121(seconds=.1,max_depth=62)
                    while g.result()[0] is None:
                        status('固定叶值实战对照',group=group,nn_side=side,arm=arm,ply=g.ply);wall=time.monotonic();cpu=time.process_time();player=g.player
                        if player!=side:path,info=opponent.think(g.board,g.player,g.ply,g.repetitions)
                        elif arm=='pure':path,_=pure(model,g);info=dict(mode='pure_policy',nodes=0)
                        else:path,info=search(model,g.board,g.player,g.ply,g.repetitions,simulations=100000,time_limit=.1,**options(arm))
                        if path is None:raise ValueError('非终局无走法')
                        infos.append(dict(player=player,ply=g.ply,engine='NN' if player==side else 'Clysto派生',elapsed=time.monotonic()-wall,cpu_seconds=time.process_time()-cpu,info=info));g.play(path);paths.append(path)
                        atomic(OUT/'current_game.json',dict(group=group,nn_side=side,arm=arm,opening=schedule[group]['opening'],paths=paths,search_info=infos))
                    win,reason=g.result();game=dict(group=group,nn_side=side,arm=arm,opening=schedule[group]['opening'],start_ply=schedule[group]['ply'],paths=paths,search_info=infos,winner=win,reason=reason,plies=g.ply);games.append(game);record('games.jsonl',game)
        for hashes in (cfg['source_sha256'],cfg['native_sha256'],cfg['input_sha256']):
            for n,sha in hashes.items():
                if digest(n)!=sha:raise ValueError('冻结工件变化：'+n)
        assert digest(REGISTRY)==cfg['registry_sha256'] and digest(cfg['model'])==cfg['model_sha256']
        atomic(OUT/'result.json',dict(roadmap_step=3,completed=True,training_started=False,promoted=False,cases=summary(cases),arena={arm:summarize_games([g for g in games if g['arm']==arm]) for arm in (*ARMS,'pure')},seconds=time.monotonic()-started));status('complete')
    except BaseException:
        atomic(OUT/'status.json',dict(stage='failed',error=traceback.format_exc(),seconds=time.monotonic()-started));raise
if __name__=='__main__':main()
