"""One fixed independent re-confirmation authorized by user. No training/retries.
120 NEW paired games against saved v3, 40 C games per network, exact same MCTS32.
Old 60 results are provenance only: never pooled into the acceptance statistic.
"""
import argparse,fcntl,hashlib,json,os,shutil,time,traceback
from pathlib import Path
import torch
from az121 import Net121,Neural121
from curriculum121 import atomic,opening,replay,finish
from engine_c121 import CSearch121
from iterate121 import summary,paired_gate
from checkpoint121_save import verify,digest,REGISTRY,restore
import rules121 as r


def start_key(prefix):
    g=replay(prefix);return (tuple(g.board),g.player)


def fresh_opening(seed,length,used):
    for attempt in range(1000):
        prefix=opening(seed+attempt*1000003,length);key=start_key(prefix)
        if key not in used:
            used.add(key);return prefix
    raise ValueError('cannot construct fresh opening; do not reuse old positions')


def seen_openings(root):
    seen=set();sources={}
    for path in sorted(Path(root).glob('*/*.jsonl')):
        config_path=path.parent/'config.json'
        if not config_path.exists() or json.loads(config_path.read_text()).get('profile')!=r.PROFILE:continue
        added=0
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if isinstance(row,dict) and 'opening' in row:
                seen.add(start_key(row['opening']));added+=1
        if added:sources[str(path)]=dict(records=added,sha256=digest(path))
    return seen,sources


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=301048);args=ap.parse_args()
    torch.set_num_threads(1)
    saved_path=Path('nn_checkpoints/saves/v3_before_unfreeze_20261030');saved=verify(saved_path)
    if digest(REGISTRY)!=saved['registry_sha256']:raise ValueError('registry changed; preserve current champion')
    parent=Path('nn_checkpoints/experiments/unfreeze121_saved_v3_20261030');candidate_source=parent/'candidate.pt'
    baseline_path=saved_path/'model.pt';baseline=Net121.load(baseline_path);candidate=Net121.load(candidate_source)
    args.out.mkdir(parents=True,exist_ok=False);lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic()
    def update(stage,**kw):atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-start,**kw))
    def record(stage,row):
        with (args.out/(stage+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
    names=['confirm121_unfrozen.py','az121.py','rules121.py','curriculum121.py','engine_c121.py','iterate121.py','checkpoint121_save.py','neural121_registry.py']
    import ccore121
    names.append(Path(ccore121.__file__).name)
    config=dict(model=str(baseline_path.resolve()),model_sha256=digest(baseline_path),candidate_source=str(candidate_source.resolve()),candidate_sha256=digest(candidate_source),
       profile=r.PROFILE,encoding=baseline.encoding,seed=args.seed,simulations=32,value_scale=1.,root_safety=False,heuristic_mix=0.,fpu_mode='zero',policy_temperature=1.,
       training='Expert correction + genuine teacher replay; CNN/policy/value unfrozen in prior run. This is a frozen-weight independent confirmation; not pure selfplay.',
       confirmation_games=120,fixed_c_games_per_model=40,independent_gate_p=.01,c_seconds=.03,c_max_depth=10,
       rule='Exactly one new fixed 120+40+40 run. Never pool original 60 or stop early on significance. No retraining or configuration selection. C candidate points must not decrease.',
       save=str(saved_path.resolve()),save_manifest_sha256=digest(saved_path/'manifest.json'),source_sha256={p:digest(p) for p in names})
    for name in names:
        target=args.out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,target)
    shutil.copyfile(candidate_source,args.out/'candidate.pt')
    data=args.out/'data_sources';data.mkdir();config['data_sha256']={}
    for name in ('config.json','selection.json','result.json'):
        dest='prior_'+name;shutil.copyfile(parent/name,data/dest);config['data_sha256'][dest]=digest(data/dest)
    atomic(args.out/'config.json',config)
    try:
        update('preregister_openings');used,sources=seen_openings('nn_checkpoints/experiments');excluded=len(used)
        schedule={}
        for name,pairs,offset in [('confirmation',60,100000),('fixed_c',20,200000)]:
            schedule[name]=[fresh_opening(args.seed+offset+i,4+(i%(5 if name=='confirmation' else 4))*2,used) for i in range(pairs)]
        atomic(args.out/'opening_exclusions.json',dict(previous_start_positions=excluded,sources=sources))
        atomic(args.out/'schedule.json',schedule)
        for name in ('schedule.json','opening_exclusions.json'):
            shutil.copyfile(args.out/name,data/name);config['data_sha256'][name]=digest(data/name)
        atomic(args.out/'config.json',config)
        matches=[]
        for pair,prefix in enumerate(schedule['confirmation']):
            for side in (1,2):
                update('independent_confirmation',completed=len(matches),total=120)
                row=finish(replay(prefix),{side:Neural121(candidate,32),3-side:Neural121(baseline,32)})
                row.update(pair=pair,side=side,opening=prefix);matches.append(row);record('confirmation',row)
        fixed={}
        for name,model in [('control',baseline),('candidate',candidate)]:
            rows=[]
            for pair,prefix in enumerate(schedule['fixed_c']):
                for side in (1,2):
                    update('fixed_c_'+name,completed=len(rows),total=40)
                    row=finish(replay(prefix),{side:Neural121(model,32),3-side:CSearch121(seconds=.03,max_depth=10)})
                    row.update(pair=pair,side=side,opening=prefix);rows.append(row);record('fixed_c_'+name,row)
            fixed[name]=summary(rows)
        if digest(candidate_source)!=config['candidate_sha256'] or digest(args.out/'candidate.pt')!=config['candidate_sha256']:raise ValueError('frozen candidate changed')
        gate=paired_gate(matches)
        qualified=gate['one_sided_p']<=.01 and fixed['candidate']['points']>=fixed['control']['points']
        result=dict(confirmation=summary(matches),paired_gate=gate,fixed_c=fixed,qualified=qualified,promoted=False,
                    prior_games_pooled=0,weights_unchanged=True,seconds=time.monotonic()-start)
        atomic(args.out/'result.json',result);verify(saved_path)
        if digest(REGISTRY)!=saved['registry_sha256']:raise ValueError('concurrent registry change; do not overwrite')
        if qualified:
            from neural121_registry import register,load_registered
            try:
                entry=register(args.out,'v4_unfrozen_confirmed');engine=load_registered()
                for p in (1,2):
                    path,info=engine.think(r.init_board(),p)
                    if not r.valid_path(r.init_board(),p,path) or info['root_visits']!=32:raise ValueError('registered engine smoke check failed')
                atomic(args.out/'adoption.json',dict(adopted=True,version=entry['version'],save_preserved=True))
            except BaseException:
                active=json.loads(REGISTRY.read_text())
                if active.get('version')=='v4_unfrozen_confirmed' and active.get('sha256')==config['candidate_sha256']:
                    restore(saved_path);atomic(args.out/'adoption.json',dict(adopted=False,rolled_back=True))
                raise
        else:atomic(args.out/'adoption.json',dict(adopted=False,retained_version=saved['version'],save_verified=True))
        update('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        update('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
