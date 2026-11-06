"""冻结机制后以真实固定深度续弈标注小步接力；不训练、不按标签补根。"""
import argparse,copy,fcntl,json,os,random,shutil,time,traceback
from collections import Counter,defaultdict
from pathlib import Path
import rules121 as r
from curriculum121 import atomic,replay
from checkpoint121_save import digest
from opening121_split_audit import orbit,decision_key
from collect121_order_pairs import tip_distance,farthest,build_history
from corrected121_decisions import ConsistentExpert
SEED=111091


def forward_chains(board,start,player):
    return [path for path in r.legal_paths(board,start) if len(path)>=3 and tip_distance(path[-1],player)<tip_distance(start,player)]


def relay_witness(g,path):
    if len(path)!=2 or path[1] not in r.NEIGHBORS[path[0]] or not r.valid_path(g.board,g.player,path):return None
    start,end=path;maximum=farthest(g.board,g.player)
    if tip_distance(start,g.player)!=maximum or tip_distance(end,g.player)>=maximum:return None
    original=forward_chains(g.board,start,g.player);all_original=[p for p in r.legal_paths(g.board,start) if tip_distance(p[-1],g.player)<tip_distance(start,g.player)];old_ends={p[-1] for p in all_original}
    child=r.apply_move(g.board,path);chains=forward_chains(child,end,g.player);new=[p for p in chains if p[-1] not in old_ends]
    if not new:return None
    new.sort(key=lambda p:(-(tip_distance(end,g.player)-tip_distance(p[-1],g.player)),-len(p),p[-1]))
    return dict(original_forward_chains=original,all_forward_paths=all_original,child_forward_chains=chains,new_forward_chains=new,best_new_chain=new[0],new_chain_advance=tip_distance(end,g.player)-tip_distance(new[0][-1],g.player),rear_start_tip_distance=maximum)


def relay_candidates(g):
    maximum=farthest(g.board,g.player);rears=[];fronts=[]
    for start,value in enumerate(g.board):
        if value!=g.player:continue
        if tip_distance(start,g.player)==maximum:
            for end in r.NEIGHBORS[start]:
                path=[start,end];witness=relay_witness(g,path)
                if witness is not None:rears.append(((-witness['new_chain_advance'],start,end),path,witness))
        else:
            for path in forward_chains(g.board,start,g.player):
                advance=tip_distance(start,g.player)-tip_distance(path[-1],g.player)
                fronts.append(((-advance,-len(path),start,path[-1]),path))
    if not rears or not fronts:return None
    _,rear,witness=min(rears,key=lambda x:x[0]);_,front=min(fronts,key=lambda x:x[0])
    return rear,front,witness


def finish_branch(row,choice,depth,status,checkpoint):
    g=replay(row['opening']);root_ply=g.ply;root_player=g.player;paths=[];infos=[];expert=ConsistentExpert(depth)
    forced=row[choice+'_path'];g.play(forced);paths.append(forced);infos.append(dict(forced=True,player=root_player,ply=root_ply,path=forced))
    while g.result()[0] is None:
        status('真实续弈',group=row['group'],choice=choice,depth=depth,ply=g.ply)
        player=g.player;ply=g.ply;path,info=expert.think(g.board,g.player,g.ply,g.repetitions)
        if path is None or info['aborted'] or info['depth_reached']!=depth:raise ValueError('固定深度续弈未完整')
        g.play(path);paths.append(path);infos.append(dict(forced=False,player=player,ply=ply,path=path,requested_depth=depth,**info))
        checkpoint(dict(group=row['group'],choice=choice,depth=depth,opening=row['opening'],paths=paths,search_info=infos))
    winner,reason=g.result()
    if reason=='max_ply' and winner!=0:raise ValueError('限步不可伪造赢家')
    return dict(group=row['group'],choice=choice,depth=depth,opening=row['opening'],root_ply=root_ply,start_ply=root_ply,root_player=root_player,player=root_player,forced_path=forced,paths=paths,search_info=infos,plies=g.ply,own_moves=(g.ply-root_ply+1)//2,winner=winner,reason=reason)


def mask_cross_inputs(rows,historical):
    inputs=defaultdict(set);trajectory=defaultdict(set)
    for row in rows:
        for identity in [row['orbit']]+row['child_orbits']:inputs[identity].add(row['group'])
        for branches in row['returns'].values():
            for game in branches.values():
                g=replay(game['opening']);trajectory[orbit(decision_key(g.board,g.player))].add(row['group'])
                for path in game['paths']:
                    g.play(path);trajectory[orbit(decision_key(g.board,g.player))].add(row['group'])
    parts={row['group']:row['split'] for row in rows};excluded=set();collisions=[]
    for identity,owners in inputs.items():
        for owner in owners:
            others={g for g in trajectory.get(identity,set()) if parts[g]!=parts[owner]}
            if others:excluded.update({owner}|others);collisions.append(dict(orbit=identity,input_group=owner,other_split_trajectory_groups=sorted(others)))
    for row in rows:
        row['raw_label']=row['label'];row['exclusion_reasons']=[]
        if row['group'] in excluded:row['label']=0;row['eligible']=False;row['exclusion_reasons']=['input_cross_split_trajectory']
    shared=[identity for identity,groups in trajectory.items() if len({parts[g] for g in groups})>1]
    return dict(masked_groups=sorted(excluded),input_collisions=collisions,trajectory_orbits=len(trajectory),historical_trajectory_orbits=len(set(trajectory)&set(historical)),cross_split_trajectory_orbits=len(shared),scope='仅输入root/child碰另一split续弈时双方整组mask；完整原始四局保留。续弈本身共享/历史状态只统计，不声称轨迹完全独立。')


def readiness(rows):
    counts={part:{str(direction):sum(row['split']==part and row['eligible'] and row['label']==direction for row in rows) for direction in (-1,1)} for part in ('train','development')}
    gates={part:dict(total=sum(c.values())>=8,both_directions=min(c.values())>=2) for part,c in counts.items()}
    return dict(training=all(gates['train'].values()),development=all(gates['development'].values())),gates,counts


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--exclude-parent',action='store_true');args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic()
    def status(stage,**kwargs):
        atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),seconds=time.monotonic()-started,**kwargs))
        if stage not in ('complete','failed') and time.monotonic()-started>=3600:raise TimeoutError('一小时机制续弈采集预算耗尽')
    def log(message):
        with (args.out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-started:.1f}秒：{message}\n')
    from relay121_targets import stable_preference
    import ccore121,clysto121_native
    sources=['collect121_relay_returns.py','tests/test_relay121_collection.py','relay121_targets.py','collect121_order_pairs.py','expert121_pilot48.py','history121_exclusions.py','opening121_split_audit.py','corrected121_decisions.py','rules121.py','curriculum121.py','engine_clysto121.py','engine_c121.py','ccore121.c','clysto121_generated/core.c',Path(ccore121.__file__).name,Path(clysto121_native.__file__).name]
    cfg=dict(profile=r.PROFILE,seed=SEED,groups=48,train=24,development=24,root_plies=[40,41,56,57],max_attempts_per_group=128,max_seconds=3600,advance_depth=3,continuation_depths=[3,4],query_safety_seconds=120,max_ply=240,source_sha256={n:digest(n) for n in sources},labels='两深度真实终局points优先；同为根方胜时own_moves差>=2才tempo；max_ply无偏好；两深度非0同号才label。',scope='rear最远棋子邻步前进并新增静态前进多跳入口；front来自更近目标棋子。root/child/8锚点历史保守几何隔离；prefix仅来源。续弈轨迹共享保留统计；标签不表示伪造胜负。')
    for n in sources:
        target=args.out/'source'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,target)
    atomic(args.out/'config.json',cfg)
    try:
        status('机制历史覆盖');history=build_history(args.out,args.exclude_parent);atomic(args.out/'history_exclusions.json',history);historical=set(history['orbits']);used=set(historical);schedule=[];expert=ConsistentExpert(3);ConsistentExpert.failure_dir=args.out;generation_queries=0
        for group in range(48):
            target=(40,41,56,57)[group%4];accepted=None
            for attempt in range(128):
                status('冻结接力机制根',group=group,attempt=attempt);rng=random.Random(SEED+group*100003+attempt*10000019);g=r.Game();prefix=[]
                for _ in range(8):
                    path=rng.choice([p for _,p in r.get_all_moves(g.board,g.player)]);g.play(path);prefix.append(path)
                anchor=orbit(decision_key(g.board,g.player))
                if anchor in used:continue
                while g.ply<target and g.result()[0] is None:
                    status('深度3推进机制候选',group=group,attempt=attempt,ply=g.ply)
                    path,info=expert.think(g.board,g.player,g.ply,g.repetitions);generation_queries+=1
                    if path is None:break
                    g.play(path);prefix.append(path)
                if g.ply!=target or g.result()[0] is not None:continue
                pair=relay_candidates(g)
                if pair is None:continue
                rear,front,witness=pair;children=[];terminal=False
                for path in (rear,front):
                    child=copy.deepcopy(g);child.play(path);terminal|=child.result()[0] is not None;children.append(orbit(decision_key(child.board,child.player)))
                identity=decision_key(g.board,g.player);rid=orbit(identity);new=[rid]+children+[anchor]
                if terminal or len(set(new))!=4 or set(new)&used:continue
                accepted=dict(group=group,split='train' if group<24 else 'development',bucket='ply40_41' if target<50 else 'ply56_57',board=g.board,player=g.player,ply=g.ply,opening=prefix,root_id=identity,decision_id=identity,orbit=rid,anchor=anchor,child_orbits=children,rear_path=rear,front_path=front,mechanism_witness=witness);break
            if accepted is None:raise ValueError(f'组{group}在128候选内未找到机制根；停止不放宽条件')
            schedule.append(accepted);used.update(new);atomic(args.out/'schedule_partial.json',schedule)
        atomic(args.out/'schedule.json',schedule);freeze=dict(schedule_sha256=digest(args.out/'schedule.json'),roots=48,generation_queries=generation_queries);atomic(args.out/'schedule_freeze.json',freeze);log('48个机制根先冻结，后192局真实续弈。')
        rows=[]
        for row in schedule:
            returns={}
            for depth in (3,4):
                returns[str(depth)]={}
                for choice in ('rear','front'):
                    status('开始强制分支真实续弈',group=row['group'],depth=depth,choice=choice)
                    game=finish_branch(row,choice,depth,status,lambda current:atomic(args.out/'current_branch.json',current));returns[str(depth)][choice]=game
                    with (args.out/'games.jsonl').open('a') as f:f.write(json.dumps(game,ensure_ascii=False)+'\n')
                    atomic(args.out/'partial_returns.json',dict(group=row['group'],returns=returns))
            preference=stable_preference(returns,row['player']);label=int(preference['label']);rows.append(dict(**row,returns=returns,label=label,eligible=label!=0,value_weight=0.,preference_audit=preference));atomic(args.out/'raw_rows.json',rows);log(f'组{row["group"]}四局完成；原偏好{label}')
        if digest(args.out/'schedule.json')!=freeze['schedule_sha256']:raise ValueError('冻结根变化')
        overlaps=mask_cross_inputs(rows,historical);atomic(args.out/'trajectory_overlap.json',overlaps);ready,gates,counts=readiness(rows);atomic(args.out/'dataset.json',dict(profile=r.PROFILE,rows=rows,readiness=ready,gates=gates,counts=counts,schedule_sha256=freeze['schedule_sha256']));log('机制续弈数据完成；'+json.dumps(dict(readiness=ready,counts=counts),ensure_ascii=False));status('complete',readiness=ready)
    except BaseException:status('failed',error=traceback.format_exc());raise


if __name__=='__main__':main()
