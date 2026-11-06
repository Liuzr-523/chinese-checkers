"""路线图第2步：固定48局专家轨迹；只收集/审计，不训练。"""
import argparse,copy,fcntl,hashlib,json,os,random,shutil,time,traceback
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
import rules121 as r
from az121 import encode,action_id
from curriculum121 import atomic,replay
from corrected121_decisions import ConsistentExpert,score_action
from engine_c121 import CSearch121
from opening121 import OpeningBook,key
from opening121_split_audit import orbit
from history121_exclusions import collect_history,resolve_legacy_endgames
from checkpoint121_save import REGISTRY,digest
BASE=Path('nn_checkpoints/experiments/fresh121_projection_20261105/history_exclusions.json')
SEED=111061


def value_label(winner,reason,player):
    if winner not in (0,1,2):raise ValueError('未结束对局不得赋值')
    if reason=='max_ply':
        if winner!=0:raise ValueError('限步不得伪造胜者')
        return 0.,0.
    return (0. if winner==0 else 1. if winner==player else -1.),1.


def exited(board):
    return sum(sum(board[i]==p and i not in r.HOME[p] for i in range(121)) for p in (1,2))/2


def incremental_history(root,out):
    root=Path(root);base=json.loads(BASE.read_text());exclude=[out,BASE];unchanged={}
    for name,sha in base['source_sha256'].items():
        path=root/name
        if path.is_file() and digest(path)==sha:unchanged[name]=sha;exclude.append(path)
    delta=resolve_legacy_endgames(collect_history(root,exclude=exclude),root)
    if base.get('incomplete') or delta.get('incomplete'):raise ValueError('历史覆盖有缺口')
    states=set(base['orbits'])|set(delta['orbits']);sources={**unchanged,**delta['source_sha256'],str(BASE.relative_to(root)):digest(BASE)}
    # 已看过的第一步诊断中显式局面也纳入；不是游戏胜负增强。
    diagnostic=Path('docs/experiments/roadmap121_step1_20261105');seen=collect_history(diagnostic)
    if seen.get('incomplete'):raise ValueError('第一步诊断覆盖未解决')
    states.update(seen['orbits'])
    return dict(orbits=sorted(states),source_sha256=sources,base_sha256=digest(BASE),base_coverage=base['coverage'],delta_coverage=delta['coverage'],diagnostic_source_sha256=seen['source_sha256'],diagnostic_coverage=seen['coverage'],incomplete=[])


def mark_exclusions(records,historical):
    splits=defaultdict(set)
    for row in records:splits[row['orbit']].add(row['split'])
    shared={k for k,v in splits.items() if len(v)>1}
    for row in records:
        reasons=[]
        if row['orbit'] in historical:reasons.append('historical_state')
        if row['orbit'] in shared:reasons.append('cross_split_state')
        row['excluded_reasons']=reasons;row['eligible']=not reasons
        if reasons:row['value_weight']=0.;row['policy_weight']=0.
    return shared


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(1)
    a.out.mkdir(parents=True,exist_ok=False);lock=(a.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic();registry=digest(REGISTRY)
    def status(stage,**kw):
        atomic(a.out/'status.json',dict(roadmap_step=2,stage=stage,pid=os.getpid(),seconds=time.monotonic()-started,**kw))
        if time.monotonic()-started>3600 and stage not in ('complete','failed'):raise TimeoutError('一小时采集预算耗尽')
    def log(s):
        with (a.out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-started:.1f}秒：{s}\n')
    sources=['expert121_pilot48.py','history121_exclusions.py','opening121_split_audit.py','rules121.py','az121.py','curriculum121.py','corrected121_decisions.py','engine_c121.py','engine_clysto121.py','opening121.py','opening121_book.json','tests/test_expert121_pilot48.py']
    import ccore121,clysto121_native
    sources.extend([Path(ccore121.__file__).name,Path(clysto121_native.__file__).name,'ccore121.c','clysto121_generated/core.c'])
    for name in sources:
        dest=a.out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
    cfg=dict(roadmap_step=2,profile=r.PROFILE,seed=SEED,groups=24,train_groups=16,development_groups=8,games=48,max_ply=240,max_seconds=3600,max_attempts_per_group=96,engines=dict(derived5=dict(depth=5,safety_seconds=120,opening_book=False),tip5=dict(depth=5,safety_seconds=120,opening_book=False)),buckets=dict(near_book=dict(groups=8,root_ply=14,min_mean_exited=4.5),off_book=dict(groups=8,root_ply=24,min_mean_exited=6.),lagged=dict(groups=8,root_ply=14,max_mean_exited=4.)),split='group<16 train，其余development；near/off/lag轮流，训练6/5/5组、开发2/3/3组；同组两局交换derived所在阵营，P1始终先手。',labels='真实胜负side-to-move；真规则和0权重1；max_ply0权重0。实际derived走法policy1，tip实际走法policy0；每局tip首次行动前额外查询derived建议并forced-child核验，probe value0。',dedup='从采集root后导出；历史/跨split保守几何轨道在双方监督均排除，完整棋谱全部保留，不改split、不补局、不按胜负筛选。',gates='48局全部合法可重放/搜索depth5完整；train>=12组、development>=6组有真实终局；过滤后train>=512/development>=256条有效value、终局前16手>=128/64；双方与三个开局桶都有有效value；有效probe>=40/48。不要求挑选胜负平衡。',training_started=False,registry_sha256=registry,source_sha256={n:digest(n) for n in sources})
    atomic(a.out/'config.json',cfg)
    try:
        status('构建历史排除');history=incremental_history('nn_checkpoints/experiments',a.out);atomic(a.out/'history_exclusions.json',history);historical=set(history['orbits']);used=set(historical);anchors=set();book=OpeningBook();expert=ConsistentExpert(5);ConsistentExpert.failure_dir=a.out;schedule=[];generation_queries=0
        for group in range(24):
            bucket=('near_book','off_book','lagged')[group%3];part='train' if group<16 else 'development';target=cfg['buckets'][bucket]['root_ply'];accepted=None
            for attempt in range(96):
                status('冻结分桶开局',completed=group,total=24,bucket=bucket,attempt=attempt);rng=random.Random(SEED+group*100003+attempt*10000019);g=r.Game();prefix=[]
                while g.ply<6:
                    choices=book.moves(g.board,g.player,g.ply,g.repetitions) if bucket=='near_book' and g.ply<4 else []
                    path=rng.choice(choices or [p for _,p in r.get_all_moves(g.board,g.player)]);g.play(path);prefix.append(path)
                anchor=orbit(key(g.board,g.player))
                if anchor in used or anchor in anchors:continue
                while g.ply<target and g.result()[0] is None:
                    if bucket=='lagged':path=rng.choice([p for _,p in r.get_all_moves(g.board,g.player)])
                    else:path,info=expert.think(g.board,g.player,g.ply,g.repetitions);generation_queries+=1
                    g.play(path);prefix.append(path)
                if g.result()[0] is not None:continue
                root=orbit(key(g.board,g.player));progress=exited(g.board)
                if root in used or root in anchors:continue
                limit=cfg['buckets'][bucket]
                if progress<limit.get('min_mean_exited',-1) or progress>limit.get('max_mean_exited',11):continue
                accepted=dict(group=group,split=part,bucket=bucket,attempt=attempt,opening=prefix,anchor=anchor,root_orbit=root,mean_exited=progress,start_ply=g.ply);break
            if accepted is None:raise ValueError('固定候选预算未集齐桶：'+bucket)
            schedule.append(accepted);used.add(root);anchors.add(anchor);atomic(a.out/'schedule.json',schedule)
        freeze=digest(a.out/'schedule.json');atomic(a.out/'schedule_freeze.json',dict(sha256=freeze,groups=24,root_and_anchor_historical_overlap=0,generation_queries=generation_queries));log('24组开局冻结，开始固定48局，不按胜负替换。')
        games=[];probes=[]
        for spec in schedule:
            for derived_side in (1,2):
                game_id=len(games);g=replay(spec['opening']);actors={derived_side:expert,3-derived_side:CSearch121(seconds=120,max_depth=5)};paths=[];infos=[];probe_done=False
                while g.result()[0] is None:
                    status('采集48局',completed=len(games),total=48,game_id=game_id,group=spec['group'],ply=g.ply)
                    if g.player!=derived_side and not probe_done:
                        preferred,pinfo=expert.think(g.board,g.player,g.ply,g.repetitions);forced=score_action(g,preferred,5);valid=forced==pinfo['score'];probes.append(dict(game_id=game_id,group=spec['group'],split=spec['split'],ply=g.ply,prefix=spec['opening']+copy.deepcopy(paths),preferred=preferred,root_score=pinfo['score'],forced_score=forced,valid=valid,value_weight=0.));probe_done=True
                    player=g.player;actor='derived5' if player==derived_side else 'tip5';path,info=actors[player].think(g.board,player,g.ply,g.repetitions)
                    if path is None or info['aborted'] or info['depth_reached']!=5:raise ValueError('actor搜索未完整达到depth5')
                    if not r.valid_path(g.board,player,path):raise ValueError('actor非法走法')
                    paths.append(path);infos.append(dict(player=player,actor=actor,score=info['score'],depth=info['depth_reached'],nodes=info['nodes_searched'],aborted=info['aborted']));g.play(path)
                    if len(paths)%10==0:atomic(a.out/'current_game.json',dict(game_id=game_id,opening=spec['opening'],paths=paths))
                winner,reason=g.result();row=dict(game_id=game_id,group=spec['group'],split=spec['split'],bucket=spec['bucket'],derived_side=derived_side,opening=spec['opening'],start_ply=spec['start_ply'],paths=paths,search_info=infos,plies=g.ply,winner=winner,reason=reason);games.append(row)
                with (a.out/'games.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                atomic(a.out/'policy_probes.json',probes);log(f'第{len(games)}/48局：组{spec["group"]}，派生方P{derived_side}，胜者{winner}，{reason}，{g.ply}手。')
        status('重放与监督过滤');records=[];samples=[]
        for game in games:
            g=replay(game['opening'])
            for index,(path,info) in enumerate(zip(game['paths'],game['search_info'])):
                if g.player!=info['player'] or not r.valid_path(g.board,g.player,path):raise ValueError('完整重放失败')
                z,w=value_label(game['winner'],game['reason'],g.player);identity=key(g.board,g.player);actions=[action_id(s,p[-1],g.player,True) for s,p in r.get_all_moves(g.board,g.player)];target=actions.index(action_id(path[0],path[-1],g.player,True));record=dict(game_id=game['game_id'],group=game['group'],split=game['split'],bucket=game['bucket'],ply=g.ply,player=g.player,decision_id=identity,orbit=orbit(identity),z=z,value_weight=w,policy_weight=float(info['actor']=='derived5'),policy_source=info['actor'],terminal_distance=len(game['paths'])-index,terminal_reason=game['reason']);records.append(record);samples.append(dict(x=torch.from_numpy(encode(g.board,g.player,g.ply,g.repetitions,canonical=True)),actions=actions,target=target,z=z,value_weight=w,policy_weight=record['policy_weight'],group=game['group'],game_id=game['game_id'],decision_id=identity));g.play(path)
            if g.result()!=(game['winner'],game['reason']) or g.ply!=game['plies']:raise ValueError('真实终局重放不一致')
        shared=mark_exclusions(records,historical)
        for s,row in zip(samples,records):s.update(value_weight=row['value_weight'],policy_weight=row['policy_weight'])
        # probes从未执行的建议不继承胜负；按同一历史/跨组排除规则。
        exported_probes=[]
        for probe in probes:
            g=replay(probe['prefix']);identity=key(g.board,g.player);o=orbit(identity)
            if not probe['valid'] or o in historical or o in shared:continue
            actions=[action_id(s,p[-1],g.player,True) for s,p in r.get_all_moves(g.board,g.player)];preferred=action_id(probe['preferred'][0],probe['preferred'][-1],g.player,True);exported_probes.append(dict(x=torch.from_numpy(encode(g.board,g.player,g.ply,g.repetitions,canonical=True)),actions=actions,target=actions.index(preferred),z=0.,value_weight=0.,policy_weight=1.,policy_source='derived5_probe',group=probe['group'],game_id=probe['game_id'],split=probe['split'],decision_id=identity))
        parts={p:[s for s,row in zip(samples,records) if row['split']==p and (row['value_weight'] or row['policy_weight'])] for p in ('train','development')};torch.save(dict(profile=r.PROFILE,encoding='axial17x17-6planes-own-home-top-pair121x121-v2',sets=parts,policy_probes=exported_probes),a.out/'dataset.pt');atomic(a.out/'sample_metadata.json',records)
        summary={}
        for part in ('train','development'):
            rows=[x for x in records if x['split']==part and x['value_weight']>0];genuine=[g for g in games if g['split']==part and g['reason']!='max_ply'];summary[part]=dict(games=sum(g['split']==part for g in games),terminal_games=len(genuine),terminal_groups=len({g['group'] for g in genuine}),effective_value_samples=len(rows),effective_groups=len({x['group'] for x in rows}),endgame16_samples=sum(x['terminal_distance']<=16 for x in rows),player_labels={str(p):dict(Counter(str(x['z']) for x in rows if x['player']==p)) for p in (1,2)},buckets=dict(Counter(x['bucket'] for x in rows)),outcomes=dict(Counter(f'{g["winner"]}:{g["reason"]}' for g in games if g['split']==part)))
        gates=dict(all_48=len(games)==48,train_terminal_groups=summary['train']['terminal_groups']>=12,development_terminal_groups=summary['development']['terminal_groups']>=6,train_value=summary['train']['effective_value_samples']>=512,development_value=summary['development']['effective_value_samples']>=256,train_endgame=summary['train']['endgame16_samples']>=128,development_endgame=summary['development']['endgame16_samples']>=64,valid_probes=sum(p['valid'] for p in probes)>=40,bucket_coverage=all(len(summary[s]['buckets'])==3 for s in summary),player_coverage=all(all(summary[s]['player_labels'][str(p)] for p in (1,2)) for s in summary))
        if digest(REGISTRY)!=registry or digest(a.out/'schedule.json')!=freeze:raise ValueError('现役或冻结开局变化')
        result=dict(roadmap_step=2,complete_games=48,summary=summary,gates=gates,data_quality_pass=all(gates.values()),cross_split_raw_orbits=len(shared),excluded_records=sum(not x['eligible'] for x in records),valid_policy_probes=sum(x['valid'] for x in probes),exported_policy_probes=len(exported_probes),dataset_sha256=digest(a.out/'dataset.pt'),seconds=time.monotonic()-started,training_started=False,champion_changed=False);atomic(a.out/'result.json',result)
        lines=['# 路线图第2步：48局专家数据试点','',f'采集完成：48局；数据质量门槛：{"通过" if all(gates.values()) else "未通过"}。本轮未训练。','', '|划分|真实终局|终局组|过滤后价值样本|终局前16手样本|','|---|---:|---:|---:|---:|']
        for s,v in summary.items():lines.append(f'|{s}|{v["terminal_games"]}|{v["terminal_groups"]}|{v["effective_value_samples"]}|{v["endgame16_samples"]}|')
        lines+=['',f'原始轨迹跨组几何轨道{len(shared)}；冲突/历史状态在双方监督过滤，完整棋谱不删除。策略探针有效{sum(x["valid"] for x in probes)}/48，最终导出{len(exported_probes)}。','',json.dumps(gates,ensure_ascii=False),'', '引擎均固定depth5，交换配置所在阵营，P1先手未变；派生搜索并非原版Clysto。所有胜负由严格规则真实终局给出，max_ply价值权重0，真实和棋0权重1。没有旋转换色胜负增强；没有按输赢补局或改split。'];(a.out/'REPORT.md').write_text('\n'.join(lines));log('48局收集与审计结束；'+('可规划第3步固定训练对照。' if all(gates.values()) else '数据门槛未通过，不启动第3步。'));status('complete',data_quality_pass=all(gates.values()))
    except BaseException:status('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
