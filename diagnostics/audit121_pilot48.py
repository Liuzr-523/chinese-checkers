"""只读独立重放48局和每条导出监督；不调用搜索或训练。"""
import argparse,json,sys
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import rules121 as r
from az121 import encode,action_id
from curriculum121 import replay
from opening121 import key
from opening121_split_audit import orbit
from checkpoint121_save import REGISTRY,digest


def ensure(condition,message):
    if not condition:raise ValueError(message)


def audit(folder):
    folder=Path(folder);cfg=json.loads((folder/'config.json').read_text());schedule=json.loads((folder/'schedule.json').read_text());history=json.loads((folder/'history_exclusions.json').read_text());historical=set(history['orbits']);games=[json.loads(s) for s in (folder/'games.jsonl').read_text().splitlines()];metadata=json.loads((folder/'sample_metadata.json').read_text());dataset=torch.load(folder/'dataset.pt',map_location='cpu',weights_only=True);result=json.loads((folder/'result.json').read_text());probes=json.loads((folder/'policy_probes.json').read_text());freeze=json.loads((folder/'schedule_freeze.json').read_text())
    ensure(len(schedule)==24 and len(games)==48 and len(probes)<=48 and len({p['game_id'] for p in probes})==len(probes),'组/对局/探针数量')
    ensure(digest(folder/'schedule.json')==freeze['sha256'],'开局冻结SHA')
    ensure(cfg['profile']==r.PROFILE==dataset['profile'],'规则版本')
    ensure(digest(REGISTRY)==cfg['registry_sha256'],'现役注册表变化')
    active=json.loads(REGISTRY.read_text());model=Path(active['checkpoint']);ensure(digest(model)==active['sha256'],'现役模型SHA不符')
    for name,sha in cfg['source_sha256'].items():ensure(digest(folder/'source'/name)==sha,'冻结源不符：'+name)
    roots=[];anchors=[]
    for i,s in enumerate(schedule):
        ensure(s['group']==i and s['split']==('train' if i<16 else 'development'),'组划分')
        g=replay(s['opening']);an=replay(s['opening'][:6]);o=orbit(key(g.board,g.player));a=orbit(key(an.board,an.player));ensure(o==s['root_orbit'] and a==s['anchor'] and o not in historical and a not in historical,'根/锚点隔离')
        ensure(g.ply==s['start_ply'] and g.result()[0] is None,'非终局采集root');roots.append(o);anchors.append(a)
    ensure(len(set(roots))==len(set(anchors))==24,'根/锚点唯一性')
    ensure(Counter(s['bucket'] for s in schedule)==dict(near_book=8,off_book=8,lagged=8),'三桶数量')
    expected=[];outcomes=defaultdict(Counter);probe_by_game={p['game_id']:p for p in probes}
    for i,row in enumerate(games):
        spec=schedule[i//2];side=1+i%2;ensure(row['game_id']==i and row['group']==i//2 and row['derived_side']==side,'交换引擎位置')
        ensure(row['winner'] in (0,1,2),'未结束对局标签')
        ensure(row['opening']==spec['opening'] and row['split']==spec['split'],'同组开局/划分')
        ensure(len(row['paths'])==len(row['search_info']),'搜索记录数');g=replay(row['opening']);probe=probe_by_game.get(i);probe_seen=False
        for offset,(path,info) in enumerate(zip(row['paths'],row['search_info'])):
            actor='derived5' if g.player==side else 'tip5'
            ensure(info['player']==g.player and info['actor']==actor and info['depth']==5 and not info['aborted'],'搜索角色/固定深度')
            if probe is not None and g.ply==probe['ply']:
                ensure(g.player!=side and probe['prefix']==row['opening']+row['paths'][:offset],'探针来自实际tip首回合')
                ensure(r.valid_path(g.board,g.player,probe['preferred']) and probe['value_weight']==0,'探针合法/无胜负')
                ensure(probe['valid']==(probe['root_score']==probe['forced_score']),'探针精确一致性');probe_seen=True
            ensure(r.valid_path(g.board,g.player,path),'非法完整棋谱')
            z=0. if row['winner']==0 else 1. if row['winner']==g.player else -1.;weight=0. if row['reason']=='max_ply' else 1.;actions=[action_id(s,p[-1],g.player,True) for s,p in r.get_all_moves(g.board,g.player)];identity=key(g.board,g.player)
            expected.append(dict(game_id=i,group=row['group'],split=row['split'],player=g.player,ply=g.ply,decision_id=identity,orbit=orbit(identity),z=z,value_weight=weight,policy_weight=float(actor=='derived5'),x=torch.from_numpy(encode(g.board,g.player,g.ply,g.repetitions,canonical=True)),actions=actions,target=actions.index(action_id(path[0],path[-1],g.player,True))))
            g.play(path)
        ensure(probe_seen if probe is not None else all(i['actor']=='derived5' for i in row['search_info']),'探针无对应状态');ensure(g.result()==(row['winner'],row['reason']) and g.ply==row['plies'],'真实终局/手数');outcomes[row['split']][f'{row["winner"]}:{row["reason"]}']+=1
    ensure(len(expected)==len(metadata),'全体监督记录数');state_splits=defaultdict(set)
    for row in expected:state_splits[row['orbit']].add(row['split'])
    shared={o for o,s in state_splits.items() if len(s)>1};parts=defaultdict(list)
    for original,meta in zip(expected,metadata):
        for name in ('game_id','group','player','ply','decision_id','orbit','z'):ensure(original[name]==meta[name],'监督身份/真实标签不一致')
        reasons=[]
        if original['orbit'] in historical:reasons.append('historical_state')
        if original['orbit'] in shared:reasons.append('cross_split_state')
        if reasons:original['value_weight']=original['policy_weight']=0.
        ensure(meta['excluded_reasons']==reasons and meta['eligible']==(not reasons),'监督排除原因')
        ensure(original['value_weight']==meta['value_weight'] and original['policy_weight']==meta['policy_weight'],'双权重不符')
        if original['value_weight'] or original['policy_weight']:parts[original['split']].append(original)
    for part in ('train','development'):
        actual=dataset['sets'][part];ensure(len(actual)==len(parts[part]),'导出数不符')
        for a,b in zip(actual,parts[part]):
            ensure(torch.equal(a['x'],b['x']),'输入编码/历史不符')
            for k in ('game_id','group','decision_id','z','value_weight','policy_weight','actions','target'):ensure(a[k]==b[k],'导出字段不符：'+k)
    exported_probe_keys={(s['game_id'],s['decision_id']) for s in dataset['policy_probes']}
    expected_probe_keys=set()
    for p in probes:
        g=replay(p['prefix']);o=orbit(key(g.board,g.player))
        if p['valid'] and o not in historical and o not in shared:expected_probe_keys.add((p['game_id'],key(g.board,g.player)))
    ensure(exported_probe_keys==expected_probe_keys and len(exported_probe_keys)==len(dataset['policy_probes']),'探针导出集合')
    ensure(all(s['value_weight']==0 and s['policy_weight']==1 for s in dataset['policy_probes']),'probe双权重')
    for sample in dataset['policy_probes']:
        probe=probe_by_game[sample['game_id']];g=replay(probe['prefix']);actions=[action_id(s,p[-1],g.player,True) for s,p in r.get_all_moves(g.board,g.player)]
        ensure(sample['split']==probe['split'] and sample['group']==probe['group'] and sample['z']==0 and sample['policy_source']=='derived5_probe','probe划分与标签来源')
        ensure(torch.equal(sample['x'],torch.from_numpy(encode(g.board,g.player,g.ply,g.repetitions,canonical=True))),'probe编码')
        ensure(sample['actions']==actions and actions[sample['target']]==action_id(probe['preferred'][0],probe['preferred'][-1],g.player,True),'probe建议目标')
    ensure(result['dataset_sha256']==digest(folder/'dataset.pt') and result['complete_games']==48,'最终工件SHA')
    eligible={part:{o['orbit'] for o in parts[part]} for part in ('train','development')};ensure(not(eligible['train']&eligible['development']),'导出监督仍跨split')
    ensure(not((eligible['train']|eligible['development'])&historical),'导出监督仍有历史状态')
    return dict(roadmap_step=2,audit_pass=True,games_replayed=48,groups_checked=24,raw_sample_records=len(expected),exported_samples={p:len(parts[p]) for p in parts},valid_policy_probes=sum(p['valid'] for p in probes),exported_policy_probes=len(exported_probe_keys),raw_cross_split_orbits=len(shared),exported_cross_split_orbits=0,exported_historical_orbits=0,terminal_outcomes={p:dict(v) for p,v in outcomes.items()},dataset_sha256=digest(folder/'dataset.pt'),source_sha256={str(Path(__file__)):digest(__file__)},input_sha256={n:digest(folder/n) for n in ('config.json','schedule.json','games.jsonl','sample_metadata.json','policy_probes.json','result.json')},champion_sha256=digest(model),scope='主代理独立程序重放；本轮子代理用量受限，非三代理复核。没有训练/搜索或新对局。')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(1);result=audit(a.folder);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','source_sha256')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
