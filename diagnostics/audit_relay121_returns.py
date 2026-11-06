"""独立重放本轮全部根和归档；不搜索、不训练、不调用生产标签/评测器。"""
import argparse, hashlib, json, sys
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rules121 as r
from opening121 import MIRROR

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(b,p):return ''.join(map(str,b))+':'+str(p)
def orb(b,p):
    mirror=[0]*121
    for i,v in enumerate(b):mirror[MIRROR[i]]=v
    return min(key(b,p),key(mirror,p),key([3-v if v else 0 for v in b[::-1]],3-p),key([3-v if v else 0 for v in mirror[::-1]],3-p))
def replay(paths):
    g=r.Game()
    for path in paths:g.play(path)
    return g

def dist(i,p):
    x,y=r.COORDS[i];a,b=r.COORDS[120 if p==1 else 0]
    return max(abs(x-a),abs(y-b),abs(x+y-a-b))
def direction(pair,p):
    a,b=pair['rear'],pair['front']
    if any(x['reason']=='max_ply' or x['winner'] is None for x in (a,b)):return 0,'invalid'
    points=[.5 if x['winner']==0 else float(x['winner']==p) for x in (a,b)]
    if points[0]!=points[1]:return (1 if points[0]>points[1] else -1),'point'
    if a['winner']==b['winner']==p and abs(a['own_moves']-b['own_moves'])>=2:return (1 if a['own_moves']<b['own_moves'] else -1),'tempo'
    return 0,'none'

def audit_partial(e,out):
    c=e/'collection';rows=json.loads((c/'schedule_partial.json').read_text());history=json.loads((c/'history_exclusions.json').read_text());used=set(history['orbits'])
    for row in rows:
        g=replay(row['opening']);p=g.player;rear,front=row['rear_path'],row['front_path']
        assert (g.board,p,g.ply)==(row['board'],row['player'],row['ply'])
        maximum=max(dist(i,p) for i,v in enumerate(g.board) if v==p)
        assert r.valid_path(g.board,p,rear) and r.valid_path(g.board,p,front)
        assert len(rear)==2 and rear[-1] in r.NEIGHBORS[rear[0]] and dist(rear[0],p)==maximum and dist(rear[-1],p)<maximum
        assert len(front)>=3 and dist(front[0],p)<maximum and dist(front[-1],p)<dist(front[0],p)
        oldends={path[-1] for path in r.legal_paths(g.board,rear[0]) if dist(path[-1],p)<maximum}
        child=r.apply_move(g.board,rear);chain=row['mechanism_witness']['best_new_chain']
        assert len(chain)>=3 and r.valid_path(child,p,chain) and chain[0]==rear[-1] and chain[-1] not in oldends and dist(chain[-1],p)<dist(chain[0],p)
        children=[]
        for path in (rear,front):
            x=replay(row['opening']);x.play(path);assert x.result()[0] is None;children.append(orb(x.board,x.player))
        anchor=replay(row['opening'][:8]);ids=[orb(g.board,p)]+children+[orb(anchor.board,anchor.player)]
        assert len(set(ids))==4 and not set(ids)&used;used.update(ids)
        assert row['orbit']==ids[0] and children==row['child_orbits'] and row['anchor']==ids[3]
    result=dict(roadmap_step=3,status='采集失败，部分根独立核验通过；完整实验审计不可适用',partial_roots_verified=len(rows),history_orbits=len(history['orbits']),historical_or_cross_input_overlap=0,games=0,dataset_exists=(c/'dataset.json').exists(),schedule_frozen=(c/'schedule_freeze.json').exists(),training_started=False,scope='独立重放所有已接收根及双方子局面；核验邻步接力新终点、前锋连跳和历史/跨组输入排除。未产生续弈归档，不宣称完成192局审计。',partial_schedule_sha256=sha(c/'schedule_partial.json'))
    assert not result['dataset_exists'] and not result['schedule_frozen'] and not (c/'games.jsonl').exists()
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--partial',action='store_true');args=ap.parse_args();e=args.experiment;c=e/'collection';load=lambda p:json.loads(p.read_text())
    if args.partial:return audit_partial(e,args.out)
    data=load(c/'dataset.json');rows=data['rows'];schedule=load(c/'schedule.json');assert len(rows)==len(schedule)==48
    assert sha(c/'schedule.json')==load(c/'schedule_freeze.json')['schedule_sha256']==data['schedule_sha256']
    resume_audit=None
    if (c/'resume_provenance.json').exists():
        provenance=load(c/'resume_provenance.json');source=Path(provenance['source']);prior=load(source/'schedule_partial.json')
        assert schedule[:len(prior)]==prior and len(prior)==provenance['reused_roots']==22
        assert sha(source/'history_exclusions.json')==sha(c/'history_exclusions.json')==provenance['history_sha256']
        assert sha(source/'schedule_partial.json')==provenance['partial_schedule_sha256']
        assert float(load(c/'status.json')['seconds'])>=provenance['previous_seconds']
        for field,file in (('source_status_sha256','status.json'),('source_config_sha256','config.json')):assert sha(source/file)==provenance[field]
        resume_audit=dict(reused_roots=len(prior),prefix_exact=True,history_byte_identical=True,prior_budget_seconds=provenance['previous_seconds'])
    historical=set(load(c/'history_exclusions.json')['orbits']);used=set(historical);trajectory=defaultdict(set);inputs={};games=[];counts=Counter();kinds=Counter();search_moves=0
    for row,frozen in zip(rows,schedule):
        for k,v in frozen.items():assert row[k]==v,(row['group'],k)
        group=row['group'];assert group==len(inputs);assert row['split']==('train' if group<24 else 'development');assert row['ply']==(40,41,56,57)[group%4]
        g=replay(row['opening']);assert (g.board,g.player,g.ply)==(row['board'],row['player'],row['ply']);p=g.player;rear,front=row['rear_path'],row['front_path'];maximum=max(dist(i,p) for i,v in enumerate(g.board) if v==p)
        assert r.valid_path(g.board,p,rear) and r.valid_path(g.board,p,front)
        assert len(rear)==2 and rear[-1] in r.NEIGHBORS[rear[0]] and dist(rear[0],p)==maximum and dist(rear[-1],p)<maximum
        assert len(front)>=3 and dist(front[0],p)<maximum and dist(front[-1],p)<dist(front[0],p)
        original=[path for path in r.legal_paths(g.board,rear[0]) if dist(path[-1],p)<maximum];oldends={path[-1] for path in original};child=r.apply_move(g.board,rear);chains=[path for path in r.legal_paths(child,rear[-1]) if len(path)>=3 and dist(path[-1],p)<dist(rear[-1],p)];new=[path for path in chains if path[-1] not in oldends];assert new
        new.sort(key=lambda path:(-(dist(rear[-1],p)-dist(path[-1],p)),-len(path),path[-1]))
        w=row['mechanism_witness'];assert w['all_forward_paths']==original and w['child_forward_chains']==chains and w['new_forward_chains']==new;assert w['best_new_chain'] in new
        root=orb(g.board,p);children=[]
        for path in (rear,front):
            x=replay(row['opening']);x.play(path);assert x.result()[0] is None;children.append(orb(x.board,x.player))
        anchor=replay(row['opening'][:8]);anchor=orb(anchor.board,anchor.player);identities=[root]+children+[anchor];assert len(set(identities))==4 and not used.intersection(identities);used.update(identities);assert root==row['orbit'] and children==row['child_orbits'] and anchor==row['anchor'];inputs[group]=set(identities[:3]);assert row['value_weight']==0
        labels=[]
        for depth in (3,4):
            pair=row['returns'][str(depth)]
            for choice in ('rear','front'):
                game=pair[choice];games.append(game);assert game['opening']==row['opening'];x=replay(game['opening']);assert game['start_ply']==game['root_ply']==x.ply and game['root_player']==p
                paths,infos=game['paths'],game['search_info'];assert paths[0]==row[choice+'_path'] and len(paths)==len(infos);trajectory[orb(x.board,x.player)].add(group)
                for i,(path,info) in enumerate(zip(paths,infos)):
                    assert x.result()[0] is None;assert info['path']==path and info['ply']==x.ply and info['player']==x.player
                    if i==0:assert info['forced'] is True
                    else:assert info['forced'] is False and not info['aborted'] and info['requested_depth']==info['depth_reached']==depth;search_moves+=1
                    x.play(path);trajectory[orb(x.board,x.player)].add(group)
                assert x.result()==(game['winner'],game['reason']) and x.ply==game['plies'];assert game['own_moves']==sum(info['player']==p for info in infos)==(x.ply-row['ply']+1)//2
                counts[game['reason']]+=1
            label,kind=direction(pair,p);labels.append(label);kinds[kind]+=1;assert label==row['preference_audit']['per_depth'][str(depth)]['label']
        raw=labels[0] if labels[0] and labels[0]==labels[1] else 0;assert raw==row['raw_label'];assert row['eligible']==(row['label']!=0)
    archived=[json.loads(line) for line in (c/'games.jsonl').read_text().splitlines()];assert games==archived and len(games)==192
    masked=set()
    for group,ids in inputs.items():
        for identity in ids:
            for other in trajectory[identity]:
                if (group<24)!=(other<24):masked.update((group,other))
    assert sorted(masked)==load(c/'trajectory_overlap.json')['masked_groups']
    for row in rows:assert row['label']==(0 if row['group'] in masked else row['raw_label'])
    effective={part:dict(Counter(str(row['label']) for row in rows if row['split']==part and row['eligible'])) for part in ('train','development')}
    independent_readiness={part:sum(count.values())>=8 and min(count.get('1',0),count.get('-1',0))>=2 for part,count in effective.items()}
    assert data['readiness']==dict(training=independent_readiness['train'],development=independent_readiness['development'])
    for name,digest in load(e/'collection_manifest.json').items():assert sha(name)==digest,name
    overlap=load(c/'trajectory_overlap.json')
    assert overlap['trajectory_orbits']==len(trajectory)
    assert overlap['historical_trajectory_orbits']==len(set(trajectory)&historical)
    assert overlap['cross_split_trajectory_orbits']==sum(len({group<24 for group in groups})>1 for groups in trajectory.values())
    hashes=0
    for config in (load(e/'config.json'),load(c/'config.json')):
        for field in ('source_sha256','native_sha256','old_data_sha256'):
            for name,digest in config.get(field,{}).items():assert sha(name)==digest,name;hashes+=1
    result=dict(roadmap_step=3,passed=True,resume=resume_audit,roots=48,games=192,search_moves=search_moves,hash_entries_verified=hashes,terminal_reasons=dict(counts),per_configuration_label_kinds=dict(kinds),effective_directions=effective,readiness=independent_readiness,trajectory_overlap=dict(total=overlap['trajectory_orbits'],historical=overlap['historical_trajectory_orbits'],cross_split=overlap['cross_split_trajectory_orbits']),masked_groups=sorted(masked),historical_input_overlap=0,scope='独立重放所有归档、机制、标签、深度、冻结根与输入隔离；不重新搜索，深度依据逐步归档审计，未证明搜索最优。')
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
