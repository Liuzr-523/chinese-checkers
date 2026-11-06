"""只读复核冻结落子次序数据；不生成局面、不搜索、不改标签。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rules121 as r
from opening121_split_audit import orbit,decision_key
from collect121_order_pairs import select_candidates,label_scores


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(run):
    run=Path(run)
    config=json.loads((run/'config.json').read_text())
    history=json.loads((run/'history_exclusions.json').read_text())
    schedule=json.loads((run/'schedule.json').read_text())
    freeze=json.loads((run/'schedule_freeze.json').read_text())
    dataset=json.loads((run/'dataset.json').read_text())
    raw=json.loads((run/'raw_scores.json').read_text())
    checks={}
    def check(name,condition):
        checks[name]=bool(condition)
        if not condition:raise ValueError('审计失败：'+name)
    check('profile',config['profile']==dataset['profile']==r.PROFILE)
    check('48_rows',len(schedule)==len(dataset['rows'])==len(raw)==48)
    check('nonterminal_children_rule',config['nonterminal_children'] is True)
    check('history_complete',not history.get('incomplete'))
    check('schedule_sha',sha(run/'schedule.json')==freeze['schedule_sha256']==dataset['schedule_sha256'])
    for name,digest in config['source_sha256'].items():
        check('snapshot:'+name,sha(run/'source'/name)==digest)
    for name in ('collect121_order_pairs.py','rules121.py','opening121_split_audit.py'):
        check('live_implementation:'+name,sha(name)==config['source_sha256'][name])
    roots=set();children=set();anchors=set();historical=set(history['orbits']);counts={'train':Counter(),'development':Counter()};depth_records=0;players=Counter();plies=Counter()
    seen=set()
    for index,(row,labeled,raw_row) in enumerate(zip(schedule,dataset['rows'],raw)):
        check(f'{index}:group_split',row['group']==index and row['split']==('train' if index<32 else 'development'))
        check(f'{index}:row_frozen',all(labeled.get(k)==v for k,v in row.items()) and labeled==raw_row)
        check(f'{index}:ply_bucket',row['ply']==(36,49,60,73)[index%4] and row['bucket']=='ply'+str(row['ply']))
        g=r.Game();anchor=None
        for path in row['opening']:
            check(f'{index}:prefix_ply{g.ply}',r.valid_path(g.board,g.player,path))
            g.play(path)
            if g.ply==8:anchor=orbit(decision_key(g.board,g.player))
        identity=decision_key(g.board,g.player);rid=orbit(identity)
        check(f'{index}:root_state',g.result()[0] is None and g.ply==row['ply'] and g.player==row['player'] and g.board==row['board'] and identity==row['root_id']==row['decision_id'] and rid==row['orbit'])
        check(f'{index}:anchor',anchor==row['anchor'] and anchor not in anchors and anchor not in historical and anchor not in seen)
        pair=select_candidates(g)
        check(f'{index}:geometry',pair is not None and list(pair)==[row['rear_path'],row['front_path']])
        ids=[rid]
        for choice,path in zip(('rear','front'),pair):
            check(f'{index}:{choice}_legal',r.valid_path(g.board,g.player,path))
            child=r.Game()
            for prior in row['opening']:child.play(prior)
            child.play(path)
            check(f'{index}:{choice}_nonterminal',child.result()[0] is None)
            ids.append(orbit(decision_key(child.board,child.player)))
            for depth in (4,5):
                a=labeled['depth_audit'][f'depth{depth}_{choice}'];records=a['search']
                check(f'{index}:{choice}_depth{depth}',a['root_requested_depth']==depth and a['child_terminal'] is False and len(records)==1 and records[0]['requested_depth']==depth-1 and records[0]['depth_reached']==depth-1 and records[0]['aborted'] is False)
                check(f'{index}:{choice}_score{depth}',labeled[f'score{depth}_{choice}']==-records[0]['score'])
                depth_records+=1
        check(f'{index}:child_orbits',ids[1:]==row['child_orbits'])
        check(f'{index}:root_child_isolation',len(set(ids))==3 and not set(ids)&historical and not set(ids)&seen and not set(ids)&anchors)
        check(f'{index}:label',labeled['label']==label_scores(**{k:labeled[k] for k in ('score4_rear','score4_front','score5_rear','score5_front')}) and labeled['eligible']==(labeled['label']!=0) and labeled['value_weight']==0.)
        counts[row['split']][labeled['label']]+=1;roots.add(rid);children.update(ids[1:]);anchors.add(anchor);seen.update(ids);players[(row['split'],g.player)]+=1;plies[g.ply]+=1
    gates=dict(train_total=counts['train'][1]+counts['train'][-1]>=16,
               development_total=counts['development'][1]+counts['development'][-1]>=8,
               train_both=min(counts['train'][1],counts['train'][-1])>=4,
               development_both=min(counts['development'][1],counts['development'][-1])>=2)
    expected_counts={part:{str(direction):counts[part][direction] for direction in (-1,1)} for part in ('train','development')}
    check('label_counts_recomputed',dataset['counts']==expected_counts)
    check('gates_recomputed',gates==dataset['gates'] and dataset['data_quality_pass']==all(gates.values()))
    # Ancestors and roots are separately fresh and globally disjoint in this collection.
    check('global_anchor_root_child_disjoint',not anchors&seen)
    check('192_depth_audits',depth_records==192)
    return dict(audit_pass=True,data_quality_pass=dataset['data_quality_pass'],gates=gates,label_counts={part:dict(c) for part,c in counts.items()},roots=len(roots),children=len(children),anchors=len(anchors),depth_audits=depth_records,players={str(k):v for k,v in players.items()},plies=dict(plies),checks=checks,source_sha256={name:sha(run/name) for name in ('config.json','history_exclusions.json','schedule.json','schedule_freeze.json','dataset.json','raw_scores.json')},scope='只读核对合法性、隔离、几何选择、冻结与固定深度标签；数据门槛失败保留为失败，不搜索/改分/补根。')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    try:result=audit(args.run)
    except Exception as error:
        result=dict(audit_pass=False,error=str(error));args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');raise
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('checks','source_sha256')},ensure_ascii=False))


if __name__=='__main__':main()
