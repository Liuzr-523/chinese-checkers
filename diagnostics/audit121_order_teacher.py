"""只诊断历史7例的教师相对排序；不更改新根标签、不训练或选择候选。"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from curriculum121 import replay,atomic
from checkpoint121_save import digest
from collect121_order_pairs import audited_score
SOURCE=Path('nn_checkpoints/experiments/targeted121_complete_cycle_20261030/seeds.json')
BRANCHES=Path('nn_checkpoints/experiments/recheck121_counterfactual_20261031/games.jsonl')
POSITIVE={9,13,47};NEGATIVE={12,31,40,44}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic();seeds=json.loads(SOURCE.read_text());branches={(r['seed_id'],r['role']):r for r in map(json.loads,BRANCHES.read_text().splitlines()) if r['repeat']==0};rows=[]
    atomic(a.out/'config.json',dict(depths=[4,5,6],seed_ids=sorted(POSITIVE|NEGATIVE),max_seconds=300,scope='只查7个已知历史动作对；教师深度相对评分与固定续弈方向是否一致，不作新训练/开发选择，不挑最好深度，不修改48新根。',source_sha256={str(p):digest(p) for p in (SOURCE,BRANCHES,Path(__file__),Path('collect121_order_pairs.py'))}))
    for i in sorted(POSITIVE|NEGATIVE):
        g=replay(seeds[i]['opening']);expected=1 if i in POSITIVE else -1;rear=branches[i,'expert']['opening'][-1];front=branches[i,'played']['opening'][-1];scores=[]
        for d in (4,5,6):
            if time.monotonic()-start>300:raise TimeoutError('历史教师诊断5分钟上限')
            atomic(a.out/'status.json',dict(stage='历史教师诊断',seed_id=i,depth=d,seconds=time.monotonic()-start))
            qa,ia=audited_score(g,rear,d);qb,ib=audited_score(g,front,d);gap=qa-qb;scores.append(dict(depth=d,rear_score=qa,front_score=qb,gap=gap,agrees_with_archived_continuation=gap*expected>0,rear_audit=ia,front_audit=ib))
        rows.append(dict(seed_id=i,player=g.player,ply=g.ply,rear_path=rear,front_path=front,archived_continuation_preference=expected,scores=scores));atomic(a.out/'rows.json',rows)
    result=dict(roots=7,queries=42,seconds=time.monotonic()-start,by_depth={str(d):dict(all_agreement=sum(s['agrees_with_archived_continuation'] for r in rows for s in r['scores'] if s['depth']==d),rear_winning_agreement=sum(s['agrees_with_archived_continuation'] for r in rows if r['archived_continuation_preference']==1 for s in r['scores'] if s['depth']==d),front_preferred_agreement=sum(s['agrees_with_archived_continuation'] for r in rows if r['archived_continuation_preference']==-1 for s in r['scores'] if s['depth']==d)) for d in (4,5,6)},rows=rows,scope='教师相对评分与两套旧续弈方向的机制比较；续弈并非最优博弈，不凭相符数择教师深度或训练模型。')
    atomic(a.out/'result.json',result);atomic(a.out/'status.json',dict(stage='complete',seconds=time.monotonic()-start));print(json.dumps(result['by_depth']))
if __name__=='__main__':main()
