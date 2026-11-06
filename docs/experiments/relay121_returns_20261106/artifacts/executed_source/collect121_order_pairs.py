"""固定48根后排/前排动作排序数据；先冻结根，再独立固定深度评分。"""
import argparse,copy,fcntl,json,os,random,shutil,time,traceback
from pathlib import Path
import rules121 as r
from curriculum121 import atomic
from checkpoint121_save import digest
from opening121_split_audit import orbit,decision_key
from history121_exclusions import collect_history
from expert121_pilot48 import incremental_history
from corrected121_decisions import ConsistentExpert
import corrected121_decisions as corrected
SEED=111081


def tip_distance(index,player):
    q,y=r.COORDS[index];a,b=r.COORDS[120 if player==1 else 0]
    return max(abs(q-a),abs(y-b),abs(q+y-a-b))


def farthest(board,player):
    return max(tip_distance(i,player) for i,v in enumerate(board) if v==player)


def select_candidates(g):
    before=farthest(g.board,g.player);rear=[];front=[]
    for start,path in r.get_all_moves(g.board,g.player):
        after=farthest(r.apply_move(g.board,path),g.player)
        advance=tip_distance(start,g.player)-tip_distance(path[-1],g.player)
        if after<before:rear.append(((after,-advance,start,path[-1]),path))
        if len(path)>=3 and advance>0 and after>=before:front.append(((-advance,-len(path),start,path[-1]),path))
    if not rear or not front:return None
    return min(rear,key=lambda x:x[0])[1],min(front,key=lambda x:x[0])[1]


def label_scores(score4_rear,score4_front,score5_rear,score5_front):
    a=score4_rear-score4_front;b=score5_rear-score5_front
    return (1 if b>0 else -1) if a*b>0 and abs(b)>=2 else 0


def build_history(out,exclude_parent=False):
    excluded=out.parent if exclude_parent else out
    history=incremental_history('nn_checkpoints/experiments',excluded)
    diagnostic=collect_history('docs/experiments',exclude=[excluded])
    if diagnostic.get('incomplete'):raise ValueError('最新文档历史覆盖缺口：'+json.dumps(diagnostic['incomplete'],ensure_ascii=False))
    history['orbits']=sorted(set(history['orbits'])|set(diagnostic['orbits']))
    history['latest_docs_source_sha256']=diagnostic['source_sha256'];history['latest_docs_coverage']=diagnostic['coverage']
    return history


def audited_score(g,path,depth):
    records=[];original=corrected.ConsistentExpert
    class AuditExpert(original):
        def think(self,*args,**kwargs):
            result=super().think(*args,**kwargs)
            records.append(dict(requested_depth=self.requested_depth,**result[1]))
            return result
    corrected.ConsistentExpert=AuditExpert
    try:score=corrected.score_action(g,path,depth)
    finally:corrected.ConsistentExpert=original
    child=copy.deepcopy(g);child.play(path)
    if child.result()[0] is None:
        if len(records)!=1 or records[0]['depth_reached']!=depth-1 or records[0]['aborted']:raise ValueError('固定深度评分未完成')
    elif records:raise ValueError('终局child不应搜索')
    return score,dict(root_requested_depth=depth,child_terminal=child.result()[0] is not None,search=records)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--exclude-parent',action='store_true');parser.add_argument('--history-only',action='store_true');a=parser.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);lock=(a.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic()
    def status(stage,**kw):
        atomic(a.out/'status.json',dict(stage=stage,pid=os.getpid(),seconds=time.monotonic()-started,**kw))
        if stage not in ('complete','failed') and time.monotonic()-started>=3600:raise TimeoutError('一小时采集预算耗尽')
    def log(text):
        with (a.out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-started:.1f}秒：{text}\n')
    names=['collect121_order_pairs.py','tests/test_order121_collection.py','expert121_pilot48.py','history121_exclusions.py','opening121_split_audit.py','corrected121_decisions.py','engine_clysto121.py','rules121.py','curriculum121.py']
    import ccore121,clysto121_native
    names.extend([Path(ccore121.__file__).name,Path(clysto121_native.__file__).name,'ccore121.c','clysto121_generated/core.c'])
    cfg=dict(profile=r.PROFILE,seed=SEED,groups=48,train=32,development=16,root_plies=[36,49,60,73],max_attempts_per_group=96,max_seconds=3600,score_depths=[4,5],expert_safety_seconds=120,advancement_depth=3,value_labels=False,nonterminal_children=True,scope='root/两个child/8手anchor保守几何隔离；prefix仅provenance，不声称整轨迹隔离。几何两child必须非终局，否则整根排除、不换次优。',source_sha256={n:digest(n) for n in names})
    for n in names:
        target=a.out/'source'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,target)
    atomic(a.out/'config.json',cfg)
    try:
        status('历史覆盖预检查');history=build_history(a.out,a.exclude_parent);atomic(a.out/'history_exclusions.json',history);log(f'历史预检查完成：{len(history["orbits"])}轨道')
        if a.history_only:status('complete',history_only=True);return
        used=set(history['orbits']);anchors=set();schedule=[];expert=ConsistentExpert(3);ConsistentExpert.failure_dir=a.out;generation_queries=0
        for group in range(48):
            target=(36,49,60,73)[group%4];accepted=None
            for attempt in range(96):
                status('冻结排序根',group=group,attempt=attempt);rng=random.Random(SEED+group*100003+attempt*10000019);g=r.Game();prefix=[]
                for _ in range(8):
                    path=rng.choice([p for _,p in r.get_all_moves(g.board,g.player)]);g.play(path);prefix.append(path)
                anchor=orbit(decision_key(g.board,g.player))
                if anchor in used or anchor in anchors:continue
                while g.ply<target and g.result()[0] is None:
                    status('深度3推进候选',group=group,attempt=attempt,ply=g.ply)
                    path,info=expert.think(g.board,g.player,g.ply,g.repetitions);generation_queries+=1
                    if path is None:break
                    g.play(path);prefix.append(path)
                if g.ply!=target or g.result()[0] is not None:continue
                pair=select_candidates(g)
                if pair is None:continue
                children=[];terminal_child=False
                for path in pair:
                    child=copy.deepcopy(g);child.play(path)
                    terminal_child=terminal_child or child.result()[0] is not None
                    children.append(orbit(decision_key(child.board,child.player)))
                # Reject the entire preselected geometric pair, never substitute a second choice.
                if terminal_child:continue
                identity=decision_key(g.board,g.player);root=orbit(identity);new=[root]+children
                if len(set(new))!=3 or any(i in used or i in anchors for i in new):continue
                accepted=dict(group=group,split='train' if group<32 else 'development',bucket='ply'+str(target),player=g.player,ply=g.ply,board=g.board,opening=prefix,root_id=identity,decision_id=identity,orbit=root,anchor=anchor,child_orbits=children,rear_path=pair[0],front_path=pair[1]);break
            if accepted is None:raise ValueError(f'组{group}在96候选内未找到合法新root，不放宽门槛')
            schedule.append(accepted);used.update(new);anchors.add(anchor);atomic(a.out/'schedule_partial.json',schedule)
        atomic(a.out/'schedule.json',schedule);freeze=dict(schedule_sha256=digest(a.out/'schedule.json'),roots=48,generation_queries=generation_queries);atomic(a.out/'schedule_freeze.json',freeze);log('48根冻结后开始评分，不按标签补根。')
        rows=[]
        for row in schedule:
            g=r.Game()
            for path in row['opening']:g.play(path)
            scores={};audit={}
            for depth in (4,5):
                for choice in ('rear','front'):
                    status('固定根深度评分',group=row['group'],depth=depth,choice=choice)
                    score,info=audited_score(g,row[choice+'_path'],depth);scores[f'score{depth}_{choice}']=score;audit[f'depth{depth}_{choice}']=info
            label=label_scores(**scores);rows.append(dict(**row,**scores,label=label,eligible=label!=0,value_weight=0.,depth_audit=audit));atomic(a.out/'raw_scores.json',rows)
        if digest(a.out/'schedule.json')!=freeze['schedule_sha256']:raise ValueError('冻结schedule变化')
        counts={part:{direction:sum(row['split']==part and row['label']==direction for row in rows) for direction in (-1,1)} for part in ('train','development')}
        gates=dict(train_total=sum(counts['train'].values())>=16,development_total=sum(counts['development'].values())>=8,train_both=min(counts['train'].values())>=4,development_both=min(counts['development'].values())>=2)
        atomic(a.out/'dataset.json',dict(profile=r.PROFILE,rows=rows,gates=gates,data_quality_pass=all(gates.values()),counts=counts,schedule_sha256=freeze['schedule_sha256']));log('标签冻结；'+json.dumps(counts,ensure_ascii=False));status('complete',data_quality_pass=all(gates.values()))
    except BaseException:status('failed',error=traceback.format_exc());raise


if __name__=='__main__':main()
