"""冻结全新对称隔离开发集后，一次广开局池的回放投影对照。"""
import argparse,copy,fcntl,hashlib,json,os,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
import rules121 as r
from az121 import Net121,encode,action_id
from curriculum121 import atomic,opening,replay,batch
from opening121 import OpeningBook,key
from opening121_split_audit import orbit
from train_opening121 import book_samples,mastery
from cap_opening121_replay import load_data,policy_gradients,norm_of,dot_of
from ablate_opening121_hard32 import terms
from project_opening121_replay import projected_objective
from corrected121_decisions import ConsistentExpert,score_action
from unfreeze121_trial import metrics
from checkpoint121_save import REGISTRY,digest
from history121_exclusions import collect_history,resolve_legacy_endgames
SOURCE=Path('nn_checkpoints/experiments/cap_opening121_replay_20261105')
SEED=111051


def generate_schedule(excluded,count=32,max_attempts=4096):
    """每分支两个连续根，先冻结再查询教师；不读取NN。"""
    used=set(excluded);anchors=set();schedule=[];rejects={}
    for attempt in range(max_attempts):
        stage=7+len(schedule)%4;length=2*(stage-1);prefix=opening(SEED+attempt*1000003,length+1)
        if len(prefix)!=length+1:rejects['terminal_short']=rejects.get('terminal_short',0)+1;continue
        anchor=replay(prefix[:6]);anchor_id=orbit(key(anchor.board,anchor.player))
        if anchor_id in used or anchor_id in anchors:rejects['old_or_duplicate_branch']=rejects.get('old_or_duplicate_branch',0)+1;continue
        games=[replay(prefix[:length]),replay(prefix)];ids=[orbit(key(g.board,g.player)) for g in games]
        if any(i in used for i in ids) or len(set(ids))!=2:rejects['old_or_duplicate_root']=rejects.get('old_or_duplicate_root',0)+1;continue
        if any(g.result()[0] is not None for g in games):rejects['terminal_root']=rejects.get('terminal_root',0)+1;continue
        schedule.append(dict(group=len(schedule),seed=SEED+attempt*1000003,own_move=stage,anchor=anchor_id,cases=[dict(player=g.player,ply=g.ply,prefix=prefix[:g.ply],decision_id=key(g.board,g.player),orbit=ids[i]) for i,g in enumerate(games)]));anchors.add(anchor_id);used.update(ids)
        if len(schedule)==count:return schedule,rejects
    raise ValueError(f'固定候选预算未集齐{count}分支；不放宽隔离')


@torch.no_grad()
def chosen_actions(model,rows):
    samples=[]
    for row in rows:
        g=replay(row['prefix']);actions=[action_id(s,p[-1],g.player,True) for s,p in r.get_all_moves(g.board,g.player)];samples.append(dict(x=encode(g.board,g.player,g.ply,g.repetitions,canonical=True),actions=actions,target=0,z=0.))
    x,*_=batch(samples);logits,_=model(x);out=[]
    for row,scores,s in zip(rows,logits,samples):
        selected=s['actions'][int(scores[s['actions']].argmax())];start,end=divmod(selected,121)
        if row['player']==2:start,end=120-start,120-end
        g=replay(row['prefix']);path=next(p for p in r.legal_paths(g.board,start) if p[-1]==end);out.append(path)
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--history-manifest',type=Path);args=ap.parse_args();torch.set_num_threads(1)
    previous=json.loads((SOURCE/'config.json').read_text());modelpath=Path(previous['model']);registry=digest(REGISTRY)
    if registry!=previous['registry_sha256'] or digest(modelpath)!=previous['model_sha256']:raise ValueError('现役或初值变化')
    args.out.mkdir(parents=True,exist_ok=False);lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic();queries=0
    def status(stage,**kw):
        atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),seconds=time.monotonic()-started,expert_queries=queries,**kw))
        if time.monotonic()-started>3600 and stage not in ('complete','failed'):raise TimeoutError('一小时开发预算耗尽')
    def log(text):
        with (args.out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-started:.1f}秒：{text}\n')
    sources=['fresh121_projection_trial.py','project_opening121_replay.py','history121_exclusions.py','opening121_split_audit.py','cap_opening121_replay.py','ablate_opening121_hard32.py','fit_opening121_hard32.py','opening121_ratios.py','train_opening121.py','opening121.py','opening121_book.json','az121.py','rules121.py','curriculum121.py','canonical121_trial.py','cycle121_targeted.py','generalization121_curve.py','unfreeze121_trial.py','checkpoint121_save.py','corrected121_decisions.py','engine_clysto121.py','engine_c121.py','tests/test_opening121_projection.py','tests/test_fresh121_development.py','diagnostics/plot121_fresh_projection.py','diagnostics/plot_style_zh.py']
    import clysto121_native
    sources.extend([Path(clysto121_native.__file__).name,'clysto121_generated/core.c'])
    for name in sources:
        dest=args.out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
    shutil.copytree(SOURCE/'data_sources',args.out/'data_sources')
    cfg=dict(profile=r.PROFILE,model=str(modelpath),model_sha256=digest(modelpath),registry_sha256=registry,source_sha256={n:digest(n) for n in sources},seed=SEED,arms=['control','projected'],steps=2400,lr=3e-5,root_seed=111052,ordinary_seed=111053,max_seconds=3600,root_pool=565,new_development_roots=64,new_development_groups=32,max_candidate_attempts=4096,teacher_depth=5,max_teacher_queries=320,
        intervention='仅负内积时将已加权回放梯度投影到当前专项梯度正交面；detached c=max(0,-dot/||root||²)，loss=(1+c)*root+replay+aux，无范数限幅。',
        development='固定随机合法前缀，不按NN错误选择；镜像+转方全轨道排除历史，冻结64根和教师建议后仅最终三模型评一次。原开局库与固定深度专家为不同教师，主指标为专家分差/差棋率。',
        gates='projected在新64例gap>=4差棋数至少比control少4，且不多于v3；平均非终局分差不高于两者；旧CE、旧value、普通CE各<=v3+.05。通过才下一阶段独立对战，不直接晋升。',
        confirmation='若开发门槛通过，固定projected最终检查点，另生成全新全轨道隔离开局，120局对v3+候选/v3各40局对C；双方NN32、C30ms深度10、无外部开局库，配对p<=.01且对C积分不降，完整审计后才晋升。')
    atomic(args.out/'config.json',cfg)
    try:
        cfg['data_sha256']={name:digest(args.out/'data_sources'/name) for name in previous['data_sha256']}
        if cfg['data_sha256']!=previous['data_sha256']:raise ValueError('原训练数据哈希变化')
        atomic(args.out/'config.json',cfg)
        status('历史隔离清单')
        if args.history_manifest:
            history=json.loads(args.history_manifest.read_text());history_root=Path('nn_checkpoints/experiments')
            for name,sha in history['source_sha256'].items():
                if digest(history_root/name)!=sha:raise ValueError('历史清单来源文件变化：'+name)
            actual={str(p.relative_to(history_root)) for p in history_root.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.pt') and 'source' not in p.relative_to(history_root).parts and args.out.resolve() not in p.resolve().parents}
            if actual!=set(history['source_sha256']):raise ValueError('历史清单之后新增或缺少历史数据文件，重新扫描')
            cfg['history_manifest_sha256']=digest(args.history_manifest);atomic(args.out/'config.json',cfg)
        else:history=resolve_legacy_endgames(collect_history('nn_checkpoints/experiments',exclude=[args.out]),'nn_checkpoints/experiments')
        atomic(args.out/'history_exclusions.json',history)
        if history.get('incomplete'):raise ValueError('历史覆盖有未解决项，先审查history_exclusions.json中的incomplete')
        fixed,old,normal,sets=load_data(args.out/'data_sources')
        if len(sets['train'])!=565:raise ValueError('广开局训练池应为565')
        excluded=set(history['orbits']);excluded.update(orbit(s['decision_id']) for pool in sets.values() for s in pool)
        schedule,rejected=generate_schedule(excluded);atomic(args.out/'schedule.json',schedule);atomic(args.out/'generation_audit.json',dict(groups=len(schedule),roots=64,rejected=rejected,historical_orbits=len(history['orbits']),excluded_orbits=len(excluded),root_overlap=0,anchor_overlap=0,schedule_sha256=digest(args.out/'schedule.json')))
        log(f'开发64根/32分支冻结；历史轨道{len(history["orbits"])}，候选拒绝{rejected}。')
        rows=[dict(group=group['group'],own_move=group['own_move'],**row) for group in schedule for row in group['cases']];teacher=ConsistentExpert(5);ConsistentExpert.failure_dir=args.out;labels=[]
        for i,row in enumerate(rows):
            status('冻结教师标签',completed=i,total=64);g=replay(row['prefix']);path,info=teacher.think(g.board,g.player,g.ply,g.repetitions);queries+=1
            if path is None:raise ValueError('活局面无教师走法')
            child=copy.deepcopy(g);child.play(path)
            if child.result()[0] is not None:raise ValueError('早期开局教师动作出现终局，暂停均值定义审查')
            q=score_action(g,path,5);queries+=1
            if q!=info['score']:atomic(args.out/'teacher_inconsistency.json',dict(case=row,preferred=path,root_score=info['score'],forced_score=q));raise ValueError('教师root/child分数不一致')
            labels.append(dict(**row,preferred=path,expert_score=info['score'],forced_preferred_score=q,depth=info['depth_reached'],value_weight=0.));atomic(args.out/'development.json',labels)
        freeze=dict(schedule_sha256=digest(args.out/'schedule.json'),labels_sha256=digest(args.out/'development.json'),roots=64,groups=32,labels_used_in_training=False);atomic(args.out/'development_freeze.json',freeze)
        base=Net121.load(modelpath);base.requires_grad_(False)
        def measure(m):return dict(book_train=mastery(m,sets['train']),old_book_validation=mastery(m,sets['validation']),old_fixed32=mastery(m,fixed),retention=metrics(m,old['validation']),normal=metrics(m,normal['validation']))
        before=measure(base);atomic(args.out/'baseline.json',before);final={};audits={};log('教师建议标签冻结；开始广565训练池两组2400更新，新开发不看中途。')
        for arm in ('control','projected'):
            m=copy.deepcopy(base);m.requires_grad_(True);params=[p for n,p in m.named_parameters() if not n.startswith('value.')];opt=torch.optim.AdamW(m.parameters(),lr=3e-5,weight_decay=1e-4);rr=np.random.default_rng(111052);ro=np.random.default_rng(111053);seq=hashlib.sha256();curve=[];updates=[];probes=[]
            for step in range(2401):
                if step%200==0:
                    curve.append(dict(step=step,**measure(m)));atomic(args.out/f'{arm}_curve.json',curve);atomic(args.out/f'{arm}_updates.json',updates);atomic(args.out/f'{arm}_probes.json',probes);status('广开局回放投影',arm=arm,step=step)
                    if step%600==0:m.save(args.out/f'{arm}_step{step}.pt')
                    if step==2400:break
                ri=rr.integers(len(sets['train']),size=4);oi=ro.integers(len(old['train']),size=6);ni=ro.integers(len(normal['train']),size=6);seq.update(ri.tobytes()+oi.tobytes()+ni.tobytes());roots=[sets['train'][int(i)] for i in ri];ordinary=[old['train'][int(i)] for i in oi]+[normal['train'][int(i)] for i in ni];root,rep,aux=terms(m,base,roots,ordinary);loss,ga,gb,c=projected_objective(root,rep,aux,params,arm=='projected');rawdot=dot_of(ga,gb);aa=dot_of(ga,ga);postdot=rawdot+c*aa
                if not bool(torch.isfinite(loss)):raise ValueError('非有限投影损失')
                tolerance=1e-5*float(norm_of(ga)*norm_of(gb))+1e-7
                if arm=='projected' and float(postdot)<-tolerance:raise ValueError('投影后仍与根梯度负内积')
                if step%200==0:saved=[p.detach().clone() for p in params]
                opt.zero_grad();loss.backward();totaldot=float(dot_of(tuple(torch.zeros_like(p) if p.grad is None else p.grad for p in params),ga));gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True));opt.step();updates.append(dict(step=step,coefficient=float(c),raw_dot=float(rawdot),projected_dot=float(postdot),total_dot_root=totaldot,preclip_norm=gn,root_ce=float(root.detach()),replay_ce=float(rep.detach()),aux=float(aux.detach())))
                if step%200==0:
                    delta=[p.detach()-oldp for p,oldp in zip(params,saved)];probes.append(dict(step=step,adam_update_norm=float(norm_of(delta)),root_descent_dot=float(-dot_of(delta,ga)),root_descent_cosine=float(-dot_of(delta,ga)/(norm_of(delta)*norm_of(ga)).clamp_min(1e-12))))
            final[arm]=curve[-1];audits[arm]=dict(sequence_sha256=seq.hexdigest(),root_exposures=9600,ordinary_exposures=28800,projection_active=sum(u['coefficient']>0 for u in updates));atomic(args.out/'sampling_audit.json',audits);log(arm+f'结束，训练库招命中{final[arm]["book_train"]["top1_in_book"]:.1%}，旧CE {final[arm]["retention"]["policy_ce"]:.3f}')
        if len({a['sequence_sha256'] for a in audits.values()})!=1:raise ValueError('抽样不一致')
        if digest(args.out/'development.json')!=freeze['labels_sha256'] or digest(args.out/'schedule.json')!=freeze['schedule_sha256']:raise ValueError('冻结开发数据变化')
        decisions={};quality={};cache={}
        for i,row in enumerate(labels):cache[(i,row['preferred'][0],row['preferred'][-1])]=row['expert_score']
        for name,path in [('v3',modelpath),('control',args.out/'control_step2400.pt'),('projected',args.out/'projected_step2400.pt')]:
            m=Net121.load(path);choices=chosen_actions(m,labels);records=[]
            for i,(row,choice) in enumerate(zip(labels,choices)):
                status('最终新开发评测',model=name,completed=i,total=64);cachekey=(i,choice[0],choice[-1]);g=replay(row['prefix'])
                if cachekey not in cache:cache[cachekey]=score_action(g,choice,5);queries+=1
                if queries>320:raise ValueError('教师查询预算超限')
                child=copy.deepcopy(g);child.play(choice)
                if child.result()[0] is not None:raise ValueError('早期开局出现终局动作，暂停审查均值定义')
                q=cache[cachekey];gap=row['expert_score']-q
                if gap<0:atomic(args.out/'development_score_inconsistency.json',dict(case=row,choice=choice,chosen_score=q,model=name));raise ValueError('实际首选得分超过教师根分数，拒绝标签而非截断')
                records.append(dict(group=row['group'],player=row['player'],own_move=row['own_move'],decision_id=row['decision_id'],choice=choice,score=q,expert_score=row['expert_score'],gap=gap,teacher_equal=gap==0,bad=gap>=4))
            decisions[name]=records;quality[name]=dict(n=64,bad=sum(x['bad'] for x in records),equal=sum(x['teacher_equal'] for x in records),mean_gap=float(np.mean([x['gap'] for x in records])),median_gap=float(np.median([x['gap'] for x in records])));atomic(args.out/'development_choices.json',decisions);log(name+f'新开发差棋{quality[name]["bad"]}/64，平均分差{quality[name]["mean_gap"]:.3f}')
        q=quality['projected'];gates=dict(new_bad_gain=q['bad']<=quality['control']['bad']-4,new_bad_vs_v3=q['bad']<=quality['v3']['bad'],new_mean_gap=q['mean_gap']<=min(quality['control']['mean_gap'],quality['v3']['mean_gap']),old_policy=final['projected']['retention']['policy_ce']<=before['retention']['policy_ce']+.05,old_value=final['projected']['retention']['value_mse']<=before['retention']['value_mse']+.05,normal_policy=final['projected']['normal']['policy_ce']<=before['normal']['policy_ce']+.05)
        if digest(REGISTRY)!=registry or digest(modelpath)!=cfg['model_sha256']:raise ValueError('冠军变化')
        result=dict(baseline=before,final=final,new_development=quality,gates=gates,development_pass=all(gates.values()),all_development_actions_nonterminal=True,promoted=False,arena_run=False,expert_queries=queries,seconds=time.monotonic()-started,freeze=freeze);atomic(args.out/'result.json',result);status('complete');log('开发结束；'+('通过，固定候选后可继续预声明整局确认。' if all(gates.values()) else '未通过，停止该配置，不追加调参，不晋升。'))
    except BaseException:status('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
