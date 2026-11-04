"""同200决策、同CNN、同预算：单偏好与多候选专家监督的有限对照。"""
import argparse,copy,fcntl,hashlib,json,os,shutil,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import rules121 as r
from az121 import Net121,Neural121,encode,action_id
from curriculum121 import atomic,batch,replay,samples_from_game,finish
from canonical121_trial import canonical_sample
from corrected121_decisions import ConsistentExpert,score_action,root_metrics
from diagnostics.fit121_objectives import objective_terms
from cycle121_targeted import expert_samples
from generalization121_curve import remove_validation_duplicates
from unfreeze121_trial import metrics
from checkpoint121_save import digest,REGISTRY,verify

PARENT=Path('nn_checkpoints/experiments/generalization121_200_20261101')
SEEDS=(111054,111055)

def multi_loss(logits,samples):
    ce=[];rank=[]
    for row,s in zip(logits,samples):
        legal=row[s['actions']];scores=s['candidate_scores'];best=max(scores.values())
        good=[int(a) for a,v in scores.items() if v==best]
        ce.append(torch.logsumexp(legal,0)-torch.logsumexp(row[good],0))
        pairs=[F.relu(.5-row[int(a)]+row[int(b)]) for a,va in scores.items() for b,vb in scores.items() if va-vb>=4]
        rank.append(torch.stack(pairs).mean() if pairs else row.sum()*0)
    return torch.stack(ce).mean(),torch.stack(rank).mean()

def witness(g,action):
    start,end=divmod(int(action),121)
    if g.player==2:start,end=120-start,120-end
    return next(p for s,p in r.get_all_moves(g.board,g.player) if s==start and p[-1]==end)

def development_gate(final,before):
    details={}
    for seed in SEEDS:
        old=final[f'old_{seed}'];new=final[f'multi_{seed}'];a=old['validation_actual'];b=new['validation_actual']
        details[str(seed)]=dict(regret=b['mean_regret']<=a['mean_regret']-1.,bad_moves=b['bad_fraction']<=a['bad_fraction']-.1+1e-9,
            equal_best=b['equal_best_rate']>=a['equal_best_rate'],retention=new['retention']['policy_ce']<=min(old['retention']['policy_ce']+.03,before['retention']['policy_ce']+.05))
    return all(all(d.values()) for d in details.values()),details

def run(out):
    torch.set_num_threads(1);saved=verify(Path('nn_checkpoints/saves/v3_before_unfreeze_20261030'))
    if digest(REGISTRY)!=saved['registry_sha256']:raise ValueError('现役不同于存档，拒绝覆盖')
    parent_config=json.loads((PARENT/'config.json').read_text());baseline_path=Path(parent_config['model']);selector_path=PARENT/'expanded160_step1600.pt'
    out.mkdir(parents=True,exist_ok=False);lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic();queries=0
    def status(stage,**kw):
        atomic(out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-started,expert_queries=queries,**kw))
        if stage not in ('failed','complete') and time.monotonic()-started>10800:raise TimeoutError('三小时预算耗尽，保留全部中间产物')
    def log(s):
        with (out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-started:.1f}秒：{s}\n')
    def record(name,row):
        with (out/(name+'.jsonl')).open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    import clysto121_native,ccore121
    names=['multicandidate121_trial.py','tests/test_multicandidate121.py','diagnostics/plot121_multicandidate.py','diagnostics/plot_style_zh.py','diagnostics/fit121_objectives.py','corrected121_decisions.py','az121.py','curriculum121.py','canonical121_trial.py','generalization121_curve.py','cycle121_targeted.py','unfreeze121_trial.py','rules121.py','engine_c121.py','engine_clysto121.py','iterate121.py','confirm121_unfrozen.py','checkpoint121_save.py','neural121_registry.py','clysto121_generated/core.c','clysto121_generated/manifest.json',Path(clysto121_native.__file__).name,Path(ccore121.__file__).name]
    for name in ('clysto121_generated/core.c',Path(clysto121_native.__file__).name,'rules121.py','engine_clysto121.py','engine_c121.py'):
        if digest(name)!=parent_config['source_sha256'][name]:raise ValueError('原始标签老师/规则实现已变化：'+name)
    config=dict(compute_scope='相同更新次数和样本暴露预算；多候选损失额外计算，不声称墙钟算力相同。',profile=r.PROFILE,encoding=parent_config['encoding'],model=str(baseline_path.resolve()),model_sha256=digest(baseline_path),registry_sha256=digest(REGISTRY),selector=str(selector_path.resolve()),selector_sha256=digest(selector_path),
        training='CNN多候选固定深度专家策略蒸馏，不是纯自对弈；老师为Clysto评估派生121版本。',seeds=list(SEEDS),primary_seed=SEEDS[0],steps=1600,learning_rate=.00003,root_batch=4,ordinary_batch=12,
        selection='固定expanded160最终模型前3合法动作，加专家建议和原坏动作；最高5候选。相同200数据/原开局分组，160训练40验证；验证标签不参与训练。',
        loss='旧：唯一专家CE+.5间隔原偏好；新：同最优分动作集合概率CE+所有分差>=4候选对的.5间隔排序。根CE .25、排序 .25，普通CE .75、真实value 1、KL .1、value保留 .1。',
        gate='两抽样种子均：验证实际top1平均遗憾至少降1分、差>=4比例至少降10个百分点、同最优分比例不降；旧CE不超过控制+.03且不超过v3+.05。仅开发筛选不是棋力证明。固定第一种子候选，不择优种子/检查点。',
        max_seconds=10800,max_new_child_queries=1600,simulations=32,confirmation_games=120,fixed_c_games_per_model=40,independent_gate_p=.01,
        source_sha256={n:digest(n) for n in names},data_sha256={})
    for name in names:
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
    data=out/'data_sources';data.mkdir()
    sources={'decisions.pt':PARENT/'decisions.pt','decisions.jsonl':PARENT/'decisions.jsonl','old_teacher.jsonl':PARENT/'data_sources/old_teacher.jsonl','old_split.json':PARENT/'data_sources/old_split.json','normal_games.jsonl':PARENT/'data_sources/normal_games.jsonl','selector.pt':selector_path}
    for name,path in sources.items():shutil.copyfile(path,data/name);config['data_sha256'][name]=digest(data/name)
    atomic(out/'config.json',config)
    try:
        status('准备补充候选评分');ConsistentExpert.failure_dir=out;baseline=Net121.load(baseline_path);baseline.requires_grad_(False);selector=Net121.load(data/'selector.pt');selector.requires_grad_(False)
        roots=torch.load(data/'decisions.pt',weights_only=True)['sets'];metadata={x['decision_id']:x for x in map(json.loads,(data/'decisions.jsonl').read_text().splitlines())}
        if digest(baseline_path)!=parent_config['model_sha256']:raise ValueError('基线模型与数据来源不符')
        if [len(roots[p]) for p in ('train','validation')]!=[160,40]:raise ValueError('数据数量变化')
        if {metadata[s['decision_id']]['pair'] for s in roots['train']}&{metadata[s['decision_id']]['pair'] for s in roots['validation']}:raise ValueError('开局泄漏')
        cache={};games={}
        for part,samples in roots.items():
            for s in samples:
                row=metadata[s['decision_id']];g=replay(row['opening']);games[s['decision_id']]=g
                if g.player!=row['player'] or not np.array_equal(s['x'],encode(g.board,g.player,g.ply,g.repetitions,canonical=True)):raise ValueError('输入重放不一致')
                actions=[action_id(a,p[-1],g.player,True) for a,p in r.get_all_moves(g.board,g.player)]
                if actions!=s['actions']:raise ValueError('合法动作不一致')
                cache[(s['decision_id'],s['actions'][s['target']])]=row['expert_score'];cache[(s['decision_id'],s['rejected_action'])]=row['rejected_score']
        def score(s,a):
            nonlocal queries
            key=(s['decision_id'],int(a));row=metadata[s['decision_id']]
            if key not in cache:
                if queries>=1600:raise RuntimeError('新评分次数预算耗尽')
                status('专家评分',decision_id=s['decision_id']);value=score_action(games[s['decision_id']],witness(games[s['decision_id']],a));queries+=1
                if value>row['expert_score']:raise ValueError('强制动作分数超过原固定深度全局最佳')
                cache[key]=value;record('score_cache',dict(decision_id=s['decision_id'],action=int(a),score=value))
            return cache[key]
        for index,s in enumerate(roots['train']+roots['validation']):
            with torch.no_grad():xs,*_=batch([s]);logits,_=selector(xs);top=torch.topk(logits[0,s['actions']],min(3,len(s['actions']))).indices.tolist()
            candidates=list(dict.fromkeys([s['actions'][i] for i in top]+[s['actions'][s['target']],s['rejected_action']]))
            s['candidate_scores']={a:score(s,a) for a in candidates};record('candidate_labels',dict(decision_id=s['decision_id'],split=metadata[s['decision_id']]['split'],scores=s['candidate_scores']))
            status('补充候选评分',completed=index+1,total=200)
        for part in ('train','validation'):
            for sample in roots[part][:2]:
                g=games[sample['decision_id']];path,info=ConsistentExpert().think(g.board,g.player,g.ply,g.repetitions)
                row=metadata[sample['decision_id']]
                if info['score']!=row['expert_score'] or (path[0],path[-1])!=(row['preferred'][0],row['preferred'][-1]):raise ValueError('原专家标签重复查询不一致')
        torch.save(dict(sets=roots),data/'multicandidate.pt');config['data_sha256']['multicandidate.pt']=digest(data/'multicandidate.pt');atomic(out/'config.json',config)
        old={'train':[],'validation':[]};normal={'train':[],'validation':[]};split=json.loads((data/'old_split.json').read_text())
        for line in (data/'old_teacher.jsonl').read_text().splitlines():
            row=json.loads(line);part=next((k for k in old if row['group'] in split[k]),None)
            if part:old[part].extend({**canonical_sample(s),'value_weight':1.} for s in samples_from_game(row))
        for line in (data/'normal_games.jsonl').read_text().splitlines():
            row=json.loads(line);normal[row['split']].extend(expert_samples(row))
        for pool in (old,normal):pool['train'],_=remove_validation_duplicates(pool['train'],roots['validation'])
        def measure(model):return dict(train=root_metrics(model,roots['train']),validation=root_metrics(model,roots['validation']),retention=metrics(model,old['validation']),normal=metrics(model,normal['validation']))
        before=measure(baseline);atomic(out/'baseline.json',before);final={};audits={};log('200局面候选已补固定深度5评分。开始两种监督×两抽样种子，同v3初始化、同普通回放和根序列，各1600步。')
        for seed in SEEDS:
            for arm in ('old','multi'):
                name=f'{arm}_{seed}';model=copy.deepcopy(baseline);model.requires_grad_(True);opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.0001)
                rr=np.random.default_rng(seed+1);ro=np.random.default_rng(seed+2);sequence=hashlib.sha256();history=[]
                for step in range(1601):
                    if step%100==0:
                        history.append(dict(step=step,**measure(model)));atomic(out/(name+'_curve.json'),history);status('CNN监督对照训练',arm=name,step=step,total=1600)
                        if step%400==0:model.save(out/(name+f'_step{step}.pt'))
                        if step==1600:break
                    ri=rr.integers(160,size=4);oi=ro.integers(len(old['train']),size=6);ni=ro.integers(len(normal['train']),size=6);sequence.update(ri.tobytes()+oi.tobytes()+ni.tobytes())
                    rs=[roots['train'][int(i)] for i in ri];ordinary=[old['train'][int(i)] for i in oi]+[normal['train'][int(i)] for i in ni]
                    loss,terms=objective_terms(model,baseline,rs,ordinary,'full_objective')
                    if arm=='multi':
                        xs,*_=batch(rs);logits,_=model(xs);ce,rank=multi_loss(logits,rs);loss=loss-terms['root_ce']-terms['rank']+.25*ce+.25*rank
                    if not torch.isfinite(loss):raise ValueError('训练损失非有限数')
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                audits[name]=dict(sequence_sha256=sequence.hexdigest(),root_exposures=6400,ordinary_exposures=19200,updates=1600);atomic(out/'sampling_audit.json',audits);final[name]=history[-1];log(name+'训练完成')
            if audits[f'old_{seed}']!=audits[f'multi_{seed}']:raise ValueError('对照样本序列不同')
        # Only fixed final models: evaluate actual legal top1 even if it was outside the scored candidate pool.
        for name in final:
            model=Net121.load(out/(name+'_step1600.pt'))
            for part in ('train','validation'):
                rows=[]
                for s in roots[part]:
                    with torch.no_grad():xs,*_=batch([s]);logits,_=model(xs);a=s['actions'][int(logits[0,s['actions']].argmax())]
                    regret=metadata[s['decision_id']]['expert_score']-score(s,a);rows.append(dict(decision_id=s['decision_id'],action=a,regret=regret))
                atomic(out/(name+'_'+part+'_actions.json'),rows);final[name][part+'_actual']=dict(n=len(rows),mean_regret=sum(x['regret'] for x in rows)/len(rows),bad_fraction=sum(x['regret']>=4 for x in rows)/len(rows),equal_best_rate=sum(x['regret']==0 for x in rows)/len(rows))
        passed,details=development_gate(final,before);atomic(out/'development_result.json',dict(final=final,passed=passed,gate=details))
        result=dict(final=final,development_passed=passed,development_gate=details,qualified=False,promoted=False,arena_run=False)
        if passed:
            from confirm121_unfrozen import fresh_opening,seen_openings
            from engine_c121 import CSearch121
            from iterate121 import summary,paired_gate
            used,_=seen_openings('nn_checkpoints/experiments');schedule={stage:[fresh_opening(111054000+offset+i*97,4+2*(i%4),used) for i in range(n)] for stage,n,offset in [('confirmation',60,0),('fixed_c',20,10000)]};atomic(out/'arena_schedule.json',schedule)
            candidate_path=out/f'multi_{SEEDS[0]}_step1600.pt';shutil.copyfile(candidate_path,out/'candidate.pt');candidate=Net121.load(out/'candidate.pt');matches=[];fixed={}
            for stage,model,opponent in [('confirmation',candidate,baseline),('fixed_c_control',baseline,None),('fixed_c_candidate',candidate,None)]:
                rows=[];prefixes=schedule['confirmation' if stage=='confirmation' else 'fixed_c']
                for pair,prefix in enumerate(prefixes):
                    for side in (1,2):
                        status('独立对战验收',comparison=stage,completed=len(rows),total=len(prefixes)*2)
                        rival=Neural121(opponent,32) if opponent is not None else CSearch121(seconds=.03,max_depth=10)
                        row=finish(replay(prefix),{side:Neural121(model,32),3-side:rival});row.update(pair=pair,side=side,opening=prefix);record(stage,row);rows.append(row)
                if stage=='confirmation':matches=rows
                else:fixed[stage.removeprefix('fixed_c_')]=summary(rows)
            gate=paired_gate(matches);result.update(arena_run=True,confirmation=summary(matches),paired_gate=gate,fixed_c=fixed,qualified=gate['one_sided_p']<=.01 and fixed['candidate']['points']>=fixed['control']['points'])
        if digest(REGISTRY)!=config['registry_sha256'] or digest(baseline_path)!=config['model_sha256']:raise ValueError('现役或基线并发改变')
        for name,sha in config['source_sha256'].items():
            if digest(out/'source'/name)!=sha or digest(name)!=sha:raise ValueError('实验代码发生变化：'+name)
        result.update(seconds=time.monotonic()-started,expert_queries=queries);atomic(out/'result.json',result)
        if result['qualified']:
            from neural121_registry import register,load_registered
            from checkpoint121_save import restore
            try:
                entry=register(out,'v4_multicandidate');engine=load_registered()
                for p in (1,2):
                    path,info=engine.think(r.init_board(),p)
                    if not r.valid_path(r.init_board(),p,path) or info['root_visits']!=32:raise ValueError('登记后烟测失败')
                result['promoted']=True;atomic(out/'adoption.json',dict(adopted=True,version=entry['version']))
            except BaseException:
                active=json.loads(REGISTRY.read_text())
                if active.get('version')=='v4_multicandidate' and active.get('sha256')==digest(out/'candidate.pt'):restore(Path('nn_checkpoints/saves/v3_before_unfreeze_20261030'))
                raise
        atomic(out/'result.json',result)
        plot=subprocess.run(['/Library/Frameworks/Python.framework/Versions/3.10/bin/python3','diagnostics/plot121_multicandidate.py',str(out)],capture_output=True,text=True);atomic(out/'plot_status.json',dict(exit_code=plot.returncode,stderr=plot.stderr))
        if plot.returncode:raise RuntimeError('简体中文图表渲染失败')
        lines=['# 多候选专家监督对照报告','',f'开发筛选：{"通过" if passed else "未通过"}；独立对战：{result["arena_run"]}；升级：{result["promoted"]}。','200局面仍为160训练/40验证，51开局组；这是专家搜索评分蒸馏，不是纯自对弈。根没有胜负标签。','|方案|验证平均遗憾|验证差棋比例（分差至少4）|验证同最优分比例|旧策略交叉熵|','|---|---:|---:|---:|---:|']
        for name,v in final.items():
            a=v['validation_actual'];label=('原单偏好' if name.startswith('old') else '多候选')+'／种子'+name.split('_')[-1];lines.append(f'|{label}|{a["mean_regret"]:.3f}|{a["bad_fraction"]:.1%}|{a["equal_best_rate"]:.1%}|{v["retention"]["policy_ce"]:.3f}|')
        lines+=['','40验证局面来自10开局组且已用于先前诊断，本轮是开发比较，不声称全新独立棋力测试。固定深度同分只表示该专家预算同分，不代表真实必胜最优。','固定最终1600步、预选第一种子；不按验证选择检查点或种子。未通过则停止，不追加训练补考；只有全新开局独立验收及完整棋谱哈希审计通过才升级。','完整中文曲线：监督对照曲线.png；实际走法评分、候选标签、采样哈希、源码快照及中文日志均保留。']
        (out/'REPORT.md').write_text('\n'.join(lines));log('本轮完成：开发筛选'+str(passed)+'；升级'+str(result['promoted']));status('complete',development_passed=passed,promoted=result['promoted'])
    except BaseException:
        status('failed',error=traceback.format_exc());raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();run(args.out)
