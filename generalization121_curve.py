"""200 fresh distinct expert decisions / >=50 opening groups, nested learning curves.
Small72 vs expanded160, identical initialization/steps/ordinary samples. No promotion.
"""
import argparse,copy,csv,fcntl,hashlib,json,os,random,shutil,subprocess,time,traceback
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import rules121 as r
from az121 import Net121,Neural121
from curriculum121 import atomic,replay,batch,samples_from_game
from canonical121_trial import canonical_sample
from corrected121_decisions import ConsistentExpert,score_action,eligible_decision,decision_sample,preference_loss,root_metrics
from unfreeze121_trial import objective,metrics
from cycle121_targeted import expert_samples
from checkpoint121_save import verify,digest,REGISTRY
from confirm121_unfrozen import fresh_opening,seen_openings


def position_key(g):return (tuple(g.board),g.player)
def input_digest(sample):return hashlib.sha256(np.asarray(sample['x'],dtype=np.float32).tobytes()).hexdigest()

def nested_subset(roots,metadata,limit=72,seed=111052):
    by_group=defaultdict(list);lookup={r['decision_id']:r for r in metadata}
    for sample in roots:by_group[lookup[sample['decision_id']]['pair']].append(sample)
    groups=sorted(by_group);random.Random(seed).shuffle(groups);chosen=[]
    for group in groups:
        chosen.extend(by_group[group])
        if len(chosen)>=limit:return chosen[:limit]
    raise ValueError('not enough roots for nested control')


def remove_validation_duplicates(samples,validation):
    forbidden={input_digest(s) for s in validation};kept=[s for s in samples if input_digest(s) not in forbidden]
    return kept,len(samples)-len(kept)


def training_batch(roots,old,normal,root_rng,ordinary_rng):
    return [roots[int(i)] for i in root_rng.integers(len(roots),size=4)]+[old[int(i)] for i in ordinary_rng.integers(len(old),size=6)]+[normal[int(i)] for i in ordinary_rng.integers(len(normal),size=6)]


@torch.no_grad()
def predict_preferences(model,samples):
    prediction=[]
    for sample in samples:
        xs,ids,mask,target,_=batch([sample]);logits,_=model(xs);legal=logits.gather(1,ids)
        good=sample['actions'][sample['target']];bad=sample['rejected_action']
        prediction.append(dict(decision_id=sample['decision_id'],ce=float(F.cross_entropy(legal,target)),correct=int(legal.argmax(1))==sample['target'],preferred=bool(logits[0,good]>logits[0,bad])))
    return prediction


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=111052);args=ap.parse_args()
    torch.set_num_threads(1);torch.manual_seed(args.seed);saved_dir=Path('nn_checkpoints/saves/v3_before_unfreeze_20261030');saved=verify(saved_dir)
    if digest(REGISTRY)!=saved['registry_sha256']:raise ValueError('champion differs from saved v3; preserve it')
    baseline_path=saved_dir/'model.pt';baseline=Net121.load(baseline_path);baseline.requires_grad_(False)
    args.out.mkdir(parents=True,exist_ok=False);lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);began=time.monotonic()
    def status(stage,**kw):
        atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-began,**kw))
        if stage not in ('complete','failed') and time.monotonic()-began>10800:raise TimeoutError('three-hour budget exhausted; preserve partial data')
    def log(message):
        with (args.out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-began:.1f}秒：{message}\n')
    def record(name,row):
        with (args.out/(name+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
    names=['generalization121_curve.py','diagnostics/plot121_generalization.py','corrected121_decisions.py','az121.py','rules121.py','curriculum121.py','canonical121_trial.py','cycle121_targeted.py','unfreeze121_trial.py','engine_clysto121.py','engine_c121.py','checkpoint121_save.py','confirm121_unfrozen.py','clysto121_generated/core.c','clysto121_generated/manifest.json']
    import clysto121_native
    names.append(Path(clysto121_native.__file__).name)
    config=dict(profile=r.PROFILE,encoding=baseline.encoding,seed=args.seed,model=str(baseline_path.resolve()),model_sha256=digest(baseline_path),registry_sha256=digest(REGISTRY),
        root_targets=dict(train=160,validation=40),minimum_opening_groups=50,maximum_opening_groups=64,maximum_decisions_per_game=2,minimum_spacing=6,early_window=40,
        teacher_depth=5,expert_safety_timeout=120,minimum_teacher_score_gap=4,maximum_timeout_prefixes=8,
        collection='Prefix-only early decision data. Nonterminal collection stops are NOT draws or invented winners. Accepted fixed-depth score preferences only; root value_weight=0.',
        arms=['small72','expanded160'],steps_per_arm=1600,learning_rate=.00003,root_item_quota=.25,policy_kl=.1,value_preservation=.1,preference_loss_weight=.25,
        control='Nested fresh training subset vs full fresh training pool; same initialization and steps; separate RNG ensures identical ordinary replay across arms',
        validation='Whole opening groups held out, global board+turn root dedup; remove exact validation-input duplicates from ordinary fine-tuning samples. Difficult selected decision domain, not overall move accuracy.',
        checkpoint='Primary comparison at fixed step1600. Save curves/checkpoints; never pick best validation step or deploy based on these curves.',
        scope='Expert score distillation / early-decision generalization diagnosis only. No new arena, promotion, or automatic1000 expansion.',
        plot_python='/Library/Frameworks/Python.framework/Versions/3.10/bin/python3',max_seconds=10800,source_sha256={name:digest(name) for name in names})
    for name in names:
        target=args.out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,target)
    data=args.out/'data_sources';data.mkdir();config['data_sha256']={}
    old=Path('nn_checkpoints/experiments/imitate_clysto121_20261030');normal_source=Path('nn_checkpoints/experiments/corrected121_early_decisions_20261031_resume1')
    for name,path in {'old_teacher.jsonl':old/'teacher_games.jsonl','old_split.json':old/'split.json','normal_games.jsonl':normal_source/'normal_games.jsonl'}.items():
        shutil.copyfile(path,data/name);config['data_sha256'][name]=digest(data/name)
    atomic(args.out/'config.json',config)
    try:
        ConsistentExpert.failure_dir=args.out;status('preregister')
        historic,sources=seen_openings('nn_checkpoints/experiments');used=set(historic);root_seen=set(historic)
        schedule=[fresh_opening(args.seed+100000+i,4+(i%5)*2,used) for i in range(64)]
        split={i:('validation' if i%5==4 else 'train') for i in range(64)}
        atomic(args.out/'schedule.json',schedule);atomic(args.out/'split.json',split);atomic(args.out/'opening_exclusions.json',sources)
        for name in ('schedule.json','split.json','opening_exclusions.json'):
            shutil.copyfile(args.out/name,data/name);config['data_sha256'][name]=digest(data/name)
        atomic(args.out/'config.json',config)
        roots={'train':[],'validation':[]};metadata=[];timeouts=0;expert=ConsistentExpert();student=Neural121(baseline,32)
        log('开始收集全新200去重早期决策，目标160训练/40验证、至少50开局组；只记固定深度专家偏好，不制造局面胜负。')
        for pair,prefix in enumerate(schedule):
            part=split[pair]
            if len(roots[part])>=config['root_targets'][part]:continue
            for nn_side in (1,2):
                if len(roots[part])>=config['root_targets'][part]:break
                g=replay(prefix);history=copy.deepcopy(prefix);paths=[];last=None;count=0;accepted=[];stop='window_or_quota'
                while g.result()[0] is None and g.ply<=40 and count<2 and len(roots[part])<config['root_targets'][part]:
                    status('collect_decisions',pair=pair,nn_side=nn_side,ply=g.ply,train_roots=len(roots['train']),validation_roots=len(roots['validation']),groups=len({s['pair'] for s in metadata}),timeouts=timeouts)
                    atomic(args.out/'collector_state.json',dict(pair=pair,nn_side=nn_side,history=history,accepted_decisions=accepted,root_counts={k:len(v) for k,v in roots.items()}))
                    try:
                        if g.player!=nn_side:path,_=expert.think(g.board,g.player,g.ply,g.repetitions)
                        else:
                            path,_=student.think(g.board,g.player,g.ply,g.repetitions)
                            key=position_key(g)
                            if eligible_decision(g.ply,last,count) and key not in root_seen:
                                preferred,info=expert.think(g.board,g.player,g.ply,g.repetitions)
                                if (preferred[0],preferred[-1])!=(path[0],path[-1]):
                                    rejected_score=score_action(g,path);gap=info['score']-rejected_score
                                    if gap<0:raise ValueError('inconsistent root best score')
                                    if gap>=4:
                                        if score_action(g,preferred)!=info['score']:raise ValueError('root/child score mismatch')
                                        sample=decision_sample(g,preferred,path);sample['decision_id']=len(metadata)
                                        row=dict(decision_id=len(metadata),pair=pair,nn_side=nn_side,split=part,ply=g.ply,player=g.player,opening=copy.deepcopy(history),preferred=preferred,rejected=path,expert_score=info['score'],rejected_score=rejected_score,score_gap=gap)
                                        roots[part].append(sample);metadata.append(row);record('decisions',row);root_seen.add(key);accepted.append(row['decision_id']);count+=1;last=g.ply
                        if path is None:raise ValueError('no move in live collection prefix')
                        g.play(path);paths.append(path);history.append(path)
                    except RuntimeError as exc:
                        if not str(exc).startswith('expert fixed-depth incomplete:'):raise
                        timeouts+=1;stop='expert_timeout';log(f'开局组{pair}的一条前缀搜索超时，拒绝该标签；此前完成的固定深度决策保留，不降深度。')
                        break
                winner,reason=g.result();record('prefixes',dict(pair=pair,nn_side=nn_side,split=part,opening=prefix,paths=paths,plies=g.ply,accepted_decisions=accepted,collection_stop=stop,terminal_winner=winner,terminal_reason=reason))
                if timeouts>8:raise RuntimeError('more than eight timed-out prefixes; stop without weakening label quality')
            log(f'开局组{pair+1}/64处理完成；训练{len(roots["train"])}/160，验证{len(roots["validation"])}/40，实际独立开局组{len({s["pair"] for s in metadata})}。')
            if all(len(roots[k])==n for k,n in config['root_targets'].items()):break
        groups={k:len({x['pair'] for x in metadata if x['split']==k}) for k in roots}
        ready=all(len(roots[k])==n for k,n in config['root_targets'].items()) and len({x['pair'] for x in metadata})>=50
        atomic(args.out/'data_quality.json',dict(ready=ready,roots={k:len(v) for k,v in roots.items()},groups=groups,total_opening_groups=len({x['pair'] for x in metadata}),global_board_turn_dedup=True,median_ply=float(np.median([s['ply'] for s in metadata])) if metadata else None,timeouts=timeouts))
        if not ready:
            result=dict(training_started=False,reason='Fixed collection budget did not provide200 unique roots across50 groups; do not duplicate to fill quotas',promoted=False)
            atomic(args.out/'result.json',result);status('complete',result=result);return
        # Re-query a fixed small balanced subset to validate deterministic expert wiring.
        for part in roots:
            for row in [x for x in metadata if x['split']==part][:4]:
                g=replay(row['opening']);p,info=expert.think(g.board,g.player,g.ply,g.repetitions)
                if (p[0],p[-1])!=(row['preferred'][0],row['preferred'][-1]) or info['score']!=row['expert_score']:raise ValueError('repeat label mismatch')
        for name in ('decisions.jsonl','prefixes.jsonl'):
            shutil.copyfile(args.out/name,data/name);config['data_sha256'][name]=digest(data/name)
        torch.save(dict(encoding=baseline.encoding,sets={k:[{**s,'x':torch.from_numpy(s['x'])} for s in v] for k,v in roots.items()}),args.out/'decisions.pt')
        old_sets={'train':[],'validation':[]};normal={'train':[],'validation':[]};old_split=json.loads((data/'old_split.json').read_text())
        for line in (data/'old_teacher.jsonl').read_text().splitlines():
            row=json.loads(line);part=next((k for k in old_sets if row['group'] in old_split[k]),None)
            if part:old_sets[part].extend({**canonical_sample(s),'value_weight':1.} for s in samples_from_game(row))
        for line in (data/'normal_games.jsonl').read_text().splitlines():
            row=json.loads(line);normal[row['split']].extend(expert_samples(row))
        removed={}
        for name,pool in [('retention',old_sets),('normal',normal)]:pool['train'],removed[name]=remove_validation_duplicates(pool['train'],roots['validation'])
        small=nested_subset(roots['train'],metadata,72,args.seed);arms={'small72':small,'expanded160':roots['train']}
        atomic(args.out/'data.json',dict(root_counts={k:len(v) for k,v in roots.items()},groups=groups,small_subset_ids=[s['decision_id'] for s in small],validation_input_duplicates_removed_from_ordinary_training=removed,
            ordinary_counts={'old':{k:len(v) for k,v in old_sets.items()},'normal':{k:len(v) for k,v in normal.items()}},root_value_labels=0))
        atomic(args.out/'config.json',config);histories={};final_models={}
        log('200局面数据集完成且标签复查通过；开始72子集 vs160完整训练池的同预算学习曲线，不做GUI升级。')
        for arm,train_roots in arms.items():
            model=copy.deepcopy(baseline);model.requires_grad_(True)
            if not all(torch.equal(v,model.state_dict()[k]) for k,v in baseline.state_dict().items()):raise ValueError('initial weights differ across arms')
            opt=torch.optim.AdamW(model.parameters(),lr=.00003,weight_decay=.0001)
            root_rng=np.random.default_rng(args.seed+1);ordinary_rng=np.random.default_rng(args.seed+2);history=[];root_counts=Counter()
            for step in range(1601):
                if step%100==0:
                    measured=dict(step=step,seen_train=root_metrics(model,train_roots),common_train160=root_metrics(model,roots['train']),validation40=root_metrics(model,roots['validation']),
                        retention_validation=metrics(model,old_sets['validation']),normal_validation=metrics(model,normal['validation']))
                    history.append(measured);atomic(args.out/(arm+'_curve.json'),history);status('learning_curve',arm=arm,step=step,total=1600)
                    if step%400==0:model.save(args.out/(arm+'_step'+str(step)+'.pt'))
                    if step==1600:break
                chosen=training_batch(train_roots,old_sets['train'],normal['train'],root_rng,ordinary_rng)
                if sum('rejected_action' in s for s in chosen)!=4:raise ValueError('25% decision quota violated')
                for s in chosen[:4]:root_counts[s['decision_id']]+=1
                xs,ids,mask,target,z=batch(chosen);weight=torch.tensor([s['value_weight'] for s in chosen]);loss=objective(model,baseline,xs,ids,mask,target,z,weight)
                logits,_=model(xs[:4]);loss=loss+.25*preference_loss(logits,chosen[:4])
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            histories[arm]=history;final_models[arm]=model
            atomic(args.out/(arm+'_sampling_audit.json'),dict(root_items=sum(root_counts.values()),ordinary_items=1600*12,root_fraction=.25,distinct_roots=len(root_counts),per_root_exposure=dict(root_counts)))
            log(arm+'1600步完成；'+json.dumps(history[-1],ensure_ascii=False))
        # Per-position heldout diagnostics preserve the grouping, do not select a checkpoint.
        with torch.no_grad():
            predictions={}
            for arm,model in final_models.items():
                predictions[arm]=predict_preferences(model,roots['validation'])
        atomic(args.out/'validation_predictions.json',predictions)
        result=dict(training_started=True,primary_checkpoint=1600,final={arm:h[-1] for arm,h in histories.items()},data=dict(roots=200,groups=len({x['pair'] for x in metadata}),train160=len(roots['train']),validation40=len(roots['validation'])),
            promoted=False,arena_run=False,seconds=time.monotonic()-began,note='Fixed-budget learning-curve diagnosis, not playing-strength evidence; no automatic1000 expansion.')
        if digest(REGISTRY)!=config['registry_sha256'] or digest(baseline_path)!=config['model_sha256']:raise ValueError('champion or baseline changed')
        atomic(args.out/'result.json',result)
        plot=subprocess.run([config['plot_python'],'diagnostics/plot121_generalization.py',str(args.out)],capture_output=True,text=True)
        atomic(args.out/'plot_status.json',dict(exit_code=plot.returncode,stderr=plot.stderr[-2000:]))
        if plot.returncode:log('曲线CSV已保存；图片渲染失败，详情见plot_status.json。')
        lines=['# 200独立决策与学习曲线\n',f'实际200去重棋盘/走棋方，覆盖{result["data"]["groups"]}开局组；160训练、40验证。源开局分组隔离，两个模型从同一v3权重开始，普通训练抽样一致，各1600步。\n',
               '主比较固定在1600步，不按验证最优点挑选。数据是刻意筛选的专家搜索偏好困难决策，准确率不代表所有跳棋走法。根数据没有胜负价值标签。\n',
               '| 训练方案 | 见过的训练根偏好正确率 | 未见验证根偏好正确率 | 验证专家首选一致率 | 旧专家保留CE | 旧价值MSE |',
               '|---|---:|---:|---:|---:|---:|']
        for arm,h in histories.items():
            last=h[-1];lines.append(f'| {arm} | {last["seen_train"]["preference_accuracy"]:.1%} | {last["validation40"]["preference_accuracy"]:.1%} | {last["validation40"]["accuracy"]:.1%} | {last["retention_validation"]["policy_ce"]:.3f} | {last["retention_validation"]["value_mse"]:.3f} |')
        lines += ['\n请结合整条曲线区分：当前目标/预算下训练集尚未拟合，或训练明显改善而验证停滞。不能仅凭此断言网络表达能力上限，不能把重复抽样条目数当作独立经验。',
                  '\n只有40个困难验证根，且同开局内相关，微小差异不构成可靠棋力提升证据。本轮没有对战升级验收，没有更改GUI，没有自动扩到1000。',
                  '\n曲线图：learning_curves.png / learning_curves.svg；数据：learning_curves.csv；原始曲线：small72_curve.json / expanded160_curve.json；所有标签、前缀和哈希快照保留。']
        (args.out/'REPORT.md').write_text('\n'.join(lines))
        log('学习曲线全部完成。模型/现役哈希未变；后续需分析泛化与旧能力，不按曲线直接升级。');status('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        status('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
