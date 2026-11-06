"""One bounded diagnosis -> factual expert continuations -> CNN study -> fresh gate.
Uses development train/validation games only. No old arena outcomes feed training.
Teacher is Clysto-evaluation-derived strict121, not original Clysto. CNN only.
"""
import argparse,copy,fcntl,json,os,random,shutil,time,traceback
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
import rules121 as r
from az121 import Net121,Neural121,encode,action_id,safe_root_actions
from curriculum121 import atomic,opening,replay,finish,batch,samples_from_game
from canonical121_trial import canonical_sample
from engine_clysto121 import ClystoDerived121
from engine_c121 import CSearch121
from iterate121 import summary,paired_gate
from checkpoint121_save import verify,digest,REGISTRY,restore
from confirm121_unfrozen import seen_openings,fresh_opening
from unfreeze121_trial import objective,metrics

RANK={'missed_immediate_win':0,'allowed_immediate_loss':1,'finish_delay':2,'rear_lag':3,'expert_disagreement':4}

def farthest(board,player):
    q0,y0=r.COORDS[120 if player==1 else 0]
    return max(max(abs(q-q0),abs(y-y0),abs(q+y-q0-y0)) for i,(q,y) in enumerate(r.COORDS) if board[i]==player)


def classify(g,played,expert):
    allowed,probe=safe_root_actions(g.board,g.player,g.ply,g.repetitions)
    chosen=(played[0],played[-1]);advice=(expert[0],expert[-1])
    if probe['immediate_wins'] and chosen not in allowed and advice in allowed:return 'missed_immediate_win'
    if probe['unsafe'] and chosen not in allowed and advice in allowed:return 'allowed_immediate_loss'
    p=g.player;before=sum(g.board[i]==p for i in r.GOAL[p]);a=r.apply_move(g.board,played);b=r.apply_move(g.board,expert)
    if before>=7 and sum(b[i]==p for i in r.GOAL[p])>sum(a[i]==p for i in r.GOAL[p]):return 'finish_delay'
    if farthest(b,p)<farthest(a,p):return 'rear_lag'
    return 'expert_disagreement'


def select_seeds(rows,part,limit):
    candidates=sorted((x for x in rows if x['split']==part),key=lambda x:(RANK[x['category']],-x['ply'],x['game_id']))
    chosen=[];counts=Counter();used=set()
    # Spread across games before taking a second seed from any game.
    for ceiling in (1,2):
        for row in candidates:
            key=(row['game_id'],row['ply'])
            if key in used or counts[row['game_id']]>=ceiling:continue
            chosen.append(row);used.add(key);counts[row['game_id']]+=1
            if len(chosen)>=limit:return chosen
    return chosen


def value_label(winner,reason,player):
    if winner not in (0,1,2):raise ValueError('unfinished game has no label')
    if reason=='max_ply':
        if winner!=0:raise ValueError('never invent a max-ply winner')
        return 0.,0.
    return (0. if winner==0 else 1. if winner==player else -1.),1.


def expert_samples(record,canonical=True):
    g=replay(record['opening']);samples=[]
    for path in record['paths']:
        if not r.valid_path(g.board,g.player,path):raise ValueError('illegal expert continuation')
        actions=[action_id(s,p[-1],g.player,canonical) for s,p in r.get_all_moves(g.board,g.player)]
        target=actions.index(action_id(path[0],path[-1],g.player,canonical));z,w=value_label(record['winner'],record['reason'],g.player)
        samples.append(dict(x=encode(g.board,g.player,g.ply,g.repetitions,canonical=canonical),actions=actions,target=target,z=z,value_weight=w))
        g.play(path)
    if g.result()!=(record['winner'],record['reason']) or g.ply!=record['plies']:raise ValueError('fabricated expert outcome')
    return samples


def verified_better(expert,forced,player):
    def points(row):return .5 if row['winner']==0 else float(row['winner']==player)
    a,b=points(expert),points(forced)
    return a>b or (a==b==1. and expert['plies']+2<=forced['plies'])


def configure_arm(model,arm):
    if arm not in ('last_block','full_cnn'):raise ValueError('unknown CNN arm')
    model.requires_grad_(arm=='full_cnn')
    if arm=='last_block':
        for module in (model.blocks[-1],model.src,model.dst,model.bias,model.value):module.requires_grad_(True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=301049);args=ap.parse_args()
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    save_path=Path('nn_checkpoints/saves/v3_before_unfreeze_20261030');saved=verify(save_path)
    if digest(REGISTRY)!=saved['registry_sha256']:raise ValueError('champion changed; preserve it')
    model_path=save_path/'model.pt';baseline=Net121.load(model_path)
    args.out.mkdir(parents=True,exist_ok=False);lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic()
    def update(stage,**kw):
        atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-started,**kw))
        if stage not in ('complete','failed') and time.monotonic()-started>10800:raise TimeoutError('predeclared three-hour budget exhausted; preserve v3')
    def log(message):
        with (args.out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-started:.1f}秒：{message}\n')
    def record(name,row):
        with (args.out/(name+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
    names=['cycle121_targeted.py','az121.py','rules121.py','curriculum121.py','canonical121_trial.py','engine_clysto121.py','engine_c121.py','iterate121.py','checkpoint121_save.py','confirm121_unfrozen.py','unfreeze121_trial.py','neural121_registry.py','clysto121_generated/core.c','clysto121_generated/manifest.json']
    import ccore121,clysto121_native
    names += [Path(ccore121.__file__).name,Path(clysto121_native.__file__).name]
    config=dict(model=str(model_path.resolve()),model_sha256=digest(model_path),profile=r.PROFILE,encoding=baseline.encoding,seed=args.seed,simulations=32,
        training='Targeted factual expert continuations + normal expert replay; last residual block vs full CNN adaptation. Clysto-evaluation derivative, not original Clysto, not pure selfplay.',
        maximum_seed_states=48,train_seeds=42,validation_seeds=6,maximum_per_game=2,branches_per_seed=2,normal_expert_games=48,
        expert_seconds=.06,root_expert_seconds=.12,normal_stronger_seconds=.12,teacher_depth=12,
        arms=['last_block','full_cnn'],steps_per_arm=1600,learning_rate=.00003,policy_kl=.1,value_preservation=.1,
        batch_mix='8 focused expert continuation / 4 new normal expert / 4 old genuine expert; prioritize verified counterfactual improvements within focused half',
        selection='Validation only: select one trained arm by equal-domain mean policy CE + 0.5 genuine value MSE; baseline is reference; no second candidate after arena failure',
        confirmation_games=120,fixed_c_games_per_model=40,independent_gate_p=.01,c_seconds=.03,c_depth=10,max_seconds=10800,
        source_sha256={name:digest(name) for name in names})
    for name in names:
        dest=args.out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
    data=args.out/'data_sources';data.mkdir();old=Path('nn_checkpoints/experiments/imitate_clysto121_20261030');dev=Path('nn_checkpoints/experiments/dagger121_expert_correction_20261030')
    sources={'old_teacher.jsonl':old/'teacher_games.jsonl','old_split.json':old/'split.json','dev_games.jsonl':dev/'expert_rollout.jsonl','dev_annotations.jsonl':dev/'annotations.jsonl','dev_split.json':dev/'split.json'}
    config['data_sha256']={}
    for name,path in sources.items():shutil.copyfile(path,data/name);config['data_sha256'][name]=digest(data/name)
    atomic(args.out/'config.json',config)
    try:
        update('preregister');used,excluded_sources=seen_openings('nn_checkpoints/experiments')
        schedule={name:[fresh_opening(args.seed+offset+i,4+(i%5)*2,used) for i in range(n)] for name,n,offset in [('normal',24,30000),('confirmation',60,100000),('fixed_c',20,200000)]}
        regular_ids=list(range(24));random.Random(args.seed).shuffle(regular_ids);regular_val=set(regular_ids[:4])
        atomic(args.out/'schedule.json',schedule);atomic(args.out/'regular_split.json',dict(validation=sorted(regular_val),train=sorted(set(range(24))-regular_val)))
        atomic(args.out/'opening_exclusions.json',excluded_sources)
        for name in ('schedule.json','regular_split.json','opening_exclusions.json'):
            shutil.copyfile(args.out/name,data/name);config['data_sha256'][name]=digest(data/name)
        atomic(args.out/'config.json',config);log('预算/日程已固定。只诊断开发训练和验证局，不读取旧验收胜负作为训练材料。')
        update('diagnosis');splits=json.loads((data/'dev_split.json').read_text());games=[json.loads(s) for s in (data/'dev_games.jsonl').read_text().splitlines()];annotations=[json.loads(s) for s in (data/'dev_annotations.jsonl').read_text().splitlines()]
        candidates=[];diagnosis=defaultdict(Counter)
        for game_id,(game,ann) in enumerate(zip(games,annotations)):
            part=next((k for k in ('train','validation') if game['pair'] in splits[k]),None)
            if part is None:continue
            g=replay(game['opening']);prefix=copy.deepcopy(game['opening']);labels={s['ply']:s for s in ann['annotations']}
            for path in game['paths']:
                if g.ply in labels:
                    label=labels[g.ply];expert=label['expert_path']
                    if label['played_path']!=path or label['player']!=g.player or not r.valid_path(g.board,g.player,expert):raise ValueError('invalid developer annotation')
                    if (path[0],path[-1])!=(expert[0],expert[-1]):
                        category=classify(g,path,expert);diagnosis[part][category]+=1
                        candidates.append(dict(game_id=game_id,group=game['pair'],split=part,ply=g.ply,player=g.player,category=category,opening=copy.deepcopy(prefix),played_path=path,original_expert_path=expert))
                g.play(path);prefix.append(path)
            if g.result()!=(game['winner'],game['reason']):raise ValueError('invalid developer outcome')
            update('diagnosis',game_id=game_id)
        seeds=select_seeds(candidates,'train',42)+select_seeds(candidates,'validation',6)
        if not seeds or not any(s['split']=='validation' for s in seeds):raise ValueError('no usable focused validation seeds')
        atomic(args.out/'diagnosis.json',dict(category_counts={k:dict(v) for k,v in diagnosis.items()},selected=len(seeds),scope='Diagnostic categories are hypotheses, not ground-truth bad moves; factual paired continuations follow.'))
        atomic(args.out/'seeds.json',seeds);log('开发局复盘完成：'+json.dumps({k:dict(v) for k,v in diagnosis.items()},ensure_ascii=False))
        teacher=ClystoDerived121(seconds=.06,max_depth=12);root_teacher=ClystoDerived121(seconds=.12,max_depth=12)
        focus={'train':[],'validation':[]};verified={'train':[],'validation':[]};comparisons=[];training_records=[]
        for seed_id,seed in enumerate(seeds):
            update('focused_expert_continuations',completed=seed_id*2,total=len(seeds)*2)
            origin=replay(seed['opening']);expert_path,_=root_teacher.think(origin.board,origin.player,origin.ply,origin.repetitions)
            branches={}
            for role,forced in [('expert',expert_path),('played',seed['played_path'])]:
                prefix=seed['opening']+[forced];g=replay(prefix);row=finish(g,{1:teacher,2:teacher})
                if role=='expert':row.update(opening=seed['opening'],start_ply=seed['ply'],paths=[forced]+row['paths'])
                else:row['opening']=prefix # Exclude the forced NN move from expert policy supervision.
                row.update(seed_id=seed_id,group=seed['group'],split=seed['split'],role=role,category=seed['category'])
                branches[role]=row;record('focused_games',row);training_records.append(row)
            better=verified_better(branches['expert'],branches['played'],seed['player'])
            comparison=dict(seed_id=seed_id,split=seed['split'],category=seed['category'],player=seed['player'],verified_better=better,
                            expert_winner=branches['expert']['winner'],played_winner=branches['played']['winner'],expert_end_ply=branches['expert']['plies'],played_end_ply=branches['played']['plies'])
            comparisons.append(comparison);record('counterfactual_comparison',comparison)
            for row in branches.values():
                part=expert_samples(row);focus[seed['split']].extend(part)
                if better:verified[seed['split']].extend(part)
            log(f'专家分支{seed_id+1}/{len(seeds)}完成；类别{seed["category"]}，同预算续弈验证更好={better}。')
        atomic(args.out/'counterfactual_summary.json',dict(seeds=len(seeds),verified_better=sum(x['verified_better'] for x in comparisons),by_category={k:dict(total=sum(x['category']==k for x in comparisons),verified=sum(x['category']==k and x['verified_better'] for x in comparisons)) for k in RANK}))
        normal={'train':[],'validation':[]};stronger=ClystoDerived121(seconds=.12,max_depth=12)
        for pair,prefix in enumerate(schedule['normal']):
            part='validation' if pair in regular_val else 'train'
            for side in (1,2):
                update('normal_expert_replay',completed=2*pair+side-1,total=48)
                row=finish(replay(prefix),{side:stronger,3-side:teacher});row.update(opening=prefix,group=pair,split=part,stronger_side=side)
                record('normal_games',row);training_records.append(row);normal[part].extend(expert_samples(row))
        retention={'train':[],'validation':[]};old_split=json.loads((data/'old_split.json').read_text())
        for line in (data/'old_teacher.jsonl').read_text().splitlines():
            game=json.loads(line);part=next((k for k in retention if game['group'] in old_split[k]),None)
            if part:
                retention[part].extend({**canonical_sample(s),'value_weight':1.} for s in samples_from_game(game))
        if any(not samples for domain in (focus,normal,retention) for samples in domain.values()):raise ValueError('empty training/validation domain')
        for name in ('focused_games.jsonl','normal_games.jsonl','counterfactual_comparison.jsonl','seeds.json'):
            shutil.copyfile(args.out/name,data/name);config['data_sha256'][name]=digest(data/name)
        atomic(args.out/'config.json',config)
        atomic(args.out/'data.json',dict(focus={k:len(v) for k,v in focus.items()},verified_focus={k:len(v) for k,v in verified.items()},normal={k:len(v) for k,v in normal.items()},retention={k:len(v) for k,v in retention.items()}))
        def evaluate(model):
            scores={name:metrics(model,domain['validation']) for name,domain in [('focus',focus),('normal',normal),('retention',retention)]}
            scores['criterion']=sum(v['policy_ce']+.5*v['value_mse'] for v in scores.values())/3
            return scores
        scores={'baseline':evaluate(baseline)};models={'baseline':baseline};log('专家补数结束，开始最后卷积块/全CNN两支受控训练。')
        for arm in config['arms']:
            model=copy.deepcopy(baseline);configure_arm(model,arm);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.00003,weight_decay=.0001)
            rng=np.random.default_rng(args.seed);best=float('inf');history=[]
            for step in range(1600):
                chosen=[]
                for pool,n in [(verified['train'] or focus['train'],4),(focus['train'],4),(normal['train'],4),(retention['train'],4)]:chosen.extend(pool[int(i)] for i in rng.integers(len(pool),size=n))
                xs,ids,mask,target,z=batch(chosen);weight=torch.tensor([s['value_weight'] for s in chosen]);loss=objective(model,baseline,xs,ids,mask,target,z,weight)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                if (step+1)%100==0:
                    val=evaluate(model);history.append(dict(step=step+1,loss=float(loss.detach()),validation=val))
                    if val['criterion']<best:best=val['criterion'];model.save(args.out/(arm+'.pt'))
                    atomic(args.out/(arm+'_training.json'),history);update('cnn_training',arm=arm,step=step+1,total=1600)
            selected=Net121.load(args.out/(arm+'.pt'));models[arm]=selected;scores[arm]=evaluate(selected)
            changed=[k for k,v in selected.state_dict().items() if not torch.equal(v,baseline.state_dict()[k])]
            if arm=='last_block' and any(k.startswith(('stem.','blocks.0.')) for k in changed):raise ValueError('partial arm modified frozen early CNN')
            if arm=='full_cnn' and not any(k.startswith('stem.') for k in changed):raise ValueError('full CNN failed to update')
            atomic(args.out/(arm+'_parameter_audit.json'),dict(changed_parameters=changed))
        arm=min(config['arms'],key=lambda k:scores[k]['criterion']);atomic(args.out/'selection.json',dict(selected=arm,validation=scores,criterion=config['selection']))
        candidate=models[arm];candidate.save(args.out/'candidate.pt');log('验证预选'+arm+'；现在冻结权重，开始全新开局独立验收。')
        matches=[]
        for pair,prefix in enumerate(schedule['confirmation']):
            for side in (1,2):
                update('independent_confirmation',completed=len(matches),total=120)
                row=finish(replay(prefix),{side:Neural121(candidate,32),3-side:Neural121(baseline,32)});row.update(pair=pair,side=side,opening=prefix);matches.append(row);record('confirmation',row)
        fixed={}
        for name,model in [('control',baseline),('candidate',candidate)]:
            rows=[]
            for pair,prefix in enumerate(schedule['fixed_c']):
                for side in (1,2):
                    update('fixed_c_'+name,completed=len(rows),total=40)
                    row=finish(replay(prefix),{side:Neural121(model,32),3-side:CSearch121(seconds=.03,max_depth=10)});row.update(pair=pair,side=side,opening=prefix);rows.append(row);record('fixed_c_'+name,row)
            fixed[name]=summary(rows)
        gate=paired_gate(matches);qualified=gate['one_sided_p']<=.01 and fixed['candidate']['points']>=fixed['control']['points']
        result=dict(selected=arm,confirmation=summary(matches),paired_gate=gate,fixed_c=fixed,qualified=qualified,promoted=False,seconds=time.monotonic()-started)
        atomic(args.out/'result.json',result)
        # Audit the newly generated teacher continuations even if the candidate loses.
        verified_games=0
        for row in training_records:expert_samples(row);verified_games+=1
        atomic(args.out/'expert_data_audit.json',dict(verified_games=verified_games,failures=0,forced_nn_moves_excluded_from_policy_targets=True))
        verify(save_path)
        if digest(REGISTRY)!=saved['registry_sha256']:raise ValueError('concurrent registry change; do not overwrite')
        if qualified:
            from neural121_registry import register,load_registered
            try:
                entry=register(args.out,'v4_targeted_cycle');engine=load_registered()
                for p in (1,2):
                    path,info=engine.think(r.init_board(),p)
                    if not r.valid_path(r.init_board(),p,path) or info['root_visits']!=32:raise ValueError('adoption smoke failed')
                atomic(args.out/'adoption.json',dict(adopted=True,version=entry['version'],save_preserved=True));log('完整审计通过，登记v4_targeted_cycle，v3存档仍保留。')
            except BaseException:
                current=json.loads(REGISTRY.read_text())
                if current.get('version')=='v4_targeted_cycle' and current.get('sha256')==digest(args.out/'candidate.pt'):restore(save_path)
                raise
        else:atomic(args.out/'adoption.json',dict(adopted=False,retained_version=saved['version'],save_verified=True));log('独立验收未达标，保留v3存档，不追加补考。')
        (args.out/'总结.md').write_text(
            '# 针对薄弱局面的一轮完整迭代\n\n'
            + f'已完成：开发局复盘、{len(training_records)}局真实专家续弈/普通对局、两支各1600步CNN训练、200局独立验收。\n\n'
            + f'验证选定：{arm}。对v3：{result["confirmation"]["wins"]}胜{result["confirmation"]["losses"]}负{result["confirmation"]["draws"]}和，配对p={gate["one_sided_p"]:.8g}。\n\n'
            + f'对C候选：{fixed["candidate"]}；对C存档v3：{fixed["control"]}。\n\n'
            + ('已通过门槛与完整审计，登记v4_targeted_cycle；v3存档保留。' if qualified else '未通过门槛，保留v3；不追加补考。')
            + '\n\n老师为Clysto评估派生121版本，不是原版；本轮为专家监督学习，不是纯自对弈提升。\n')
        update('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        update('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
