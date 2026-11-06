"""Correct late selection, mixed expert budgets, and trajectory-diluted supervision.
Early explicit decision pool; one fixed-depth expert; 25% decision exposure.
Teacher scores are search-distillation targets, NEVER fabricated game outcomes.
"""
import argparse,copy,fcntl,json,os,random,shutil,time,traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import rules121 as r
from az121 import Net121,Neural121,encode,action_id
from curriculum121 import atomic,replay,finish,batch,samples_from_game
from canonical121_trial import canonical_sample
from engine_clysto121 import ClystoDerived121
from engine_c121 import CSearch121
from cycle121_targeted import configure_arm,expert_samples
from unfreeze121_trial import objective,metrics
from checkpoint121_save import verify,digest,REGISTRY,restore
from confirm121_unfrozen import fresh_opening,seen_openings
from iterate121 import summary,paired_gate

class ConsistentExpert(ClystoDerived121):
    failure_dir=None
    def __init__(self,depth=5):
        super().__init__(seconds=120.,max_depth=depth);self.requested_depth=depth
    def think(self,*args,**kw):
        started=time.monotonic();path,info=super().think(*args,**kw);elapsed=time.monotonic()-started
        if self.failure_dir is not None and elapsed>5:
            with (Path(self.failure_dir)/'slow_expert_queries.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(seconds=elapsed,requested_depth=self.requested_depth,info=info,ply=args[2] if len(args)>2 else kw.get('ply',0)))+'\n')
        if path is not None and (info['aborted'] or info['depth_reached']!=self.requested_depth):
            if self.failure_dir is not None:
                board=args[0] if args else kw['board'];player=args[1] if len(args)>1 else kw['player']
                ply=args[2] if len(args)>2 else kw.get('ply',0);history=args[3] if len(args)>3 else kw.get('repetitions',{})
                atomic(Path(self.failure_dir)/('fixed_depth_failure_'+str(time.time_ns())+'.json'),dict(board=board,player=player,ply=ply,history=[dict(board=list(b),player=p,count=n) for (b,p),n in (history or {}).items() if n>0],info=info,requested_depth=self.requested_depth))
            raise RuntimeError(f'expert fixed-depth incomplete: requested={self.requested_depth}, reached={info["depth_reached"]}, aborted={info["aborted"]}, nodes={info.get("nodes")}; reject label')
        return path,info


def score_action(g,path,depth=5):
    child=copy.deepcopy(g);child.play(path)
    winner,reason=child.result()
    if winner is not None:return 0 if winner==0 else 100000 if winner==g.player else -100000
    _,info=ConsistentExpert(depth-1).think(child.board,child.player,child.ply,child.repetitions)
    return -info['score']


def eligible_decision(ply,last_ply,count):
    return ply<=40 and count<2 and (last_ply is None or ply-last_ply>=6)


def decision_sample(g,preferred,rejected):
    if not r.valid_path(g.board,g.player,preferred) or not r.valid_path(g.board,g.player,rejected):raise ValueError('illegal decision target')
    actions=[action_id(s,p[-1],g.player,True) for s,p in r.get_all_moves(g.board,g.player)]
    positive=action_id(preferred[0],preferred[-1],g.player,True);negative=action_id(rejected[0],rejected[-1],g.player,True)
    if positive==negative:raise ValueError('no distinct decision preference')
    return dict(x=encode(g.board,g.player,g.ply,g.repetitions,canonical=True),actions=actions,target=actions.index(positive),
                rejected_action=negative,z=0.,value_weight=0.)


def mixed_batch(roots,old,normal,rng):
    if not roots or not old or not normal:raise ValueError('empty sampling domain')
    return [roots[int(i)] for i in rng.integers(len(roots),size=4)]+[old[int(i)] for i in rng.integers(len(old),size=6)]+[normal[int(i)] for i in rng.integers(len(normal),size=6)]


def preference_loss(logits,samples):
    preferred=torch.tensor([s['actions'][s['target']] for s in samples]);rejected=torch.tensor([s['rejected_action'] for s in samples]);idx=torch.arange(len(samples))
    return F.relu(.5-logits[idx,preferred]+logits[idx,rejected]).mean()


@torch.no_grad()
def root_metrics(model,roots):
    ce=margin=hits=ranked=0.
    for start in range(0,len(roots),32):
        group=roots[start:start+32];xs,ids,mask,target,_=batch(group);logits,_=model(xs);legal=logits.gather(1,ids).masked_fill(~mask,-1e9)
        ce+=float(F.cross_entropy(legal,target,reduction='sum'));hits+=int((legal.argmax(1)==target).sum())
        good=ids[torch.arange(len(group)),target];bad=torch.tensor([s['rejected_action'] for s in group]);diff=logits[torch.arange(len(group)),good]-logits[torch.arange(len(group)),bad]
        margin+=float(F.relu(.5-diff).sum());ranked+=int((diff>0).sum())
    n=len(roots);return dict(n=n,policy_ce=ce/n,accuracy=hits/n,preference_accuracy=ranked/n,hinge=margin/n)


def data_gate(metadata):
    counts={part:dict(roots=sum(x['split']==part for x in metadata),groups=len({x['pair'] for x in metadata if x['split']==part})) for part in ('train','validation')}
    passed=counts['train']['roots']>=24 and counts['train']['groups']>=12 and counts['validation']['roots']>=8 and counts['validation']['groups']>=4
    return passed,counts


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=311050);ap.add_argument('--resume-from',type=Path);args=ap.parse_args()
    torch.set_num_threads(1);torch.manual_seed(args.seed);save_path=Path('nn_checkpoints/saves/v3_before_unfreeze_20261030');saved=verify(save_path)
    if digest(REGISTRY)!=saved['registry_sha256']:raise ValueError('champion differs from v3 save; preserve it')
    model_path=save_path/'model.pt';baseline=Net121.load(model_path);args.out.mkdir(parents=True,exist_ok=False)
    lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous_elapsed=float(json.loads((args.resume_from/'status.json').read_text()).get('elapsed',0)) if args.resume_from else 0.
    began=time.monotonic()-previous_elapsed
    def update(stage,**kw):
        atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-began,**kw))
        if stage not in ('failed','complete') and time.monotonic()-began>10800:raise TimeoutError('fixed three-hour budget exhausted')
    def log(message):
        with (args.out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-began:.1f}秒：{message}\n')
    def record(name,row):
        with (args.out/(name+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
    def snapshot_data(name):
        shutil.copyfile(args.out/name,data/name);config['data_sha256'][name]=digest(data/name)
    names=['corrected121_decisions.py','cycle121_targeted.py','az121.py','rules121.py','curriculum121.py','canonical121_trial.py','engine_clysto121.py','engine_c121.py','unfreeze121_trial.py','confirm121_unfrozen.py','neural121_registry.py','checkpoint121_save.py','iterate121.py','clysto121_generated/core.c','clysto121_generated/manifest.json']
    import ccore121,clysto121_native
    names += [Path(ccore121.__file__).name,Path(clysto121_native.__file__).name]
    config=dict(model=str(model_path.resolve()),model_sha256=digest(model_path),profile=r.PROFILE,encoding=baseline.encoding,seed=args.seed,simulations=32,
        training='Early explicit decision distillation from fixed-depth Clysto-evaluation-derived121, real-outcome expert replay; not original Clysto and not pure selfplay.',
        expert_depth=5,expert_safety_timeout=120,early_window=40,maximum_per_game=2,minimum_spacing=6,teacher_score_gap=4,
        development_games=48,pilot_games=8,pilot_minimum_roots=4,maximum_roots=96,coverage_gate='>=24 training roots/12 opening groups AND >=8 validation roots/4 groups',
        root_quota='Exactly 4/16=25% item exposure; explicit root decisions, never whole trajectories',normal_teacher_games=16,
        score_scope='Fixed depth5 expert best score vs forced-action child depth4. Search scores are NOT win/outcome labels; root value_weight=0.',
        arms=['last_block','full_cnn'],steps_per_arm=1200,learning_rate=.00003,policy_kl=.1,value_preservation=.1,preference_hinge_weight=.25,
        selection='Validation once: .5 root CE + .25 old CE + .25 new CE + .125(old+new value MSE) + .25 root hinge',
        confirmation_games=120,fixed_c_games_per_model=40,independent_gate_p=.01,c_seconds=.03,max_seconds=10800,
        source_sha256={name:digest(name) for name in names})
    for name in names:
        dest=args.out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
    data=args.out/'data_sources';data.mkdir();old=Path('nn_checkpoints/experiments/imitate_clysto121_20261030')
    config['data_sha256']={}
    for dest,source in [('old_teacher.jsonl',old/'teacher_games.jsonl'),('old_split.json',old/'split.json')]:shutil.copyfile(source,data/dest);config['data_sha256'][dest]=digest(data/dest)
    config['resume_from']=str(args.resume_from.resolve()) if args.resume_from else None
    config['previous_elapsed_seconds']=previous_elapsed
    atomic(args.out/'config.json',config)
    try:
        ConsistentExpert.failure_dir=args.out
        update('preregister');used,sources=seen_openings('nn_checkpoints/experiments')
        if args.resume_from:
            prior=json.loads((args.resume_from/'config.json').read_text())
            if prior['seed']!=args.seed or prior['model_sha256']!=config['model_sha256'] or prior['expert_depth']!=5:raise ValueError('resume lineage mismatch')
            for name,sha in prior['source_sha256'].items():
                if digest(args.resume_from/'source'/name)!=sha:raise ValueError('old source snapshot changed')
            for name,sha in prior['data_sha256'].items():
                if digest(args.resume_from/'data_sources'/name)!=sha:raise ValueError('old input data changed')
            schedule=json.loads((args.resume_from/'schedule.json').read_text());prior_split=json.loads((args.resume_from/'split.json').read_text())
            root_split={int(k):v for k,v in prior_split['root_pair_split'].items()};normal_val=set(prior_split['normal_validation'])
            shutil.copyfile(args.resume_from/'config.json',data/'resume_parent_config.json');config['data_sha256']['resume_parent_config.json']=digest(data/'resume_parent_config.json')
        else:
            schedule={name:[fresh_opening(args.seed+offset+i,4+(i%5)*2,used) for i in range(n)] for name,n,offset in [('development',24,10000),('normal',16,30000),('confirmation',60,100000),('fixed_c',20,200000)]}
            root_split={i:('validation' if i%4==3 else 'train') for i in range(24)}
            normal_val=set(random.Random(args.seed).sample(range(16),4))
        atomic(args.out/'schedule.json',schedule);atomic(args.out/'split.json',dict(root_pair_split=root_split,normal_validation=sorted(normal_val)))
        atomic(args.out/'opening_exclusions.json',sources)
        for name in ('schedule.json','split.json','opening_exclusions.json'):snapshot_data(name)
        atomic(args.out/'config.json',config);log('开始修正版：固定深度5、前40手、每局最早两处合格分歧、关键首步每批25%。专家分数仅作走法蒸馏，不作为胜负标签。')
        expert=ConsistentExpert();student=Neural121(baseline,32);roots={'train':[],'validation':[]};metadata=[];dev=[]
        if args.resume_from:
            dev=[json.loads(line) for line in (args.resume_from/'development.jsonl').read_text().splitlines()]
            for index,row in enumerate(dev):
                if row['pair']!=index//2 or row['side']!=1+index%2 or row['opening']!=schedule['development'][index//2]:raise ValueError('resume game order/schedule mismatch')
                g=replay(row['opening'])
                for path in row['paths']:g.play(path)
                if g.result()!=(row['winner'],row['reason']) or g.ply!=row['plies']:raise ValueError('resume game outcome mismatch')
                record('development',row)
            for line in (args.resume_from/'decisions.jsonl').read_text().splitlines():
                row=json.loads(line)
                if row['game_id']>=len(dev):continue # Discard ALL data from the incomplete game.
                if row['decision_id']!=len(metadata) or row['split']!=root_split[row['pair']]:raise ValueError('resume decision order/split mismatch')
                source_game=dev[row['game_id']];offset=row['ply']-source_game['start_ply']
                if row['opening']!=source_game['opening']+source_game['paths'][:offset] or row['rejected']!=source_game['paths'][offset]:raise ValueError('resumed decision not from its completed source game')
                origin=replay(row['opening']);sample=decision_sample(origin,row['preferred'],row['rejected']);sample['decision_id']=row['decision_id']
                roots[row['split']].append(sample);metadata.append(row);record('decisions',row)
            for name in ('development.jsonl','decisions.jsonl'):
                dest='resumed_'+name;shutil.copyfile(args.out/name,data/dest);config['data_sha256'][dest]=digest(data/dest)
            if len(dev)>=8:
                pilot=json.loads((args.resume_from/'pilot.json').read_text())
                if not pilot['passed']:raise ValueError('cannot resume past failed pilot')
                atomic(args.out/'pilot.json',pilot)
            atomic(args.out/'config.json',config);log(f'核验接续{len(dev)}局完成对局及{len(metadata)}根决策；丢弃未完成局数据。仅把安全等待上限延至120秒，仍须完整深度5，不改日程/标签/门槛。')
        for pair,prefix in enumerate(schedule['development']):
            for side in (1,2):
                if 2*pair+side-1<len(dev):continue
                update('early_development',completed=len(dev),total=48,root_counts={k:len(v) for k,v in roots.items()})
                g=replay(prefix);history=copy.deepcopy(prefix);paths=[];count=0;last=None
                while g.result()[0] is None:
                    if g.ply%5==0:update('early_development',completed=len(dev),total=48,current_ply=g.ply,root_counts={k:len(v) for k,v in roots.items()})
                    if g.player==side:path,_=expert.think(g.board,g.player,g.ply,g.repetitions)
                    else:
                        path,_=student.think(g.board,g.player,g.ply,g.repetitions)
                        if eligible_decision(g.ply,last,count):
                            advice,info=expert.think(g.board,g.player,g.ply,g.repetitions)
                            if (advice[0],advice[-1])!=(path[0],path[-1]):
                                chosen_score=score_action(g,path);gap=info['score']-chosen_score
                                if gap<0:raise ValueError('expert root score below forced legal action; reject inconsistent search label')
                                if gap>=4:
                                    advice_score=score_action(g,advice)
                                    if advice_score!=info['score']:raise ValueError('root/forced-child expert score mismatch; reject label')
                                    part=root_split[pair];root=decision_sample(g,advice,path);root['decision_id']=len(metadata);roots[part].append(root)
                                    row=dict(decision_id=len(metadata),pair=pair,side=side,split=part,game_id=len(dev),ply=g.ply,opening=copy.deepcopy(history),player=g.player,preferred=advice,rejected=path,expert_score=advice_score,rejected_score=chosen_score,score_gap=gap)
                                    metadata.append(row);record('decisions',row);count+=1;last=g.ply
                    if path is None:raise ValueError('no move in a live game')
                    g.play(path);paths.append(path);history.append(path)
                winner,reason=g.result();row=dict(pair=pair,side=side,opening=prefix,start_ply=len(prefix),plies=g.ply,paths=paths,winner=winner,reason=reason);dev.append(row);record('development',row)
                if len(dev)==8:
                    repeated=0
                    for item in metadata[:4]:
                        origin=replay(item['opening']);p,info=expert.think(origin.board,origin.player,origin.ply,origin.repetitions)
                        if (p[0],p[-1])!=(item['preferred'][0],item['preferred'][-1]) or info['score']!=item['expert_score']:raise ValueError('pilot expert label changed')
                        repeated+=1
                    pilot=dict(games=8,roots=len(metadata),groups=len({x['pair'] for x in metadata}),repeat_checks=repeated,median_ply=float(np.median([x['ply'] for x in metadata])) if metadata else None,label_consistency='accepted roots checked by forced-child score; first four re-queried',passed=len(metadata)>=4 and len({x['pair'] for x in metadata})>=2)
                    atomic(args.out/'pilot.json',pilot);log('8局开发小批检查：'+json.dumps(pilot,ensure_ascii=False))
                    if not pilot['passed']:
                        result=dict(qualified=False,promoted=False,training_started=False,reason='Pilot did not reach four decisions across two opening groups; stopped without duplicated samples')
                        atomic(args.out/'result.json',result);update('complete',result=result);return
        coverage,counts=data_gate(metadata);teacher_summary=summary(dev);teacher_gate=paired_gate(dev)
        atomic(args.out/'data_quality.json',dict(coverage_passed=coverage,coverage=counts,teacher_vs_v3=teacher_summary,teacher_gate=teacher_gate,median_ply=float(np.median([x['ply'] for x in metadata]))))
        if not coverage or teacher_summary['points']<.65*48 or teacher_gate['one_sided_p']>.01:
            result=dict(qualified=False,promoted=False,training_started=False,reason='Insufficient independent early decisions or fixed-depth expert not qualified versus v3',coverage=counts)
            atomic(args.out/'result.json',result);log('数据质量/专家资格未达预定门槛，停止，不以重复少量首步凑数量。');update('complete',result=result);return
        # Check the SAME depth-5 policy reproduces a small cross-domain sample;
        # do not replace any target or reclassify after this check.
        checked=[]
        for part in ('train','validation'):
            for row in [x for x in metadata if x['split']==part][:4]:
                g=replay(row['opening']);p,info=expert.think(g.board,g.player,g.ply,g.repetitions)
                if (p[0],p[-1])!=(row['preferred'][0],row['preferred'][-1]) or info['score']!=row['expert_score']:raise ValueError('fixed expert label failed repeatability')
                checked.append(dict(game_id=row['game_id'],ply=row['ply'],split=part))
        atomic(args.out/'label_repeatability.json',dict(checked=checked,failures=0))
        # Small factual mixed-policy branch probe is diagnostic only. No value
        # target is assigned to an unplayed alternative and no rollout token
        # is quietly counted as a root decision.
        for probe_id,row in enumerate(metadata[:8]):
            for role,forced in [('preferred',row['preferred']),('rejected',row['rejected'])]:
                prefix=row['opening']+[forced];g=replay(prefix)
                engines={row['player']:student,3-row['player']:expert}
                outcome=finish(g,engines);outcome.update(opening=prefix,probe_id=probe_id,role=role,original_player=row['player']);record('factual_branch_probe',outcome)
                update('factual_branch_probe',completed=2*probe_id+(role=='rejected')+1,total=16)
        normal={'train':[],'validation':[]}
        for i,prefix in enumerate(schedule['normal']):
            update('normal_expert_replay',completed=i,total=16);part='validation' if i in normal_val else 'train'
            row=finish(replay(prefix),{1:expert,2:expert});row.update(opening=prefix,group=i,split=part);record('normal_games',row);normal[part].extend(expert_samples(row))
        retention={'train':[],'validation':[]};old_split=json.loads((data/'old_split.json').read_text())
        for line in (data/'old_teacher.jsonl').read_text().splitlines():
            row=json.loads(line);part=next((k for k in retention if row['group'] in old_split[k]),None)
            if part:retention[part].extend({**canonical_sample(s),'value_weight':1.} for s in samples_from_game(row))
        for name in ('decisions.jsonl','development.jsonl','normal_games.jsonl','label_repeatability.json'):snapshot_data(name)
        atomic(args.out/'config.json',config);atomic(args.out/'data.json',dict(roots={k:len(v) for k,v in roots.items()},normal={k:len(v) for k,v in normal.items()},retention={k:len(v) for k,v in retention.items()},root_value_labels=0))
        def evaluate(model):
            a=root_metrics(model,roots['validation']);b=metrics(model,retention['validation']);c=metrics(model,normal['validation'])
            score=.5*a['policy_ce']+.25*(b['policy_ce']+c['policy_ce'])+.125*(b['value_mse']+c['value_mse'])+.25*a['hinge']
            return dict(root=a,retention=b,normal=c,criterion=score)
        scores={'baseline':evaluate(baseline)};models={};log('数据门槛通过，开始CNN训练；每批4个独立决策样本+12个普通专家样本。')
        for arm in config['arms']:
            model=copy.deepcopy(baseline);configure_arm(model,arm);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.00003,weight_decay=.0001)
            rng=np.random.default_rng(args.seed);best=float('inf');history=[];exposures=Counter();per_root=Counter();best_audit=None
            for step in range(1200):
                chosen=mixed_batch(roots['train'],retention['train'],normal['train'],rng)
                if sum('rejected_action' in s for s in chosen)!=4:raise ValueError('explicit 25% root quota violated')
                exposures['root']+=4;exposures['ordinary']+=12
                for root in chosen[:4]:per_root[root['decision_id']]+=1
                xs,ids,mask,target,z=batch(chosen);weight=torch.tensor([s['value_weight'] for s in chosen]);loss=objective(model,baseline,xs,ids,mask,target,z,weight)
                root_logits,_=model(xs[:4]);loss=loss+.25*preference_loss(root_logits,chosen[:4])
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                if (step+1)%100==0:
                    val=evaluate(model);history.append(dict(step=step+1,loss=float(loss.detach()),validation=val))
                    if val['criterion']<best:
                        best=val['criterion'];model.save(args.out/(arm+'.pt'));best_audit=dict(step=step+1,exposure=dict(exposures),per_root_exposure=dict(per_root))
                    atomic(args.out/(arm+'_training.json'),history);update('cnn_training',arm=arm,step=step+1,total=1200)
            selected=Net121.load(args.out/(arm+'.pt'));models[arm]=selected;scores[arm]=evaluate(selected)
            changed=[k for k,v in selected.state_dict().items() if not torch.equal(v,baseline.state_dict()[k])]
            if arm=='last_block' and any(k.startswith(('stem.','blocks.0.')) for k in changed):raise ValueError('frozen early CNN changed')
            if arm=='full_cnn' and not any(k.startswith('stem.') for k in changed):raise ValueError('full CNN did not update')
            atomic(args.out/(arm+'_sampling_audit.json'),dict(full_training_exposure=dict(exposures),selected_checkpoint=best_audit,root_fraction=.25,distinct_training_roots=len(roots['train']),actually_sampled_roots=len(best_audit['per_root_exposure'])))
        arm=min(config['arms'],key=lambda a:scores[a]['criterion']);candidate=models[arm];candidate.save(args.out/'candidate.pt')
        atomic(args.out/'selection.json',dict(selected=arm,validation=scores,criterion=config['selection']));log('验证选定'+arm+'；冻结候选，进入新开局200局验收。')
        matches=[]
        for pair,prefix in enumerate(schedule['confirmation']):
            for side in (1,2):
                update('independent_confirmation',completed=len(matches),total=120);row=finish(replay(prefix),{side:Neural121(candidate,32),3-side:Neural121(baseline,32)})
                row.update(pair=pair,side=side,opening=prefix);matches.append(row);record('confirmation',row)
        fixed={}
        for name,model in [('control',baseline),('candidate',candidate)]:
            games=[]
            for pair,prefix in enumerate(schedule['fixed_c']):
                for side in (1,2):
                    update('fixed_c_'+name,completed=len(games),total=40);row=finish(replay(prefix),{side:Neural121(model,32),3-side:CSearch121(seconds=.03,max_depth=10)})
                    row.update(pair=pair,side=side,opening=prefix);games.append(row);record('fixed_c_'+name,row)
            fixed[name]=summary(games)
        gate=paired_gate(matches);qualified=gate['one_sided_p']<=.01 and fixed['candidate']['points']>=fixed['control']['points']
        result=dict(selected=arm,confirmation=summary(matches),paired_gate=gate,fixed_c=fixed,qualified=qualified,promoted=False,seconds=time.monotonic()-began)
        atomic(args.out/'result.json',result);verify(save_path)
        if digest(REGISTRY)!=saved['registry_sha256']:raise ValueError('registry changed concurrently; preserve it')
        if qualified:
            from neural121_registry import register,load_registered
            try:
                entry=register(args.out,'v4_corrected_decisions');engine=load_registered()
                for p in (1,2):
                    path,info=engine.think(r.init_board(),p)
                    if not r.valid_path(r.init_board(),p,path) or info['root_visits']!=32:raise ValueError('adoption smoke failed')
                atomic(args.out/'adoption.json',dict(adopted=True,version=entry['version'],save_preserved=True));log('完整审计通过，登记修正版；v3存档保留。')
            except BaseException:
                active=json.loads(REGISTRY.read_text())
                if active.get('version')=='v4_corrected_decisions' and active.get('sha256')==digest(args.out/'candidate.pt'):restore(save_path)
                raise
        else:atomic(args.out/'adoption.json',dict(adopted=False,retained_version=saved['version'],save_verified=True));log('未通过独立验收，保留v3，不追加补考。')
        update('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        update('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
