"""受控首轮：固定WDL自弈→真实终局纠错→概率与搜索复核，不自动晋升。"""
import argparse,copy,fcntl,json,math,os,shutil,time,traceback
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
import rules121 as r
from az121 import Net121,search
from wdl121 import WDLNet121,training_loss,training_batch
from wdl121_metrics import wdl_metrics,metrics_from_probabilities
from wdl121_data import make_schedule,run_selfplay,build_samples,identity
from curriculum121 import atomic,replay
from collect121_order_pairs import build_history
from cap_opening121_replay import load_data
from checkpoint121_save import digest,REGISTRY
from order121_learning import value_state_hash
from diagnostics.search121_diagnostic import summarize_games
OLD=Path('nn_checkpoints/experiments/fresh121_projection_20261105/data_sources')
PROTOCOL=Path('docs/experiments/wdl121_loop_20261107/protocol.md')

def policy_only(sample):
    # Old replay outcome metadata lacks current-player semantics; only its policy is used.
    return {**{k:v for k,v in sample.items() if k not in ('winner','reason','wdl_class')},'wdl_weight':0.}

@torch.no_grad()
def policy_ce(model,samples):
    total=0.;n=0;was=model.training;flags=[p.requires_grad for p in model.parameters()];model.eval();model.requires_grad_(False)
    try:
        for off in range(0,len(samples),32):
            ss=samples[off:off+32];b=training_batch([policy_only(s) for s in ss]);logits,_=model(b['x']);lp=logits.gather(1,b['ids']).masked_fill(~b['mask'],-1e9).log_softmax(1)
            total+=float(-(lp*b['policy']).sum());n+=len(ss)
        return total/max(1,n)
    finally:
        model.train(was)
        for p,flag in zip(model.parameters(),flags):p.requires_grad_(flag)

def prior_for_training(samples):
    groups=defaultdict(list)
    for s in samples:groups[s['group']].append(s)
    counts={p:np.ones(3) for p in (1,2)}
    # Actual game-uniform then sample-uniform target, conditioned on player.
    for ss in groups.values():
        for s in ss:
            if s['wdl_class'] is not None:counts[s['player']][s['wdl_class']]+=1/len(ss)
    return {p:(v/v.sum()).tolist() for p,v in counts.items()}

def development_checks(final,initial,prior,pc,basepc,old_after,old_before,unchanged):
    f,i,p=final['game_macro'],initial['game_macro'],prior['game_macro']
    exists=lambda x:x is not None
    checks=dict(development_coverage=f['n_games']>=8,
        nll_uniform=exists(f['nll']) and f['nll']<=i['nll']-.02,
        nll_prior=exists(f['nll']) and f['nll']<=p['nll']-.02,
        brier_uniform=exists(f['brier3']) and f['brier3']<=i['brier3']-.02,
        brier_prior=exists(f['brier3']) and f['brier3']<=p['brier3'],
        new_policy=pc<=basepc+.05,old_policy=old_after['old']<=old_before['old']+.05,
        normal_policy=old_after['normal']<=old_before['normal']+.05,value_parameters_unchanged=unchanged)
    for player in ('1','2'):
        x,y=final['by_player'][player],initial['by_player'][player]
        checks['player'+player+'_brier']=x['n']>0 and x['brier3']<=y['brier3']
    checks['late_brier']=final['late']['n']==0 or final['late']['brier3']<=initial['late']['brier3']
    return checks

def initial_model(v3,parent_wdl=None,iteration=1):
    if iteration<1 or (iteration>1 and parent_wdl is None):raise ValueError('后续迭代必须指定父WDL checkpoint')
    if parent_wdl is None:return WDLNet121.from_v3(v3,iteration=0).eval()
    model=WDLNet121.load(parent_wdl)
    model.metadata.update(iteration=iteration-1,parent_wdl_sha256=digest(parent_wdl))
    return model


def decisive_metrics(metrics):
    records=[row for row in metrics['records'] if row['label'] in (0,2)]
    def measure(rows):
        groups=defaultdict(list)
        for row in rows:
            win,draw,loss=row['probabilities'];mass=win+loss
            if not math.isfinite(mass) or mass<=0:raise ValueError('胜负条件概率无定义')
            groups[row['game_id']].append((win/mass-float(row['label']==0))**2)
        return dict(n=len(rows),wins=sum(row['label']==0 for row in rows),losses=sum(row['label']==2 for row in rows),n_games=len(groups),game_macro_brier=float(np.mean([np.mean(v) for v in groups.values()])) if groups else None)
    return dict(overall=measure(records),by_player={str(p):measure([row for row in records if row['player']==p]) for p in (1,2)})


def decisive_checks(final,parent,prior):
    f,p,t=[decisive_metrics(m) for m in (final,parent,prior)];a,b,c=[m['overall'] for m in (f,p,t)]
    winners={1:set(),2:set()}
    for row in final['records']:
        if row['label'] in (0,2):winners[row['player'] if row['label']==0 else 3-row['player']].add(row['game_id'])
    coverage=all(len(ids)>=2 for ids in winners.values()) and all(m['wins']>0 and m['losses']>0 for m in f['by_player'].values())
    checks=dict(decisive_coverage=coverage,decisive_parent_gain005=coverage and a['game_macro_brier']<=b['game_macro_brier']-.005,decisive_prior_gain005=coverage and a['game_macro_brier']<=c['game_macro_brier']-.005)
    for player in ('1','2'):
        x,y=f['by_player'][player],p['by_player'][player]
        checks['player'+player+'_decisive_nonworse']=x['wins']>0 and x['losses']>0 and x['game_macro_brier']<=y['game_macro_brier']
    return checks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--parent-wdl',type=Path);ap.add_argument('--iteration',type=int,default=1);ap.add_argument('--seed-base',type=int,default=111120);ap.add_argument('--protocol',type=Path,default=PROTOCOL);ap.add_argument('--preflight-charge-seconds',type=float,default=0.);a=ap.parse_args()
    if not math.isfinite(a.preflight_charge_seconds) or not 0<=a.preflight_charge_seconds<2400:raise ValueError('预检预算扣除非法')
    if a.iteration<1 or (a.iteration>1 and a.parent_wdl is None):raise ValueError('后续迭代必须指定父WDL checkpoint')
    if not a.protocol.is_file():raise ValueError('协议文件不存在')
    a.out.mkdir(parents=True,exist_ok=False);lock=(a.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic()-a.preflight_charge_seconds;torch.set_num_threads(1);torch.manual_seed(a.seed_base)
    reg=json.loads(REGISTRY.read_text());parent=Path(reg['checkpoint']);assert reg['version']=='v3_expert_imitation' and digest(parent)==reg['sha256']
    v3=Net121.load(parent);v3.requires_grad_(False);initial=initial_model(v3,a.parent_wdl,a.iteration);initial.metadata.update(parent_sha256=reg['sha256']);initial.requires_grad_(False);initial.save(a.out/'initial_wdl.pt')
    import ccore121,clysto121_native
    names=['train121_wdl_loop.py','wdl121.py','wdl121_metrics.py','wdl121_data.py','wdl121_display.py','gui121.py','az121.py','rules121.py','curriculum121.py','collect121_order_pairs.py','expert121_pilot48.py','corrected121_decisions.py','engine_clysto121.py','engine_c121.py','history121_exclusions.py','opening121_split_audit.py','cap_opening121_replay.py','canonical121_trial.py','cycle121_targeted.py','generalization121_curve.py','train_opening121.py','opening121.py','opening121_book.json','unfreeze121_trial.py','order121_learning.py','checkpoint121_save.py','diagnostics/search121_diagnostic.py',str(a.protocol)]
    names+=['tests/'+n for n in ('test_wdl121.py','test_wdl121_data.py','test_wdl121_display.py','test_wdl121_search.py','test_train121_wdl_loop.py','test_history121_search_metadata.py','test_wdl121_continuation.py')]
    cfg=dict(roadmap_step=3,iteration=a.iteration,parent_wdl=str(a.parent_wdl.resolve()) if a.parent_wdl else None,parent_wdl_sha256=digest(a.parent_wdl) if a.parent_wdl else None,protocol=str(a.protocol),profile=r.PROFILE,model=str(parent),model_sha256=digest(parent),registry_sha256=digest(REGISTRY),
        initial_wdl_sha256=digest(a.out/'initial_wdl.pt'),seed=a.seed_base,schedule_seed=a.seed_base+1,selfplay_seed=a.seed_base+2,sampling_seed=a.seed_base+3,old_sampling_seed=a.seed_base+4,
        groups=56,train=32,development=16,probe=8,simulations=32,selfplay_games=48,steps=2000,lr=1e-4,weight_decay=1e-4,clip=1.,
        loss='new policyCE + new WDLCE + .1 old legal policyKL(v3)',old_data_sha256={str(OLD/n):digest(OLD/n) for n in ('fixed32.pt','old_teacher.jsonl','old_split.json','normal_games.jsonl')},
        preflight_charge_seconds=a.preflight_charge_seconds,max_seconds=5400,collection_max_seconds=2400,training_max_seconds=1200,evaluation_max_seconds=1200,automatic_promotion=False,
        source_sha256={n:digest(n) for n in names},native_sha256={str(Path(m.__file__)):digest(m.__file__) for m in (ccore121,clysto121_native)})
    for n in names:
        dest=a.out/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,dest)
    atomic(a.out/'config.json',cfg)
    def status(stage,**kw):
        atomic(a.out/'status.json',dict(roadmap_step=3,stage=stage,pid=os.getpid(),seconds=time.monotonic()-started,**kw))
        if time.monotonic()-started>5400:raise TimeoutError('90分钟总预算耗尽')
    def log(message):
        with (a.out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-started:.1f}秒：{message}\n')
    def append(name,item):
        with (a.out/name).open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
    def guards():
        if a.parent_wdl and digest(a.parent_wdl)!=cfg['parent_wdl_sha256']:raise ValueError('父WDL文件变化')
        assert digest(parent)==cfg['model_sha256'] and digest(REGISTRY)==cfg['registry_sha256'] and digest(a.out/'initial_wdl.pt')==cfg['initial_wdl_sha256']
        for hashes in (cfg['source_sha256'],cfg['native_sha256'],cfg['old_data_sha256']):
            for n,h in hashes.items():
                if digest(n)!=h:raise ValueError('冻结输入/代码变化：'+n)
    def collect_status(stage,**kw):
        status(stage,**kw)
        if time.monotonic()-started>2400:raise TimeoutError('40分钟采集预算耗尽')
    try:
        status('完整历史覆盖预检');guards();history=build_history(a.out/'history',exclude_parent=True);atomic(a.out/'history_exclusions.json',history);historical=set(history['orbits']);history_sha=digest(a.out/'history_exclusions.json')
        schedule=make_schedule(initial,historical,seed=a.seed_base+1,status=collect_status);atomic(a.out/'schedule.json',schedule);schedule_sha=digest(a.out/'schedule.json');atomic(a.out/'schedule_freeze.json',dict(schedule_sha256=schedule_sha,groups=56,initial_wdl_sha256=cfg['initial_wdl_sha256']));log('56组开局全部冻结，开始固定本轮父模型的48盘完整自弈。')
        games=[]
        for row in schedule[:48]:
            game=run_selfplay(initial,row,seed=a.seed_base+2+row['group'],status=collect_status);games.append(game);append('selfplay_games.jsonl',game);log(f"组{row['group']}完整对局结束：winner={game['winner']}，reason={game['reason']}。")
        guards();assert digest(a.out/'history_exclusions.json')==history_sha;assert digest(a.out/'schedule.json')==schedule_sha
        pools,data_audit=build_samples(games,schedule,historical);atomic(a.out/'data_audit.json',data_audit)
        train,dev=pools['train'],pools['development'];groups=defaultdict(list)
        for s in train:groups[s['group']].append(s)
        true_groups={s['group'] for s in train if s['wdl_class'] is not None}
        torch.save({part:[{**s,'x':torch.from_numpy(s['x'])} for s in ss] for part,ss in pools.items()},a.out/'samples.pt')
        freeze=dict(schedule_sha256=schedule_sha,games_sha256=digest(a.out/'selfplay_games.jsonl'),samples_sha256=digest(a.out/'samples.pt'),train_groups=sorted(groups),train_complete_groups=sorted(true_groups),development_complete_groups=sorted({s['group'] for s in dev if s['wdl_class'] is not None}),game_outcomes=dict(Counter(str(g['winner'])+':'+g['reason'] for g in games)),training_ready=len(true_groups)>=16)
        atomic(a.out/'dataset_freeze.json',freeze);log('数据与真实标签冻结：'+json.dumps(freeze,ensure_ascii=False))
        if not freeze['training_ready']:
            atomic(a.out/'result.json',dict(training_started=False,promoted=False,development_pass=False,reason='train真实完整有效对局不足16，不补局或伪造截断标签',freeze=freeze));status('complete');return
        _,old,normal,_=load_data(OLD);old_before={k:policy_ce(v3,ss['validation']) for k,ss in [('old',old),('normal',normal)]}
        status('冻结本轮父模型开发基线');before=wdl_metrics(initial,dev);uniform=metrics_from_probabilities(dev,[[1/3]*3 for _ in dev]);prior=prior_for_training(train);prior_metrics=metrics_from_probabilities(dev,[prior[s['player']] for s in dev]);basepc=policy_ce(initial,dev)
        atomic(a.out/'baseline.json',dict(wdl=before,uniform=uniform,train_player_prior=prior,prior_metrics=prior_metrics,new_policy_ce=basepc,old_policy=old_before))
        model=copy.deepcopy(initial);model.requires_grad_(True);model.value.requires_grad_(False);initial_value=value_state_hash(initial)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);rr=np.random.default_rng(a.seed_base+3);ro=np.random.default_rng(a.seed_base+4);group_ids=sorted(groups);trace=[];curve=[];losses=[];training_started=time.monotonic()
        for step in range(2001):
            if time.monotonic()-training_started>1200 or time.monotonic()-started>5400:raise TimeoutError('训练或总预算耗尽')
            if step%1000==0:model.save(a.out/f'candidate_step{step}.pt',dict(iteration=a.iteration,step=step))
            if step%400==0:
                status('真实终局WDL联合训练',step=step,total=2000)
                model.eval();curve.append(dict(step=step,train=wdl_metrics(model,train)))
                atomic(a.out/'curve.json',curve);atomic(a.out/'sample_trace.json',trace);atomic(a.out/'losses.json',losses)
                if time.monotonic()-training_started>1200:raise TimeoutError('20分钟训练预算耗尽')
                if step==2000:break
            gi=rr.integers(len(group_ids),size=16);chosen_groups=[group_ids[int(i)] for i in gi];si=[int(rr.integers(len(groups[g]))) for g in chosen_groups];new=[groups[g][i] for g,i in zip(chosen_groups,si)]
            oi=ro.integers(len(old['train']),size=8);ni=ro.integers(len(normal['train']),size=8);ordinary=[policy_only(old['train'][int(i)]) for i in oi]+[policy_only(normal['train'][int(i)]) for i in ni]
            trace.append(dict(groups=chosen_groups,samples=si,old=oi.tolist(),normal=ni.tolist()));model.train()
            terms=training_loss(model,new);ret=training_loss(model,ordinary,reference=v3,policy_weight=0,wdl_weight=0,kl_weight=.1);loss=terms['total']+ret['total']
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            opt.zero_grad();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True));opt.step()
            if step%100==0:losses.append(dict(step=step,policy_ce=float(terms['policy_ce'].detach()),wdl_ce=float(terms['wdl_ce'].detach()),old_kl=float(ret['policy_kl'].detach()),preclip_norm=norm))
        evaluation_started=time.monotonic();candidate=WDLNet121.load(a.out/'candidate_step2000.pt');candidate.requires_grad_(False);status('固定终点开发概率复核');after=wdl_metrics(candidate,dev);pc=policy_ce(candidate,dev);old_after={k:policy_ce(candidate,ss['validation']) for k,ss in [('old',old),('normal',normal)]};unchanged=value_state_hash(candidate)==initial_value
        checks=development_checks(after,uniform,prior_metrics,pc,basepc,old_after,old_before,unchanged)
        if a.iteration>1:checks.update(decisive_checks(after,before,prior_metrics))
        passed=all(checks.values());atomic(a.out/'final.json',dict(wdl=after,new_policy_ce=pc,old_policy=old_after,checks=checks))
        parent_name='PARENT_WDL' if a.iteration>1 else 'WDL0'
        train_orbits={s['orbit'] for s in train};probe_audit=[];probes=[]
        for row in schedule[48:]:
            isolated=not ({row['orbit'],row['anchor']}&train_orbits);probe_audit.append(dict(group=row['group'],input_isolated=isolated))
            for opponent_name,opponent in [(parent_name,initial),('v3',v3)]:
                for side in (1,2):
                    g=replay(row['opening']);paths=[];infos=[]
                    while g.result()[0] is None:
                        status('固定候选搜索行为诊断',group=row['group'],opponent=opponent_name,nn_side=side,ply=g.ply)
                        if time.monotonic()-evaluation_started>1200:raise TimeoutError('20分钟评测预算耗尽')
                        player=g.player;net=candidate if player==side else opponent;wall=time.monotonic();path,info=search(net,g.board,g.player,g.ply,g.repetitions,simulations=32)
                        infos.append(dict(player=player,ply=g.ply,engine='NN' if player==side else opponent_name,elapsed=time.monotonic()-wall,cpu_seconds=None,info=info));paths.append(path);g.play(path)
                    winner,reason=g.result();item=dict(group=row['group'],nn_side=side,opponent=opponent_name,input_isolated=isolated,opening=row['opening'],start_ply=row['ply'],paths=paths,search_info=infos,winner=winner,reason=reason,plies=g.ply);probes.append(item);append('probe_games.jsonl',item)
        atomic(a.out/'probe_isolation.json',probe_audit);guards();assert digest(a.out/'history_exclusions.json')==history_sha;assert digest(a.out/'schedule.json')==schedule_sha and digest(a.out/'selfplay_games.jsonl')==freeze['games_sha256'] and digest(a.out/'samples.pt')==freeze['samples_sha256']
        result=dict(roadmap_step=3,iteration=a.iteration,training_started=True,development_pass=passed,ready_for_step4=passed,calibrated=False,promoted=False,checks=checks,before=before['game_macro'],after=after['game_macro'],prior=prior_metrics['game_macro'],old_policy_before=old_before,old_policy_after=old_after,value_parameters_unchanged=unchanged,checkpoint_sha256=digest(a.out/'candidate_step2000.pt'),probe_counts={name:dict(games=sum(g['opponent']==name for g in probes),wins=sum(g['opponent']==name and g['winner']==g['nn_side'] for g in probes),losses=sum(g['opponent']==name and g['winner'] not in (0,g['nn_side']) for g in probes),true_draws=sum(g['opponent']==name and g['winner']==0 and g['reason']!='max_ply' for g in probes),truncated=sum(g['opponent']==name and g['reason']=='max_ply' for g in probes)) for name in (parent_name,'v3')},seconds=time.monotonic()-started)
        atomic(a.out/'result.json',result)
        atomic(a.out/'display_model.json',dict(checkpoint=str((a.out/'candidate_step2000.pt').resolve()),sha256=result['checkpoint_sha256'],model_version=f'WDL受控第{a.iteration}轮_step2000',role='experimental',calibrated=False,development_pass=passed))
        if passed and a.out.name!='wdl121_loop_20261107':atomic(Path('nn_checkpoints/experiments/wdl121_loop_20261107/display_model.json'),json.loads((a.out/'display_model.json').read_text()))
        log(f'第{a.iteration}轮闭环完成；'+('开发门槛通过，仅准备后续独立搜索验证。' if passed else '开发门槛未通过；候选仅实验分析，不切换冠军。'));status('complete',development_pass=passed)
    except BaseException:
        atomic(a.out/'status.json',dict(roadmap_step=3,stage='failed',pid=os.getpid(),seconds=time.monotonic()-started,error=traceback.format_exc()));raise
if __name__=='__main__':main()
