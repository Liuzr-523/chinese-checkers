"""Frozen diagnostic: new policy/new critic vs new policy/complete old critic.
Matched fresh openings, 32 simulations, no training, no promotion.
"""
import argparse,copy,fcntl,json,math,os,sys,time,traceback
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from az121 import Net121,Neural121,encode
from curriculum121 import atomic,replay
from confirm121_unfrozen import fresh_opening,seen_openings
from checkpoint121_save import digest,REGISTRY
from iterate121 import summary,paired_gate
import rules121 as r

class CrossPolicyValue(torch.nn.Module):
    def __init__(self,policy_model,value_model):
        super().__init__()
        if policy_model.encoding!=value_model.encoding:raise ValueError('policy/value encoding mismatch')
        self.policy_net=policy_model.eval();self.value_net=value_model.eval()
        self.policy_net.requires_grad_(False);self.value_net.requires_grad_(False)
        self.canonical=policy_model.canonical;self.encoding=policy_model.encoding
    def forward(self,x):
        logits,_=self.policy_net(x)
        _,value=self.value_net(x)
        return logits,value


def pair_scores(rows):
    result={}
    for row in rows:
        result.setdefault(row['pair'],0.)
        result[row['pair']]+= .5 if row['winner']==0 else float(row['winner']==row['side'])
    return result


def matched_effect(new_rows,hybrid_rows,seed=111051):
    a,b=pair_scores(new_rows),pair_scores(hybrid_rows)
    if set(a)!=set(b):raise ValueError('mismatched opening groups')
    for pair in a:
        left=[x for x in new_rows if x['pair']==pair];right=[x for x in hybrid_rows if x['pair']==pair]
        if len(left)!=2 or len(right)!=2 or {x['side'] for x in left}!={1,2} or {x['side'] for x in right}!={1,2} or left[0]['opening']!=right[0]['opening']:raise ValueError('incomplete matched pair')
    diff=np.array([(b[k]-a[k])/2 for k in sorted(a)])
    positive=int((diff>0).sum());negative=int((diff<0).sum());n=positive+negative
    p=sum(math.comb(n,k) for k in range(positive,n+1))/2**n if n else 1.
    rng=np.random.default_rng(seed);means=rng.choice(diff,size=(10000,len(diff)),replace=True).mean(1)
    return dict(opening_groups=len(diff),hybrid_better_groups=positive,new_better_groups=negative,tied_groups=len(diff)-n,
                mean_game_score_rate_difference=float(diff.mean()),bootstrap95=list(map(float,np.quantile(means,[.025,.975]))),
                one_sided_sign_p=p,note='Opening-group paired exploratory diagnostic; not a model-upgrade test.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(1);seed=111051
    baseline_path=Path('nn_checkpoints/models/checkpoints/neural121_v3_expert_imitation.pt')
    candidate_path=Path('nn_checkpoints/experiments/corrected121_early_decisions_20261031_resume1/candidate.pt')
    old=Net121.load(baseline_path);new=Net121.load(candidate_path);cross=CrossPolicyValue(new,old)
    models={'v3':old,'new':new,'hybrid':cross}
    args.out.mkdir(parents=True,exist_ok=False);lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();registry_sha=digest(REGISTRY)
    def status(stage,**kw):
        atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-started,**kw))
        if stage not in ('failed','complete') and time.monotonic()-started>3600:raise TimeoutError('one-hour diagnostic budget exhausted')
    def log(message):
        with (args.out/'中文日志.md').open('a') as f:f.write(f'\n- {time.monotonic()-started:.1f}秒：{message}\n')
    source_names=['diagnostics/policy_value121_cross.py','az121.py','rules121.py','curriculum121.py','confirm121_unfrozen.py','checkpoint121_save.py','iterate121.py']
    config=dict(profile=r.PROFILE,encoding=old.encoding,seed=seed,opening_pairs=24,games_per_comparison=48,total_games=144,simulations=32,
        primary='Matched hybrid-v3 score minus new-v3 score; direct hybrid-new is a consistency check',
        decision='Only call supportive evidence of harmful critic replacement if BOTH matched one-sided p<=.025 and direct one-sided p<=.025; otherwise inconclusive or contrary. No promotion.',
        models={'v3':dict(path=str(baseline_path.resolve()),sha256=digest(baseline_path)),'new':dict(path=str(candidate_path.resolve()),sha256=digest(candidate_path))},
        cross='new complete policy logits + old complete CNN/value computation, identical canonical input; not only replacing last value layer',
        scope='Development diagnosis, equal simulation budget, NOT equal-time strength, no training or GUI updates',
        source_sha256={name:digest(name) for name in source_names},registry_sha256=registry_sha)
    atomic(args.out/'config.json',config)
    for name in source_names:
        target=args.out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes())
    try:
        seen,sources=seen_openings('nn_checkpoints/experiments')
        prefixes=[fresh_opening(seed+100000+i,4+(i%5)*2,seen) for i in range(24)]
        atomic(args.out/'schedule.json',prefixes);atomic(args.out/'opening_exclusions.json',sources)
        # Exact component wiring on both player frames and real board/history inputs.
        checks=0
        with torch.no_grad():
            for prefix in prefixes[:4]:
                g=replay(prefix)
                for p in (1,2):
                    x=torch.from_numpy(encode(g.board,p,g.ply,g.repetitions,canonical=True)).unsqueeze(0)
                    lp,v=cross(x);expected_lp,_=new(x);_,expected_v=old(x)
                    if not torch.equal(lp,expected_lp) or not torch.equal(v,expected_v):raise ValueError('cross model wired incorrectly')
                    checks+=1
        atomic(args.out/'wiring_audit.json',dict(positions=checks,exact_policy_matches_new=True,exact_value_matches_old=True))
        log('方案固定：24全新开局组，各交换双方；新版对v3、组合版对v3、组合版对新版各48局。相同32次搜索，只诊断，不升级。完整旧CNN/value分支已精确核对。')
        specs={'new_vs_v3':('new','v3'),'hybrid_vs_v3':('hybrid','v3'),'hybrid_vs_new':('hybrid','new')}
        labels=list(specs);records={k:[] for k in specs};times={k:defaultdict(lambda:dict(seconds=0.,moves=0)) for k in specs};probes=[]
        for pair,prefix in enumerate(prefixes):
            order=labels[pair%3:]+labels[:pair%3]
            for label in order:
                a,b=specs[label]
                for side in (1,2):
                    status('matched_games',comparison=label,completed=sum(len(v) for v in records.values()),total=144,pair=pair,side=side)
                    g=replay(prefix);engines={side:Neural121(models[a],32),3-side:Neural121(models[b],32)};paths=[];move_stats=[]
                    if label=='new_vs_v3' and side==1:probes.append(copy.deepcopy(g))
                    while g.result()[0] is None:
                        name=a if g.player==side else b;began=time.monotonic();path,info=engines[g.player].think(g.board,g.player,g.ply,g.repetitions);duration=time.monotonic()-began
                        if path is None or info['root_visits']!=32:raise ValueError('wrong move/simulation budget')
                        times[label][name]['seconds']+=duration;times[label][name]['moves']+=1
                        move_stats.append(dict(player=g.player,model=name,seconds=duration,root_q=info['root_q'],visits=info['root_visits']))
                        g.play(path);paths.append(path)
                        if label=='new_vs_v3' and side==1 and g.ply-len(prefix) in (30,60) and g.result()[0] is None:probes.append(copy.deepcopy(g))
                    winner,reason=g.result();row=dict(pair=pair,side=side,opening=prefix,start_ply=len(prefix),paths=paths,plies=g.ply,winner=winner,reason=reason,move_stats=move_stats)
                    records[label].append(row)
                    with (args.out/(label+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
            log('开局组'+str(pair+1)+'/24完成；'+json.dumps({k:summary(v) for k,v in records.items()},ensure_ascii=False))
        log('144局完成，开始同局面探针与棋谱重放核验。')
        probe_rows=[]
        for i,g in enumerate(probes):
            status('same_position_probes',completed=i,total=len(probes))
            with torch.no_grad():
                x=torch.from_numpy(encode(g.board,g.player,g.ply,g.repetitions,canonical=True)).unsqueeze(0)
                _,oldv=old(x);_,newv=new(x)
            results={}
            for name in ('new','hybrid'):
                began=time.monotonic();path,info=Neural121(models[name],32).think(g.board,g.player,g.ply,g.repetitions)
                results[name]=dict(path=path,priors=info['priors'],seconds=time.monotonic()-began,root_q=info['root_q'])
            if results['new']['priors']!=results['hybrid']['priors']:raise ValueError('same-position policy priors differ; value is not only variable')
            probe_rows.append(dict(index=i,player=g.player,ply=g.ply,old_value=float(oldv[0]),new_value=float(newv[0]),
                changed_move=(results['new']['path'][0],results['new']['path'][-1])!=(results['hybrid']['path'][0],results['hybrid']['path'][-1]),
                new_seconds=results['new']['seconds'],hybrid_seconds=results['hybrid']['seconds'],new_path=results['new']['path'],hybrid_path=results['hybrid']['path']))
        atomic(args.out/'probes.json',probe_rows)
        audit_games=0
        for label,rows in records.items():
            for row in rows:
                g=replay(row['opening'])
                for path in row['paths']:g.play(path)
                if g.result()!=(row['winner'],row['reason']) or g.ply!=row['plies']:raise ValueError('trajectory replay failed')
                audit_games+=1
        for entry in config['models'].values():
            if digest(entry['path'])!=entry['sha256']:raise ValueError('model weights changed')
        if digest(REGISTRY)!=registry_sha:raise ValueError('registry changed during diagnostic')
        effect=matched_effect(records['new_vs_v3'],records['hybrid_vs_v3'],seed);direct=paired_gate(records['hybrid_vs_new'])
        support=effect['one_sided_sign_p']<=.025 and direct['one_sided_p']<=.025
        timing={label:{name:dict(**value,seconds_per_move=value['seconds']/value['moves']) for name,value in groups.items()} for label,groups in times.items()}
        result=dict(summary={k:summary(v) for k,v in records.items()},matched_effect=effect,direct_gate=direct,supports_harmful_value_replacement=support,
            timing=timing,probe_summary=dict(n=len(probe_rows),changed_moves=sum(x['changed_move'] for x in probe_rows),mean_absolute_value_change=float(np.mean([abs(x['old_value']-x['new_value']) for x in probe_rows]))),
            audit=dict(games_replayed=audit_games,models_unchanged=True,registry_unchanged=True,policy_priors_identical_in_probes=True),seconds=time.monotonic()-started,
            note='No training/registration. Cross uses extra inference; this is equal-simulation diagnostic, not an equal-time upgrade.')
        atomic(args.out/'result.json',result);log('诊断全部结束，原v3和GUI未变。'+json.dumps(result['summary'],ensure_ascii=False));status('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        status('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
