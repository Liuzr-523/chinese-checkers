"""Bounded expert-qualification -> imitation -> independent tests.
Teacher is an explicitly adapted Clysto positional evaluator, NOT original Clysto.
Never uses 81-cell games/models/book. Never promotes automatically.
"""
import argparse,copy,fcntl,hashlib,json,os,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import rules121 as r
from az121 import Net121,Neural121
from canonical121_trial import canonical_sample
from curriculum121 import atomic,opening,replay,finish,samples_from_game,batch,metrics,split_groups
from engine_c121 import CSearch121
from engine_clysto121 import ClystoDerived121
from iterate121 import summary,paired_gate


def expert_gate(rows):
    stats=summary(rows);gate=paired_gate(rows)
    return stats['games']==40 and stats['points']>=28 and gate['one_sided_p']<=.01


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=301043)
    args=ap.parse_args();torch.set_num_threads(1);torch.manual_seed(args.seed)
    args.out.mkdir(parents=True,exist_ok=False);lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();baseline_path=Path('nn_checkpoints/models/checkpoints/neural121_v2_canonical.pt');baseline=Net121.load(baseline_path)
    def update(stage,**kw):atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-start,**kw))
    def record(stage,row):
        with (args.out/(stage+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
    sources=['imitate_clysto121.py','build_clysto121.py','engine_clysto121.py','verify_clysto121.py','ccore121.c','ccore121_tables.h','clysto121_generated/core.c','clysto121_generated/manifest.json','clysto121_generated/verification.json','clysto121_generated/upstream_game.hpp','az121.py','rules121.py','curriculum121.py','canonical121_trial.py','iterate121.py']
    import clysto121_native
    sources.append(Path(clysto121_native.__file__).name)
    config=dict(model=str(baseline_path.resolve()),model_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),profile=r.PROFILE,encoding=baseline.encoding,
        seed=args.seed,simulations=32,training='Supervised imitation of Clysto positional-evaluation derivative in strict121; not original Clysto, not pure selfplay.',
        expert_qualification_games=40,expert_gate='40 games vs v2, >=28 points and opening-pair p<=.01; otherwise STOP without training',
        teacher_seconds=.04,teacher_max_depth=10,teacher_c_comparison_games=20,teacher_games=80,steps_per_arm=800,arms=['policy_heads','joint_low_lr'],
        selection='validation policy CE + value MSE, best checkpoint each arm, select once before fresh matches',
        confirmation_games=60,independent_gate_p=.01,fixed_c_games_per_model=20,automatic_promotion=False,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    for name in sources:
        dest=args.out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes())
    atomic(args.out/'config.json',config)
    try:
        verification=json.loads(Path('clysto121_generated/verification.json').read_text())
        for name,sha in verification['sha256'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError('teacher changed since verification')
        teacher=ClystoDerived121(seconds=.04,max_depth=10)
        rows=[]
        for pair in range(20):
            prefix=opening(args.seed+10000+pair,0 if pair==0 else 4+(pair%5)*2)
            for side in (1,2):
                update('expert_qualification',completed=len(rows),total=40)
                row=finish(replay(prefix),{side:teacher,3-side:Neural121(baseline,32)})
                row.update(pair=pair,side=side,opening=prefix);rows.append(row);record('expert_qualification',row)
        qualified=expert_gate(rows)
        qualification=dict(summary=summary(rows),paired_gate=paired_gate(rows),qualified_as_nn_teacher=qualified,
                           scope='Adapted evaluator candidate only, not original Clysto; extra offline compute allowed for expert labels.')
        atomic(args.out/'expert_qualification.json',qualification)
        if not qualified:
            result=dict(qualified=False,training_started=False,teacher_qualification=qualification,promoted=False,seconds=time.monotonic()-start,
                        note='Expert failed predeclared test; no imitation data generated and no NN updated.')
            atomic(args.out/'result.json',result);update('complete',result=result);return
        comparisons=[]
        for pair in range(10):
            prefix=opening(args.seed+20000+pair,4+(pair%4)*2)
            for side in (1,2):
                update('expert_vs_c',completed=len(comparisons),total=20)
                row=finish(replay(prefix),{side:teacher,3-side:CSearch121(seconds=.04,max_depth=10)})
                row.update(pair=pair,side=side,opening=prefix);comparisons.append(row);record('expert_vs_c',row)
        atomic(args.out/'expert_vs_c.json',summary(comparisons))
        splits=split_groups(40,args.seed);atomic(args.out/'split.json',{k:sorted(v) for k,v in splits.items()})
        games=[]
        for group in range(40):
            prefix=opening(args.seed+30000+group,4+(group%5)*4)
            for side in (1,2):
                update('expert_generation',completed=len(games),total=80)
                row=finish(replay(prefix),{side:ClystoDerived121(seconds=.08,max_depth=12),3-side:teacher})
                row.update(group=group,stronger_side=side,opening=prefix);games.append(row);record('teacher_games',row)
        data_sources=args.out/'data_sources';data_sources.mkdir()
        config['data_sha256']={}
        for name in ('teacher_games.jsonl','split.json'):
            content=(args.out/name).read_bytes();(data_sources/name).write_bytes(content)
            config['data_sha256'][name]=hashlib.sha256(content).hexdigest()
        atomic(args.out/'config.json',config)
        sets={name:[canonical_sample(s) for g in games if g['group'] in groups for s in samples_from_game(g)] for name,groups in splits.items()}
        if any(not part for part in sets.values()):raise ValueError('No genuine decisive games in a split; refuse training')
        atomic(args.out/'data.json',{name:dict(samples=len(part),p1_wins=sum(s['winner']==1 for s in part),p2_wins=sum(s['winner']==2 for s in part)) for name,part in sets.items()})
        torch.save(dict(encoding=baseline.encoding,sets=sets),args.out/'expert_data.pt')
        candidates={};scores={}
        for arm in config['arms']:
            model=copy.deepcopy(baseline)
            if arm=='policy_heads':
                model.requires_grad_(False)
                for head in [model.src,model.dst,model.bias]:head.requires_grad_(True)
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003 if arm=='policy_heads' else .00003,weight_decay=.0001)
            rng=np.random.default_rng(args.seed);best=float('inf');history=[]
            for step in range(800):
                chosen=[sets['train'][int(i)] for i in rng.integers(len(sets['train']),size=16)]
                xs,ids,mask,target,z=batch(chosen);logits,value=model(xs)
                loss=F.cross_entropy(logits.gather(1,ids).masked_fill(~mask,-1e9),target)
                if arm!='policy_heads':loss=loss+F.mse_loss(value,z)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                if (step+1)%50==0:
                    val=metrics(model,sets['validation']);criterion=val['policy_ce']+val['value_mse']
                    if not np.isfinite(criterion):raise ValueError('nonfinite training')
                    if criterion<best:best=criterion;model.save(args.out/(arm+'.pt'))
                    history.append(dict(step=step+1,loss=float(loss.detach()),validation=val));atomic(args.out/(arm+'_training.json'),history)
                    update('imitation_training',arm=arm,step=step+1,total=800)
            candidates[arm]=Net121.load(args.out/(arm+'.pt'));scores[arm]=metrics(candidates[arm],sets['validation'])
        selected=min(scores,key=lambda arm:scores[arm]['policy_ce']+scores[arm]['value_mse']);candidate=candidates[selected];candidate.save(args.out/'candidate.pt')
        atomic(args.out/'selection.json',dict(selected=selected,validation=scores,heldout_test=metrics(candidate,sets['test']),baseline_test=metrics(baseline,sets['test'])))
        matches=[]
        for pair in range(30):
            prefix=opening(args.seed+100000+pair,0 if pair==0 else 4+(pair%5)*2)
            for side in (1,2):
                update('independent_confirmation',completed=len(matches),total=60)
                row=finish(replay(prefix),{side:Neural121(candidate,32),3-side:Neural121(baseline,32)})
                row.update(pair=pair,side=side,opening=prefix);matches.append(row);record('confirmation',row)
        fixed={}
        for name,model in [('control',baseline),('candidate',candidate)]:
            rows=[]
            for pair in range(10):
                prefix=opening(args.seed+200000+pair,0 if pair==0 else 4+(pair%4)*2)
                for side in (1,2):
                    update('fixed_c_'+name,completed=len(rows),total=20)
                    row=finish(replay(prefix),{side:Neural121(model,32),3-side:CSearch121(seconds=.03,max_depth=10)})
                    row.update(pair=pair,side=side,opening=prefix);rows.append(row);record('fixed_c_'+name,row)
            fixed[name]=summary(rows)
        gate=paired_gate(matches)
        result=dict(selected=selected,confirmation=summary(matches),paired_gate=gate,fixed_c=fixed,teacher_qualification=qualification,
                    qualified=gate['one_sided_p']<=.01 and fixed['candidate']['points']>=fixed['control']['points'],
                    promoted=False,training_started=True,seconds=time.monotonic()-start)
        atomic(args.out/'result.json',result);update('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        update('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
