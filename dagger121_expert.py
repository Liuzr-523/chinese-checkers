"""Expert corrections on v3-visited states, with teacher-only retention control.
Strict121 Clysto-evaluator derivative, not original Clysto or pure selfplay.
No counterfactual value labels: CNN/value frozen; only policy heads learn.
"""
import argparse,copy,fcntl,hashlib,json,os,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
import rules121 as r
from az121 import Net121,Neural121,encode,action_id
from canonical121_trial import canonical_sample
from curriculum121 import atomic,opening,replay,samples_from_game,finish,split_groups
from engine_clysto121 import ClystoDerived121
from engine_c121 import CSearch121
from iterate121 import summary,paired_gate
from reanalyze121_policy import features,logits_from_features,batch,loss_on


def correction_sample(model,g,path):
    moves=r.get_all_moves(g.board,g.player)
    actions=[action_id(s,p[-1],g.player,model.canonical) for s,p in moves]
    chosen=action_id(path[0],path[-1],g.player,model.canonical)
    if not r.valid_path(g.board,g.player,path) or chosen not in actions:raise ValueError('illegal expert annotation')
    x=torch.from_numpy(encode(g.board,g.player,g.ply,g.repetitions,canonical=model.canonical))
    with torch.no_grad():cells=features(model,x.unsqueeze(0))[0].detach().clone()
    pi=[0.]*len(actions);pi[actions.index(chosen)]=1.
    return dict(features=cells,actions=actions,clean=pi)


def retain_sample(model,s):
    s=canonical_sample(s);x=torch.from_numpy(s['x'])
    with torch.no_grad():cells=features(model,x.unsqueeze(0))[0].detach().clone()
    pi=[0.]*len(s['actions']);pi[s['target']]=1.
    return dict(features=cells,actions=s['actions'],clean=pi)


def teacher_qualified(rows):
    return len(rows)==40 and summary(rows)['points']>=28 and paired_gate(rows)['one_sided_p']<=.01


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=301045);args=ap.parse_args()
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    model_path=Path('nn_checkpoints/models/checkpoints/neural121_v3_expert_imitation.pt');baseline=Net121.load(model_path)
    args.out.mkdir(parents=True,exist_ok=False);lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic()
    def update(stage,**kw):atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-start,**kw))
    def record(name,row):
        with (args.out/(name+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
    import clysto121_native
    source_names=['dagger121_expert.py','az121.py','rules121.py','curriculum121.py','canonical121_trial.py','engine_clysto121.py','engine_c121.py','iterate121.py','reanalyze121_policy.py','clysto121_generated/manifest.json','clysto121_generated/core.c',Path(clysto121_native.__file__).name]
    config=dict(model=str(model_path.resolve()),model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),profile=r.PROFILE,encoding=baseline.encoding,
                seed=args.seed,simulations=32,rollout_games=40,expert_seconds=.04,expert_depth=10,teacher_gate='>=28/40 points vs v3 and opening-pair p<=.01',
                training='Expert decision correction on v3-visited strict121 states + retention; policy heads only. No value labels from unplayed corrections; not pure selfplay.',
                arms=['retention_only','dagger_mix'],steps_per_arm=800,learning_rate=.0001,kl_to_v3=.1,
                selection='Mean correction/retention validation hard CE; baseline included; new test groups never used for selection',
                confirmation_games=60,independent_gate_p=.01,automatic_promotion=False,
                source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_names})
    for name in source_names:
        target=args.out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes())
    old=Path('nn_checkpoints/experiments/imitate_clysto121_20261030');data_sources=args.out/'data_sources';data_sources.mkdir()
    config['data_sha256']={}
    for name in ['teacher_games.jsonl','split.json']:
        data=(old/name).read_bytes();(data_sources/name).write_bytes(data);config['data_sha256'][name]=hashlib.sha256(data).hexdigest()
    atomic(args.out/'config.json',config)
    try:
        splits=split_groups(20,args.seed);atomic(args.out/'split.json',{k:sorted(v) for k,v in splits.items()})
        sets={k:[] for k in splits};rows=[];agreements={k:[0,0] for k in splits};teacher=ClystoDerived121(seconds=.04,max_depth=10)
        for pair in range(20):
            prefix=opening(args.seed+10000+pair,4+(pair%5)*4);part=next(k for k,v in splits.items() if pair in v)
            for side in (1,2):
                update('expert_correction_rollout',completed=len(rows),total=40)
                g=replay(prefix);paths=[];annotations=[];neural=Neural121(baseline,32)
                while g.result()[0] is None:
                    if g.player==side:path,_=teacher.think(g.board,g.player,g.ply,g.repetitions)
                    else:
                        label,label_info=teacher.think(g.board,g.player,g.ply,g.repetitions)
                        path,_=neural.think(g.board,g.player,g.ply,g.repetitions)
                        sets[part].append(correction_sample(baseline,g,label))
                        same=(path[0],path[-1])==(label[0],label[-1]);agreements[part][0]+=same;agreements[part][1]+=1
                        annotations.append(dict(ply=g.ply,player=g.player,expert_path=label,played_path=path,expert_depth=label_info['depth_reached']))
                    if path is None:raise ValueError('no move in live rollout')
                    g.play(path);paths.append(path)
                win,reason=g.result();row=dict(pair=pair,side=side,opening=prefix,start_ply=len(prefix),plies=g.ply,paths=paths,winner=win,reason=reason)
                rows.append(row);record('expert_rollout',row);record('annotations',dict(game_id=len(rows)-1,pair=pair,split=part,annotations=annotations))
        qualified=teacher_qualified(rows)
        atomic(args.out/'teacher_qualification.json',dict(summary=summary(rows),gate=paired_gate(rows),passed=qualified,policy_agreement=agreements))
        if not qualified:
            result=dict(qualified=False,promoted=False,training_started=False,note='Expert failed qualification versus v3. No update.',seconds=time.monotonic()-start)
            atomic(args.out/'result.json',result);update('complete',result=result);return
        old_split=json.loads((data_sources/'split.json').read_text());retention={'train':[],'validation':[]}
        for line in (data_sources/'teacher_games.jsonl').read_text().splitlines():
            game=json.loads(line);key=next((k for k in retention if game['group'] in old_split[k]),None)
            if key:retention[key].extend(retain_sample(baseline,s) for s in samples_from_game(game))
        if any(not v for v in sets.values()) or any(not v for v in retention.values()):raise ValueError('empty split')
        for name in ('expert_rollout.jsonl','annotations.jsonl'):
            content=(args.out/name).read_bytes();(data_sources/name).write_bytes(content);config['data_sha256'][name]=hashlib.sha256(content).hexdigest()
        atomic(args.out/'config.json',config)
        torch.save(dict(encoding=baseline.encoding,sets=sets,retention=retention),args.out/'cached_data.pt')
        atomic(args.out/'data.json',dict(corrections={k:len(v) for k,v in sets.items()},retention={k:len(v) for k,v in retention.items()},value_labels_generated=0))
        def criterion(model):return .5*(loss_on(model,sets['validation'])+loss_on(model,retention['validation']))
        scores={'baseline':criterion(baseline)};models={'baseline':baseline}
        for arm in config['arms']:
            model=copy.deepcopy(baseline);model.requires_grad_(False)
            for head in (model.src,model.dst,model.bias):head.requires_grad_(True)
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=.0001)
            rng=np.random.default_rng(args.seed);best=float('inf');history=[]
            for step in range(800):
                first=sets['train'] if arm=='dagger_mix' else retention['train']
                chosen=[first[int(i)] for i in rng.integers(len(first),size=8)]+[retention['train'][int(i)] for i in rng.integers(len(retention['train']),size=8)]
                cells,ids,mask,pi=batch(chosen,'clean');logp=logits_from_features(model,cells).gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)
                with torch.no_grad():oldlogp=logits_from_features(baseline,cells).gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)
                loss=-(pi*logp).sum(1).mean()+.1*(oldlogp.exp()*(oldlogp-logp)).sum(1).mean()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                if (step+1)%50==0:
                    score=criterion(model);history.append(dict(step=step+1,loss=float(loss.detach()),validation_ce=score))
                    if score<best:best=score;model.save(args.out/(arm+'.pt'))
                    atomic(args.out/(arm+'_training.json'),history);update('policy_head_training',arm=arm,step=step+1,total=800)
            selected=Net121.load(args.out/(arm+'.pt'))
            for k,v in selected.state_dict().items():
                if not k.startswith(('src.','dst.','bias.')) and not torch.equal(v,baseline.state_dict()[k]):raise ValueError('frozen parameter changed')
            models[arm]=selected;scores[arm]=criterion(selected)
        arm=min(scores,key=scores.get)
        atomic(args.out/'selection.json',dict(selected=arm,validation_ce=scores,criterion=config['selection']))
        if arm=='baseline':
            result=dict(qualified=False,promoted=False,training_started=True,selected=arm,note='No validation improvement over unchanged v3; stop before matches.',seconds=time.monotonic()-start)
            atomic(args.out/'result.json',result);update('complete',result=result);return
        candidate=models[arm];candidate.save(args.out/'candidate.pt')
        atomic(args.out/'heldout_test.json',dict(candidate=loss_on(candidate,sets['test']),baseline=loss_on(baseline,sets['test'])))
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
        result=dict(selected=arm,confirmation=summary(matches),paired_gate=gate,fixed_c=fixed,backbone_value_unchanged=True,
                    qualified=gate['one_sided_p']<=.01 and fixed['candidate']['points']>=fixed['control']['points'],
                    promoted=False,training_started=True,seconds=time.monotonic()-start)
        atomic(args.out/'result.json',result);update('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        update('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
