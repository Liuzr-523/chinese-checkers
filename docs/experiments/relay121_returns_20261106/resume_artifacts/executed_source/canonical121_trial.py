"""Controlled representation study; no rotated games or assumed rule symmetry.
Absolute vs current-player coordinate frame, same original turn identity and labels.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback
import torch
from az121 import Net121,Neural121,ENCODING,CANONICAL_ENCODING,action_id
from curriculum121 import atomic,samples_from_game,train_curriculum,metrics,opening,replay,finish
from engine_c121 import CSearch121
from iterate121 import summary,paired_gate
import rules121 as r


def canonical_sample(sample):
    if sample.get('encoding',ENCODING)!=ENCODING:raise ValueError('expected original absolute sample')
    player=1 if sample['x'][3].sum()>0 else 2
    return {**sample,'x':sample['x'][:,::-1,::-1].copy() if player==2 else sample['x'].copy(),
            'actions':[action_id(a//121,a%121,player,True) for a in sample['actions']],
            'encoding':CANONICAL_ENCODING}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=1600);ap.add_argument('--seed',type=int,default=301037)
    args=ap.parse_args();torch.set_num_threads(1)
    parent=Path('nn_checkpoints/experiments/correct121_failures_20261030_b')
    baseline=Net121.load(parent/'candidate.pt');args.out.mkdir(parents=True,exist_ok=False)
    lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);began=time.monotonic()
    def update(stage,**data):atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-began,**data))
    def record(stage,row):
        with (args.out/(stage+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
    config=dict(model=str((parent/'candidate.pt').resolve()),model_sha256=hashlib.sha256((parent/'candidate.pt').read_bytes()).hexdigest(),
                seed=args.seed,steps=args.steps,profile=r.PROFILE,simulations=32,confirmation_games=60,independent_gate_p=.01,
                training='Controlled teacher retraining: absolute vs current-player canonical coordinates, same labels and init',
                encoding_study=[ENCODING,CANONICAL_ENCODING],first_player_identity_preserved=True,automatic_promotion=False,
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['canonical121_trial.py','az121.py','rules121.py','curriculum121.py','iterate121.py']})
    atomic(args.out/'config.json',config);snapshot=args.out/'source';snapshot.mkdir()
    for name in config['source_sha256']:(snapshot/name).write_bytes(Path(name).read_bytes())
    try:
        old=Path('nn_checkpoints/experiments/curriculum121_100_20261029')
        data_sources={'teacher_games.jsonl':old/'teacher_games.jsonl','split.json':old/'split.json',
                      'corrections.jsonl':parent/'corrections.jsonl','correction_split.json':parent/'correction_split.json'}
        frozen=args.out/'data_sources';frozen.mkdir()
        config['data_sha256']={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in data_sources.items()}
        for name,path in data_sources.items():shutil.copyfile(path,frozen/name)
        atomic(args.out/'config.json',config)
        splits=json.loads((frozen/'split.json').read_text())
        old_train=[];old_val=[];new_train=[];new_val=[]
        for line in (frozen/'teacher_games.jsonl').read_text().splitlines():
            g=json.loads(line)
            if g['group'] in splits['train']:old_train.extend(samples_from_game(g))
            elif g['group'] in splits['validation']:old_val.extend(samples_from_game(g))
        val_ids=set(json.loads((frozen/'correction_split.json').read_text())['validation'])
        for line in (frozen/'corrections.jsonl').read_text().splitlines():
            g=json.loads(line);(new_val if g['group'] in val_ids else new_train).extend(samples_from_game(g))
        rng=random.Random(args.seed);n=max(len(old_train),len(new_train))
        training=[rng.choice(old_train) for _ in range(n)]+[rng.choice(new_train) for _ in range(n)]
        validation=old_val+new_val;studies={};models={}
        atomic(args.out/'data.json',dict(train=len(training),validation=len(validation),old_train=len(old_train),new_train=len(new_train),source='same prior train/validation groups, no final test reuse'))
        for name,canonical in [('absolute',False),('canonical',True)]:
            folder=args.out/name;folder.mkdir()
            # Same architecture, parameters and initialization; only representation differs.
            torch.manual_seed(args.seed);model=Net121(canonical=canonical)
            train_set=[canonical_sample(s) for s in training] if canonical else training
            val_set=[canonical_sample(s) for s in validation] if canonical else validation
            def progress(stage,**data):update('training_'+name,**data)
            selected=train_curriculum(model,train_set,val_set,args.steps,args.seed,folder,progress)
            models[name]=selected
            scores=dict(retention=metrics(selected,[canonical_sample(s) for s in old_val] if canonical else old_val),
                        correction=metrics(selected,[canonical_sample(s) for s in new_val] if canonical else new_val))
            studies[name]=scores;atomic(args.out/'representation_study.json',studies)
        arm=min(studies,key=lambda name:sum(s['policy_ce']+s['value_mse'] for s in studies[name].values()))
        candidate=models[arm];candidate.save(args.out/'candidate.pt');config['encoding']=candidate.encoding
        atomic(args.out/'config.json',config);atomic(args.out/'selection.json',dict(arm=arm,encoding=candidate.encoding,study=studies))
        matches=[]
        for pair in range(30):
            prefix=opening(args.seed+100000+pair,0 if pair==0 else 4+(pair%5)*2)
            for side in (1,2):
                update('independent_confirmation',arm=arm,completed=len(matches),total=60)
                row=finish(replay(prefix),{side:Neural121(candidate,32),3-side:Neural121(baseline,32)})
                row.update(pair=pair,side=side,opening=prefix);matches.append(row);record('confirmation',row)
                print(json.dumps(dict(stage='confirmation',arm=arm,**summary(matches))),flush=True)
        fixed={}
        for name,network in [('control',baseline),('candidate',candidate)]:
            rows=[]
            for pair in range(10):
                prefix=opening(args.seed+200000+pair,0 if pair==0 else 4+(pair%4)*2)
                for side in (1,2):
                    update('fixed_c_'+name,completed=len(rows),total=20)
                    row=finish(replay(prefix),{side:Neural121(network,32),3-side:CSearch121(seconds=.03,max_depth=10)})
                    row.update(pair=pair,side=side,opening=prefix);rows.append(row);record('fixed_c_'+name,row)
            fixed[name]=summary(rows)
        gate=paired_gate(matches)
        result=dict(arm=arm,encoding=candidate.encoding,study=studies,confirmation=summary(matches),paired_gate=gate,fixed_c=fixed,
                    qualified=gate['passed'] and gate['one_sided_p']<=.01 and fixed['candidate']['points']>=fixed['control']['points'],
                    promoted=False,seconds=time.monotonic()-began,note='Teacher representation study, not pure selfplay or game-rule symmetry augmentation.')
        atomic(args.out/'result.json',result);update('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        update('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
