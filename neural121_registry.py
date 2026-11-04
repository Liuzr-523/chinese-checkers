"""Audited, reversible registration of independently qualified neural models."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
from curriculum121 import atomic,replay
from iterate121 import paired_gate,summary
import rules121 as r

ROOT=Path(__file__).resolve().parent
REGISTRY=ROOT/'nn_checkpoints/models/champions/neural121.json'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audit_game_counts(config):
    confirmation=config.get('confirmation_games',60)
    fixed=config.get('fixed_c_games_per_model',20)
    if type(confirmation) is not int or confirmation<60 or confirmation%2:
        raise ValueError('confirmation budget must be even and at least 60')
    if type(fixed) is not int or fixed<20 or fixed%2:
        raise ValueError('fixed-C budget must be even and at least 20')
    return [('confirmation',confirmation),('fixed_c_control',fixed),('fixed_c_candidate',fixed)]


def audit_experiment(folder):
    folder=Path(folder);config=json.loads((folder/'config.json').read_text());result=json.loads((folder/'result.json').read_text())
    if config.get('profile')!=r.PROFILE or not result.get('qualified'):raise ValueError('unqualified rule profile or result')
    for name,sha in config['source_sha256'].items():
        if digest(folder/'source'/name)!=sha:raise ValueError('source snapshot mismatch: '+name)
    if digest(config['model'])!=config['model_sha256']:raise ValueError('baseline checkpoint changed')
    for name,sha in config.get('data_sha256',{}).items():
        if digest(folder/'data_sources'/name)!=sha:raise ValueError('data snapshot mismatch: '+name)
    logs={}
    verified=0
    for name,expected in audit_game_counts(config):
        rows=[json.loads(line) for line in (folder/(name+'.jsonl')).read_text().splitlines()]
        if len(rows)!=expected:raise ValueError('incomplete games: '+name)
        pairs=defaultdict(list)
        for row in rows:
            pairs[row['pair']].append(row)
            g=replay(row['opening'])
            if g.ply!=row['start_ply']:raise ValueError('incorrect start ply')
            for path in row['paths']:g.play(path)
            if g.result()!=(row['winner'],row['reason']) or g.ply!=row['plies']:raise ValueError('incorrect game outcome')
            verified+=1
        for pair in pairs.values():
            if len(pair)!=2 or {x['side'] for x in pair}!={1,2} or pair[0]['opening']!=pair[1]['opening']:
                raise ValueError('incomplete or mismatched opening pair')
        logs[name]=rows
    for name,expected in [('confirmation',result['confirmation']),('fixed_c_control',result['fixed_c']['control']),('fixed_c_candidate',result['fixed_c']['candidate'])]:
        if summary(logs[name])!=expected:raise ValueError('score mismatch '+name)
    if [x['opening'] for x in logs['fixed_c_control']]!=[x['opening'] for x in logs['fixed_c_candidate']]:raise ValueError('C schedules differ')
    gate=paired_gate(logs['confirmation'])
    threshold=min(.05,float(config.get('independent_gate_p',.05)))
    if not gate['passed'] or gate!=result['paired_gate'] or gate['one_sided_p']>threshold:raise ValueError('confirmation gate failed')
    if summary(logs['fixed_c_candidate'])['points']<summary(logs['fixed_c_control'])['points']:raise ValueError('C non-regression threshold failed')
    from az121 import Net121
    audited_model=Net121.load(folder/'candidate.pt')
    return dict(encoding=audited_model.encoding,verified_games=verified,gate=gate,confirmation=result['confirmation'],fixed_c=result['fixed_c'],
                experiment=str(folder.resolve()),checkpoint_sha256=digest(folder/'candidate.pt'),
                source_config_sha256=digest(folder/'config.json'),scope='Improved versus prior neural baseline; still much weaker than fixed C reference.')

def register(folder,version):
    if not version or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in version):raise ValueError('invalid version')
    folder=Path(folder);evidence=audit_experiment(folder)
    config=json.loads((folder/'config.json').read_text())
    target=ROOT/'nn_checkpoints/models/checkpoints'/('neural121_'+version+'.pt')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and digest(target)!=evidence['checkpoint_sha256']:raise ValueError('immutable checkpoint collision')
    if not target.exists():
        temporary=target.with_suffix('.tmp');shutil.copyfile(folder/'candidate.pt',temporary);temporary.replace(target)
    previous=json.loads(REGISTRY.read_text()) if REGISTRY.exists() else None
    entry=dict(schema='neural121-registry-v1',version=version,profile=r.PROFILE,encoding=evidence['encoding'],
               checkpoint=str(target.relative_to(ROOT)),sha256=evidence['checkpoint_sha256'],
               simulations=int(config.get('simulations',32)),value_scale=float(config.get('value_scale',1.)),root_safety=bool(config.get('root_safety',False)),heuristic_mix=float(config.get('heuristic_mix',0.)),fpu_mode=config.get('fpu_mode','zero'),policy_temperature=float(config.get('policy_temperature',1.)),
               training_source=config.get('training',config.get('kind',config.get('change','C counterexample teacher correction with retention'))),evidence=evidence)
    REGISTRY.parent.mkdir(parents=True,exist_ok=True)
    if previous and previous!=entry:
        backup=REGISTRY.parent/('neural121_previous_'+previous['version']+'.json')
        if not backup.exists():atomic(backup,previous)
    atomic(folder/'registration_audit.json',evidence);atomic(REGISTRY,entry)
    return entry

def load_registered():
    entry=json.loads(REGISTRY.read_text())
    if entry.get('schema')!='neural121-registry-v1' or entry.get('profile')!=r.PROFILE:raise ValueError('incompatible neural registry')
    checkpoint=(ROOT/entry['checkpoint']).resolve()
    if not checkpoint.is_relative_to((ROOT/'nn_checkpoints/models').resolve()):raise ValueError('invalid registered checkpoint path')
    if digest(checkpoint)!=entry['sha256']:raise ValueError('registered model hash mismatch')
    import torch
    torch.set_num_threads(1)
    from az121 import Net121,Neural121
    model=Net121.load(checkpoint)
    if entry.get('encoding',model.encoding)!=model.encoding:raise ValueError('registered encoding mismatch')
    engine=Neural121(model,entry['simulations'],entry['value_scale'],root_safety=entry.get('root_safety',False),heuristic_mix=entry.get('heuristic_mix',0.),fpu_mode=entry.get('fpu_mode','zero'),policy_temperature=entry.get('policy_temperature',1.))
    engine.name='神经网络 '+entry['version']+' / MCTS'+str(entry['simulations'])
    return engine

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,required=True);parser.add_argument('--version',required=True)
    args=parser.parse_args();print(json.dumps(register(args.experiment,args.version),indent=2))
