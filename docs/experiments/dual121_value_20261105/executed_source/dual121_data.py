"""121格试点数据双权重读取与损失；屏蔽来源不可被后续override复活。"""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import rules121 as r
from az121 import CANONICAL_ENCODING
from history121_exclusions import decode_sample_inputs


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _weights(sample):
    weights=tuple(float(sample[k]) for k in ('policy_weight','value_weight'))
    if any(not math.isfinite(w) or w<0 for w in weights):raise ValueError('invalid dual weights')
    return weights

def _identity(sample):
    x=sample['x'];x=x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    identities=decode_sample_inputs(x,CANONICAL_ENCODING)
    if identities!={sample['decision_id']}:raise ValueError('encoded decision identity mismatch')
    ply=round(float(x[4].max())*240)
    if not np.allclose(x[4][x[2]>0],ply/240,atol=1e-7):raise ValueError('invalid ply plane')
    return int(sample['game_id']),ply,sample['decision_id']

def load_pilot(folder):
    folder=Path(folder);result=json.loads((folder/'result.json').read_text())
    if digest(folder/'dataset.pt')!=result['dataset_sha256']:raise ValueError('dataset hash mismatch')
    raw=torch.load(folder/'dataset.pt',weights_only=True,map_location='cpu')
    if raw['profile']!=r.PROFILE or raw['encoding']!=CANONICAL_ENCODING:raise ValueError('profile/encoding mismatch')
    schedule=json.loads((folder/'schedule.json').read_text());groups={}
    for row in schedule:
        group=int(row['group']);part=row['split']
        if part not in ('train','development') or group in groups:raise ValueError('invalid/duplicate group split')
        groups[group]=part
    metadata=json.loads((folder/'sample_metadata.json').read_text());index={}
    for row in metadata:
        identity=(int(row['game_id']),int(row['ply']),row['decision_id'])
        if identity in index:raise ValueError('duplicate metadata join key')
        if groups.get(int(row['group']))!=row['split']:raise ValueError('metadata group/split mismatch')
        index[identity]=row
    out={part:dict(value=[],policy=[]) for part in ('train','development')};seen=set();seen_actual=set();counts={}
    for part in out:
        if part not in raw['sets']:raise ValueError('missing dataset split')
        counts[part]=dict(actual=0,excluded_dual_zero=0,probes=0)
        for sample in raw['sets'][part]:
            join=_identity(sample);row=index.get(join)
            if row is None:raise ValueError('actual sample missing metadata')
            if row['split']!=part or int(row['group'])!=int(sample['group']) or groups.get(int(sample['group']))!=part:raise ValueError('actual split mismatch')
            pw,vw=_weights(sample)
            if int(row['player'])!=int(join[2][-1]):raise ValueError('metadata player mismatch')
            seen_actual.add(join)
            if (pw,vw)!=_weights(row) or float(sample['z'])!=float(row['z']):raise ValueError('actual mask/target mismatch')
            if not math.isfinite(float(sample['z'])) or float(sample['z']) not in (-1.,0.,1.):raise ValueError('invalid target')
            if not row['eligible'] and (pw or vw):raise ValueError('excluded sample resurrected')
            if row['policy_source']!='derived5' and pw>0:raise ValueError('nonteacher actual policy activated')
            if row['terminal_reason']=='max_ply' and vw>0:raise ValueError('truncated outcome activated')
            sid=f'actual:{join[0]}:{join[1]}'
            if sid in seen:raise ValueError('duplicate actual sample id')
            seen.add(sid);counts[part]['actual']+=1
            merged={**sample,**{k:row[k] for k in ('player','ply','terminal_distance','policy_source')},'sample_id':sid,'split':part}
            if vw>0:out[part]['value'].append(merged.copy())
            if pw>0:out[part]['policy'].append(merged.copy())
            if not pw and not vw:counts[part]['excluded_dual_zero']+=1
    expected={join for join,row in index.items() if any(_weights(row))}
    if not expected.issubset(seen_actual):raise ValueError('active metadata sample missing from dataset')
    for sample in raw['policy_probes']:
        join=_identity(sample);row=index.get(join);part=sample['split'];pw,vw=_weights(sample)
        if row is None or part not in out or row['split']!=part or int(row['group'])!=int(sample['group']) or groups.get(int(sample['group']))!=part:raise ValueError('probe join/split mismatch')
        if not row['eligible']:raise ValueError('excluded probe resurrected')
        if vw!=0 or pw!=1 or float(sample['z'])!=0 or sample['policy_source']!='derived5_probe':raise ValueError('invalid probe dual mask/source')
        sid=f'probe:{join[0]}:{join[1]}'
        if sid in seen:raise ValueError('duplicate probe id')
        seen.add(sid);merged={**sample,'player':row['player'],'ply':join[1],'terminal_distance':row['terminal_distance'],'sample_id':sid};out[part]['policy'].append(merged);counts[part]['probes']+=1
    for part in out:
        counts[part]['metadata_excluded_dual_zero']=sum(row['split']==part and not any(_weights(row)) for row in metadata)
        counts[part].update(value=len(out[part]['value']),policy=len(out[part]['policy']))
    out['pool_counts']=counts;out['manifest']={name:digest(folder/name) for name in ('dataset.pt','result.json','schedule.json','sample_metadata.json')}
    return out


def dual_loss(logits,value,samples):
    if logits.ndim!=2 or value.ndim!=1:raise ValueError('expected logits[N,A], value[N]')
    if len(samples)!=len(logits) or len(value)!=len(samples) or not samples:raise ValueError('nonempty matching batch required')
    weights=[_weights(s) for s in samples];pw=logits.new_tensor([w[0] for w in weights]);vw=value.new_tensor([w[1] for w in weights]);targets=value.new_tensor([s['z'] for s in samples])
    if not bool(torch.isfinite(targets).all()):raise ValueError('nonfinite target')
    ce=[]
    for i,s in enumerate(samples):
        ids=torch.as_tensor(s['actions'],dtype=torch.long,device=logits.device)
        if not len(ids) or bool(((ids<0)|(ids>=logits.shape[1])).any()) or len(set(s['actions']))!=len(ids) or not 0<=int(s['target'])<len(ids):raise ValueError('invalid legal action target')
        ce.append(F.cross_entropy(logits[i,ids].unsqueeze(0),torch.tensor([int(s['target'])],device=logits.device)))
    pden=torch.where(pw.sum()>0,pw.sum(),pw.new_tensor(1.));vden=torch.where(vw.sum()>0,vw.sum(),vw.new_tensor(1.))
    policy=(torch.stack(ce)*pw).sum()/pden;mse=((value-targets).square()*vw).sum()/vden
    return dict(policy_ce=policy,value_mse=mse,policy_weight_sum=pw.sum(),value_weight_sum=vw.sum())
