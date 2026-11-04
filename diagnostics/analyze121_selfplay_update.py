"""Diagnose the failed v2 update before generating more training games."""
from collections import Counter
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from az121 import Net121
from curriculum121 import samples_from_game
from canonical121_trial import canonical_sample
from selfplay121_retained import batch,metrics


def teacher_soft(sample):
    result=canonical_sample(sample);pi=[0.]*len(result['actions']);pi[result['target']]=1.
    return {**result,'policy':pi,'value_weight':1.}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('nn_checkpoints/experiments/analyze121_selfplay_update_20261030'));args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(301039)
    root=Path('nn_checkpoints/experiments');run=root/'selfplay121_v2_retained_20261030'
    models={'v2':Net121.load(run/'baseline.pt'),'updated_no_kl':Net121.load(run/'kl_0.0.pt'),'updated_kl':Net121.load(run/'kl_0.3.pt')}
    split=json.loads((run/'split.json').read_text());validation=set(split['validation_groups'])
    selfplay=[]
    for shard in sorted((run/'selfplay').glob('*.pt')):
        data=torch.load(shard,weights_only=True)
        for s in data['samples']:
            if s['opening_group'] in validation:selfplay.append({**s,'x':s['x'].numpy()})
    teacher=[];folder=root/'curriculum121_100_20261029';groups=set(json.loads((folder/'split.json').read_text())['validation'])
    for line in (folder/'teacher_games.jsonl').read_text().splitlines():
        g=json.loads(line)
        if g['group'] in groups:teacher.extend(teacher_soft(s) for s in samples_from_game(g))
    result={'metrics':{},'gradient_probe':[],'policy_shape':{}}
    for name,model in models.items():
        result['metrics'][name]={'selfplay_heldout':metrics(model,selfplay),'teacher_retention':metrics(model,teacher)}
        counts=Counter();hits=0;entropy=0.
        with torch.no_grad():
            for start in range(0,len(teacher),32):
                part=teacher[start:start+32];xs,ids,mask,pi,*_=batch(part);pl,v=model(xs)
                logp=pl.gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)
                hits+=int((logp.argmax(1)==pi.argmax(1)).sum());entropy+=float(-(logp.exp()*logp).sum())
                for s,value in zip(part,v):
                    p=1 if s['x'][3].sum()>0 else 2
                    counts['predicted_P1' if (p if value>=0 else 3-p)==1 else 'predicted_P2']+=1
        result['metrics'][name]['value_identity']=dict(counts)
        result['policy_shape'][name]=dict(teacher_agreement=hits/len(teacher),mean_policy_entropy=entropy/len(teacher))
    model=models['v2'];params=[p for name,p in model.named_parameters() if name.startswith(('stem.','blocks.'))]
    rng=np.random.default_rng(301039)
    for index in range(12):
        chosen=[selfplay[i] for i in rng.integers(len(selfplay),size=16)]
        xs,ids,mask,pi,z,w=batch(chosen);logits,value=model(xs)
        policy=-(pi*logits.gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)).sum(1).mean()
        critic=((value-z).square()*w).sum()/w.sum().clamp_min(1)
        pg=torch.autograd.grad(policy,params,retain_graph=True);vg=torch.autograd.grad(critic,params)
        a=torch.cat([v.flatten() for v in pg]);b=torch.cat([v.flatten() for v in vg])
        result['gradient_probe'].append(dict(policy_norm=float(a.norm()),value_norm=float(b.norm()),cosine=float(F.cosine_similarity(a,b,dim=0))))
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
