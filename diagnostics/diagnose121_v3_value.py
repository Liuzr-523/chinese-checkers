"""Development-only v3 critic calibration/aliasing diagnosis. Does not read test set."""
import hashlib,json,sys
from collections import defaultdict,Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from az121 import Net121
from canonical121_trial import canonical_sample
from curriculum121 import batch,samples_from_game

def main():
    torch.set_num_threads(1)
    source=Path('nn_checkpoints/experiments/imitate_clysto121_20261030')
    out=Path('nn_checkpoints/experiments/diagnose121_v3_value_20261030');out.mkdir(exist_ok=False)
    split=json.loads((source/'split.json').read_text());samples={'train':[],'validation':[]};counts=defaultdict(Counter)
    for line in (source/'teacher_games.jsonl').read_text().splitlines():
        g=json.loads(line)
        name=next((k for k in samples if g['group'] in split[k]),None)
        if name is None:continue
        for s in samples_from_game(g):
            s=canonical_sample(s);samples[name].append(s)
            counts[hashlib.sha256(s['x'].tobytes()).hexdigest()][s['z']]+=1
    result={'models':{},'source_sha256':hashlib.sha256((source/'teacher_games.jsonl').read_bytes()).hexdigest(),
            'identical_inputs':dict(distinct=len(counts),conflicting=sum(len(v)>1 for v in counts.values()),
                                   examples_in_conflicting=sum(sum(v.values()) for v in counts.values() if len(v)>1))}
    for name in ('v2_canonical','v3_expert_imitation'):
        path=Path('nn_checkpoints/models/checkpoints')/('neural121_'+name+'.pt');model=Net121.load(path);result['models'][name]={}
        for split_name,part in samples.items():
            buckets=defaultdict(list)
            with torch.no_grad():
                for start in range(0,len(part),32):
                    subset=part[start:start+32];xs,*_=batch(subset);_,values=model(xs)
                    for sample,v in zip(subset,values.tolist()):
                        p=1 if sample['x'][3].sum()>0 else 2;phase='endgame' if sample['endgame'] else 'earlier'
                        for key in ('all','P'+str(p),phase):buckets[key].append((v,sample['z'],p))
            stats={}
            for key,rows in buckets.items():
                a=np.asarray(rows);v,z,p=a.T;identity=np.where(p==1,1.,-1.)
                stats[key]=dict(n=len(rows),mse=float(np.mean((v-z)**2)),zero_mse=float(np.mean(z*z)),
                    p1_identity_mse=float(np.mean((identity-z)**2)),accuracy=float(np.mean((v>=0)==(z>0))),
                    mean_value=float(v.mean()),std_value=float(v.std()),saturated=float(np.mean(abs(v)>.999)),
                    predicted_p1_fraction=float(np.mean(np.where(v>=0,p,3-p)==1)),
                    true_p1_fraction=float(np.mean(np.where(z>0,p,3-p)==1)))
            result['models'][name][split_name]=stats
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
