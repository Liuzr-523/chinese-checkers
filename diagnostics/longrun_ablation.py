"""Post-run attribution probe, not used for model selection or training."""
import sys,json,pickle,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nn_resnet import NetResNet
from az_longrun import play,atomic_pickle,atomic_json

class Hybrid(torch.nn.Module):
    def __init__(self,policy,value):
        super().__init__();self.policy=policy;self.value=value
    def forward(self,x):
        p,_=self.policy(x);_,v=self.value(x)
        return p,v

def main():
    torch.set_num_threads(1)
    root=Path('nn_checkpoints/experiments/longrun_20261022')
    base=NetResNet.load(root/'bootstrap.pt').eval()
    trained=NetResNet.load(root/'round_05/candidate.pt').eval()
    # Using whole independent networks avoids mixing incompatible hidden features.
    variants={'trained_policy_base_value':Hybrid(trained,base).eval(),
              'base_policy_trained_value':Hybrid(base,trained).eval()}
    output=root/'ablation';output.mkdir(exist_ok=True)
    result={}
    for label,model in variants.items():
        records=[];start=time.monotonic()
        for i in range(4):
            side=1+i%2
            rec=play([model,base] if side==1 else [base,model],990000+i//2,64,[.995,.995],opening=4)
            rec['candidate_side']=side;records.append(rec)
            atomic_pickle(output/(label+'.pkl'),records)
            print(label,i+1,rec['winner'],rec['reason'],len(rec['steps']),flush=True)
        result[label]=dict(wins=sum(r['winner']==r['candidate_side'] for r in records),
                           losses=sum(r['winner']==3-r['candidate_side'] for r in records),
                           unfinished=sum(r['winner'] is None for r in records),seconds=time.monotonic()-start)
        atomic_json(output/'result.json',result)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
