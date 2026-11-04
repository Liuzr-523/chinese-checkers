"""Fixed-policy ablation: test 512 simulations with/without NN value vs 64 baseline.
Runs locally; saves each legal, validated game and counts unfinished separately.
"""
import json,os,pickle,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nn_resnet import NetResNet
from az_longrun import play,atomic_json,atomic_pickle,stamp

class NoValue(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model
    def forward(self,x):
        policy,value=self.model(x)
        return policy,torch.zeros_like(value)


def main():
    torch.set_num_threads(1)
    root=Path('nn_checkpoints/experiments/value_ablation_20261027')
    root.mkdir(parents=True,exist_ok=False)
    source=Path('nn_checkpoints/experiments/search_budget_20261022/model.pt')
    model=NetResNet.load(source).eval();zero=NoValue(model).eval()
    # Compare same four opening seeds and colors; no automatic champion promotion.
    results=[];started=time.monotonic()
    try:
        for group in ('value_on','value_zero'):
            for i in range(4):
                side=1+i%2;high=model if group=='value_on' else zero
                models=[high,model] if side==1 else [model,high]
                budgets=[512,64] if side==1 else [64,512]
                state=dict(status='running',group=group,game=i+1,finished=len(results),pid=os.getpid(),started=stamp())
                atomic_json(root/'status.json',state)
                rec=play(models,1300000+i//2,budgets,[.995,.995],opening=4,
                         progress=lambda ply:atomic_json(root/'status.json',dict(state,ply=ply)))
                rec.update(group=group,high_side=side)
                results.append(rec);atomic_pickle(root/f'{group}_{i+1}.pkl',rec)
                print(json.dumps(dict(group=group,game=i+1,winner=rec['winner'],side=side,reason=rec['reason'],plies=len(rec['steps'])),ensure_ascii=False),flush=True)
        counts={}
        for group in ('value_on','value_zero'):
            a=[x for x in results if x['group']==group]
            counts[group]=dict(wins=sum(x['winner']==x['high_side'] for x in a),losses=sum(x['winner']==3-x['high_side'] for x in a),unfinished=sum(x['winner'] is None for x in a))
        atomic_json(root/'result.json',dict(counts=counts,seconds=time.monotonic()-started,caveat='Four games per setting, same two opening seeds; exploratory only.'))
        atomic_json(root/'status.json',dict(status='complete',finished=len(results),pid=os.getpid()))
        print(json.dumps(counts),flush=True)
    except BaseException:
        atomic_json(root/'status.json',dict(status='failed',finished=len(results),error=traceback.format_exc(),pid=os.getpid()))
        raise

if __name__=='__main__':main()
