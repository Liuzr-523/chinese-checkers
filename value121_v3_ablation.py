"""Test whether the weak v3 critic actually hurts search; no training.
Predeclared zero-value vs learned-value comparison, identical frozen CNN/MCTS32.
"""
import argparse,fcntl,hashlib,json,os,shutil,time,traceback
from pathlib import Path
import torch
from az121 import Net121,Neural121
from curriculum121 import atomic,opening,replay,finish
from engine_c121 import CSearch121
from iterate121 import summary,paired_gate
import rules121 as r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=301044);args=ap.parse_args()
    torch.set_num_threads(1);source=Path('nn_checkpoints/models/checkpoints/neural121_v3_expert_imitation.pt');model=Net121.load(source)
    args.out.mkdir(parents=True,exist_ok=False);lock=(args.out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic()
    def update(stage,**kw):atomic(args.out/'status.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-start,**kw))
    def record(stage,row):
        with (args.out/(stage+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
    sources=['value121_v3_ablation.py','az121.py','rules121.py','curriculum121.py','iterate121.py','engine_c121.py','diagnostics/diagnose121_v3_value.py']
    config=dict(model=str(source.resolve()),model_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),profile=r.PROFILE,encoding=model.encoding,
                seed=args.seed,simulations=32,value_scale=0.,control_value_scale=1.,confirmation_games=60,independent_gate_p=.01,
                training='No training: same immutable v3 CNN, zero nonterminal value vs learned value. Search calibration only, not learned improvement.',
                automatic_promotion=False,source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources})
    atomic(args.out/'config.json',config)
    for name in sources:
        dest=args.out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes())
    shutil.copyfile(source,args.out/'candidate.pt')
    shutil.copyfile('nn_checkpoints/experiments/diagnose121_v3_value_20261030/result.json',args.out/'diagnosis.json')
    try:
        # Nonterminal zero-value rollout must have exactly zero backed-up value;
        # rule terminal returns remain intact and are tested in az121 tests.
        _,probe=Neural121(model,4,value_scale=0.).think(r.init_board(),1)
        if probe['root_q']!=0. or probe['root_visits']!=4:raise ValueError('ablation probe failed')
        matches=[]
        for pair in range(30):
            prefix=opening(args.seed+100000+pair,0 if pair==0 else 4+(pair%5)*2)
            for side in (1,2):
                update('independent_confirmation',completed=len(matches),total=60)
                row=finish(replay(prefix),{side:Neural121(model,32,value_scale=0.),3-side:Neural121(model,32,value_scale=1.)})
                row.update(pair=pair,side=side,opening=prefix);matches.append(row);record('confirmation',row)
        fixed={}
        for name,scale in [('control',1.),('candidate',0.)]:
            rows=[]
            for pair in range(10):
                prefix=opening(args.seed+200000+pair,0 if pair==0 else 4+(pair%4)*2)
                for side in (1,2):
                    update('fixed_c_'+name,completed=len(rows),total=20)
                    row=finish(replay(prefix),{side:Neural121(model,32,value_scale=scale),3-side:CSearch121(seconds=.03,max_depth=10)})
                    row.update(pair=pair,side=side,opening=prefix);rows.append(row);record('fixed_c_'+name,row)
            fixed[name]=summary(rows)
        gate=paired_gate(matches)
        result=dict(confirmation=summary(matches),paired_gate=gate,fixed_c=fixed,
                    qualified=gate['one_sided_p']<=.01 and fixed['candidate']['points']>=fixed['control']['points'],
                    promoted=False,weights_unchanged=True,seconds=time.monotonic()-start,
                    note='Diagnostic: disabling learned nonterminal critic only. Terminal wins/draws remain exact. No NN trained.')
        atomic(args.out/'result.json',result);update('complete',result=result);print(json.dumps(result),flush=True)
    except BaseException:
        update('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
