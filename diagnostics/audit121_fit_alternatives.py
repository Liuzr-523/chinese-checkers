"""Score the fitted models' actual top1 on the same32 TRAINING states only.
Exact same expert root horizon5, forced child4; ties are accepted as equal scores.
No training/checkpoint selection/arena use.
"""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import rules121 as r
from az121 import Net121
from curriculum121 import replay,batch,atomic
from corrected121_decisions import score_action,ConsistentExpert
RUN=Path('nn_checkpoints/experiments/fit121_objectives32_20261101')
def main():
 torch.set_num_threads(1);ConsistentExpert.failure_dir=RUN
 metadata={s['decision_id']:s for s in [json.loads(line) for line in (RUN/'data_sources/decisions.jsonl').read_text().splitlines()]}
 validation='--validation' in sys.argv
 cached=torch.load(RUN/'data_sources/decisions.pt',weights_only=True)['sets']['validation' if validation else 'train'];samples={s['decision_id']:s for s in cached};selected=[s['decision_id'] for s in cached] if validation else json.loads((RUN/'fixed32.json').read_text())['decision_ids'];scores={};result={}
 with torch.no_grad():
  for arm in ('root_policy','policy_replay','full_objective'):
   model=Net121.load(RUN/(arm+'_step1600.pt'));rows=[]
   for did in selected:
    sample=samples[did];row=metadata[did];g=replay(row['opening']);xs,_,_,_,_=batch([sample]);logits,_=model(xs);action=sample['actions'][int(logits[0,sample['actions']].argmax())]
    s,e=divmod(action,121)
    if g.player==2:s,e=120-s,120-e
    key=(did,s,e)
    if (s,e)==(row['preferred'][0],row['preferred'][-1]):chosen=row['expert_score']
    elif (s,e)==(row['rejected'][0],row['rejected'][-1]):chosen=row['rejected_score']
    else:
     if key not in scores:
      path=next(p for start,p in r.get_all_moves(g.board,g.player) if start==s and p[-1]==e);scores[key]=score_action(g,path)
     chosen=scores[key]
    if chosen>row['expert_score']:raise ValueError('forced-action score exceeded recorded fixed root best')
    rows.append(dict(decision_id=did,chosen=(s,e),best_score=row['expert_score'],chosen_score=chosen,regret=row['expert_score']-chosen,identical_expert_move=(s,e)==(row['preferred'][0],row['preferred'][-1])))
   result[arm]=dict(n=len(selected),exact_expert_matches=sum(x['identical_expert_move'] for x in rows),equal_best_score=sum(x['regret']==0 for x in rows),regret_at_least4=sum(x['regret']>=4 for x in rows),mean_regret=sum(x['regret'] for x in rows)/len(selected),details=rows)
 output=dict(scope=('Original40 development validation positions' if validation else 'Original32 training positions')+'. Search-score ties, not proof of game-winning optimality. No weights modified.',arms=result,extra_child_searches=len(scores))
 atomic(RUN/('alternative_action_validation.json' if validation else 'alternative_action_audit.json'),output);(RUN/('alternative_validation_source.py' if validation else 'alternative_audit_source.py')).write_bytes(Path(__file__).read_bytes());print(json.dumps({a:{k:v for k,v in row.items() if k!='details'} for a,row in result.items()}))
if __name__=='__main__':main()
