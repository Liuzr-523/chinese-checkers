"""Initial8-batch gradient/duplicate audit only; no optimizer update."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from az121 import Net121
from curriculum121 import samples_from_game,atomic
from canonical121_trial import canonical_sample
from cycle121_targeted import expert_samples
from generalization121_curve import input_digest,remove_validation_duplicates
from diagnostics.fit121_objectives import objective_terms
RUN=Path('nn_checkpoints/experiments/fit121_objectives32_20261101')
def main():
 torch.set_num_threads(1);cfg=json.loads((RUN/'config.json').read_text());data=RUN/'data_sources';root_data=torch.load(data/'decisions.pt',weights_only=True)['sets'];ids=set(json.loads((RUN/'fixed32.json').read_text())['decision_ids']);roots=[s for s in root_data['train'] if s['decision_id'] in ids]
 # Preserve the actual stored32 order used for sampling.
 by_id={s['decision_id']:s for s in roots};roots=[by_id[i] for i in json.loads((RUN/'fixed32.json').read_text())['decision_ids']]
 split=json.loads((data/'old_split.json').read_text());old=[];normal=[]
 for line in (data/'old_teacher.jsonl').read_text().splitlines():
  g=json.loads(line)
  if g['group'] in split['train']:old.extend({**canonical_sample(s),'value_weight':1.} for s in samples_from_game(g))
 for line in (data/'normal_games.jsonl').read_text().splitlines():
  g=json.loads(line)
  if g['split']=='train':normal.extend(expert_samples(g))
 old,_=remove_validation_duplicates(old,root_data['validation']);normal,_=remove_validation_duplicates(normal,root_data['validation'])
 index={}
 for s in old+normal:index.setdefault(input_digest(s),[]).append(s)
 duplicated=conflicting=0
 for root in roots:
  matches=index.get(input_digest(root),[]);duplicated+=bool(matches)
  preferred=root['actions'][root['target']]
  conflicting+=any(s['actions'][s['target']]!=preferred for s in matches)
 model=Net121.load(cfg['model']);reference=Net121.load(cfg['model']);params=[p for n,p in model.named_parameters() if not n.startswith('value.')]
 rr=np.random.default_rng(cfg['seed']+1);ro=np.random.default_rng(cfg['seed']+2);rows=[]
 for i in range(8):
  selected=[roots[int(j)] for j in rr.integers(len(roots),size=4)];ordinary=[old[int(j)] for j in ro.integers(len(old),size=6)]+[normal[int(j)] for j in ro.integers(len(normal),size=6)]
  _,terms=objective_terms(model,reference,selected,ordinary,'policy_replay')
  a=torch.autograd.grad(terms['root_ce']+terms['rank'],params,retain_graph=True);b=torch.autograd.grad(terms['replay_ce'],params)
  a=torch.cat([g.flatten() for g in a]);b=torch.cat([g.flatten() for g in b]);cos=float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-12))
  rows.append(dict(batch=i,cosine=cos,root_gradient_norm=float(a.norm()),ordinary_gradient_norm=float(b.norm())))
 result=dict(exact_root_input_duplicates_in_ordinary=duplicated,conflicting_exact_input_labels=conflicting,initial_batches=8,negative_cosine_batches=sum(r['cosine']<0 for r in rows),mean_cosine=float(np.mean([r['cosine'] for r in rows])),details=rows,
             scope='Initial-state weighted policy gradients only; no optimizer step. Does not attribute whole-run behavior to one gradient statistic.')
 atomic(RUN/'initial_gradient_audit.json',result);(RUN/'gradient_audit_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result))
if __name__=='__main__':main()
