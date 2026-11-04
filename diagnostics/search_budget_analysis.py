"""Audit completed fixed-budget games and value head against recorded outcomes."""
import collections,json,math,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import Chinese_Checker as cc
from alphazero_nn import encode_board
from nn_resnet import NetResNet

root=Path('nn_checkpoints/experiments/search_budget_20261022')
torch.set_num_threads(1)
model=NetResNet.load(root/'model.pt').eval()
records=[pickle.loads(p.read_bytes()) for p in sorted((root/'games').glob('*.pkl'))]
by_group={}; samples=[]
for rec in records:
 key='%s vs %s'%tuple(rec['comparison']);g=by_group.setdefault(key,{'games':0,'high_p1_win':0,'high_p1_loss':0,'high_p2_win':0,'high_p2_loss':0,'unfinished':0,'pairs':{}})
 g['games']+=1
 side=rec['high_side'];w=rec['winner'];g['pairs'].setdefault(rec['seed'],{})[str(side)]=w
 if w is None:g['unfinished']+=1
 else:g['high_p%d_%s'%(side,'win' if w==side else 'loss')]+=1
 b=cc.init_board()
 for ply,step in enumerate(rec['steps']):
  p=step['player']
  if w is not None and ply%10 in (0, 1):
   samples.append((encode_board(b,p),p,1 if w==p else -1,ply,key))
  b=cc.apply_move(b,step['move_path'])
 assert cc.is_game_over(b)==w
for key,g in by_group.items():
 paired=collections.Counter()
 for p in g.pop('pairs').values():
  if set(p)=={'1','2'}:
   outcomes=[None if p[str(side)] is None else (1 if p[str(side)]==side else 0) for side in (1,2)]
   if None in outcomes:paired['unfinished']+=1
   else:paired[str(sum(outcomes))+'_high_wins']+=1
 g['paired_openings']=dict(paired)
 # Paired sign test considers only opening pairs where one engine wins both games.
 n=paired['0_high_wins']+paired['2_high_wins'];lo=min(paired['0_high_wins'],paired['2_high_wins'])
 g['paired_sign_test_p_two_sided']=min(1,2*sum(math.comb(n,k) for k in range(lo+1))/2**n) if n else None
vals=[]
with torch.no_grad():
 for i in range(0,len(samples),64):
  _,v=model(torch.from_numpy(np.stack([s[0] for s in samples[i:i+64]])))
  vals.extend(v.numpy().tolist())
cal={}
for player in (1,2):
 subset=[(v,s[2]) for v,s in zip(vals,samples) if s[1]==player]
 pred=np.array([p for p,_ in subset]);actual=np.array([a for _,a in subset])
 cal[str(player)]={'samples':len(subset),'mean_predicted_value':float(pred.mean()),'mean_actual_outcome':float(actual.mean()),'mean_error':float((pred-actual).mean()),'mse':float(np.mean((pred-actual)**2)),'sign_accuracy':float(np.mean((pred>=0)==(actual>0)))}
result={'groups':by_group,'value_calibration':cal,'caveat':'Outcomes follow mixed-budget policies; the same board may recur across games. Calibration is descriptive, not an independent causal estimate.'}
(root/'diagnostic_outcome_calibration.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print(json.dumps(result,indent=2,ensure_ascii=False))
