"""Inspect corrected training targets and actual errors after the ratio experiment."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from az121 import Net121
from opening121 import OpeningBook
from train_opening121 import book_samples
from opening121_ratios import stage_of
from curriculum121 import batch,atomic
from multicandidate121_trial import witness
import rules121 as r

def main(folder):
 torch.set_num_threads(1);p=Path(folder);cfg=json.loads((p/'config.json').read_text());book=OpeningBook();sets,_=book_samples(book)
 @torch.no_grad()
 def predictions(model):
  out=[]
  for s in sets['train']:
   x,*_=batch([s]);l,_=model(x);a=s['actions'][int(l[0,s['actions']].argmax())];good=max(s['positive'],key=lambda a:float(l[0,a]));mass=float((torch.logsumexp(l[0,s['positive']],0)-torch.logsumexp(l[0,s['actions']],0)).exp());out.append(dict(decision_id=s['decision_id'],own_move=stage_of(s),correct=a in s['positive'],chosen=a,preferred=good,probability=mass,logit_gap=float(l[0,a]-l[0,good])))
  return out
 baseline=predictions(Net121.load(cfg['model']));results={};hard=[]
 for arm in ('book25','book50','book75'):
  rows=predictions(Net121.load(p/(arm+'_step2400.pt')));result={}
  for stage,lo,hi in [('moves1_3',1,3),('moves4_6',4,6),('moves7_10',7,10)]:
   paired=[(a,b) for a,b in zip(baseline,rows) if lo<=a['own_move']<=hi];result[stage]=dict(n=len(paired),newly_correct=sum(not a['correct'] and b['correct'] for a,b in paired),newly_wrong=sum(a['correct'] and not b['correct'] for a,b in paired),still_wrong=sum(not a['correct'] and not b['correct'] for a,b in paired))
  results[arm]=result
  if arm=='book50':
   for row in sorted((x for x in rows if not x['correct'] and x['own_move']>=7),key=lambda x:-x['logit_gap'])[:10]:
    raw,player=row['decision_id'].split(':');player=int(player);board=list(map(int,raw));g=r.Game(board=board,player=player,ply=2*(row['own_move']-1)+(player-1));hard.append({**row,'chosen_path':witness(g,row['chosen']),'book_path':witness(g,row['preferred'])})
 atomic(p/'case_transitions.json',results);atomic(p/'hard_cases.json',hard);print(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__':main(sys.argv[1])
