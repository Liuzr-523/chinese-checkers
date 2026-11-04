"""Small, clearly labelled policy-distillation trial from recorded online JumpStar.
Unfinished games provide only search policy targets, never fictitious value labels.
The frozen policy head is lightly adapted; incumbent remains untouched.
"""
import copy,hashlib,json,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
import Chinese_Checker as cc
from alphazero_nn import encode_board,all_moves_indexed
from nn_resnet import NetResNet,_make_batch
from az_longrun import play,atomic_json,atomic_pickle,stamp
from engine_jumpstar import STAR_TO_LOCAL,checked_path,RuleMismatch


def samples(game):
    result=[];discarded=0
    for step in game['steps']:
        if step['by']!='jumpstar':continue
        raw=step['details'].get('root_moves') or []
        if not raw:discarded+=1;continue
        player=step['player'];board=step['board'];moves=all_moves_indexed(board,player)
        lookup={(s,d):i for i,(s,d,_) in enumerate(moves)};weights=np.zeros(len(moves),dtype=np.float64)
        total=sum(max(0,float(m.get('policy_target',0))) for m in raw)
        if total<=0:discarded+=1;continue
        valid=0.0
        for m in raw:
            w=max(0,float(m.get('policy_target',0)))
            if not w:continue
            try:
                path=checked_path(board,player,m.get('path') or [m['from'],m['to']],{'legal':raw})
                i=lookup[(path[0],path[-1])]
            except (RuleMismatch,KeyError):continue
            weights[i]+=w;valid+=w
        # Preserve near-complete teacher mass; don't silently drop important side-arm actions.
        if valid/total < .95:discarded+=1;continue
        result.append(dict(s=encode_board(board,player),player=player,moves=moves,
                           pi=(weights/valid).astype(np.float32),z=0.,vw=0.,
                           teacher_mass_kept=valid/total,ply=step['ply']))
    return result,discarded


@torch.no_grad()
def metrics(model,data):
    model.eval();loss=[];same=[]
    for i in range(0,len(data),32):
        xs,src,dst,pi,mask,z,vw=_make_batch(data[i:i+32],'cpu')
        logits,_=model(xs);scores=(logits.gather(1,src)+logits.gather(1,dst)).masked_fill(mask==0,-1e30)
        loss.extend((-(pi*torch.log_softmax(scores,1)).sum(1)).tolist())
        same.extend((scores.argmax(1)==pi.argmax(1)).tolist())
    return dict(samples=len(data),policy_ce=float(np.mean(loss)),top_move_agreement=float(np.mean(same))) if data else None


def main():
    torch.set_num_threads(1);torch.manual_seed(20261028)
    matches=Path('nn_checkpoints/experiments/jumpstar_probe_20261028_b')
    out=Path('nn_checkpoints/experiments/jumpstar_distill_20261028');out.mkdir(parents=True,exist_ok=False)
    result=dict(status='running',started=stamp(),type='external JumpStar policy distillation, not pure self-play',promoted=False)
    atomic_json(out/'status.json',result)
    try:
        games=[json.loads((matches/f'game_{i}.json').read_text()) for i in (1,2)]
        tr,skip1=samples(games[0]);va,skip2=samples(games[1])
        result.update(train_samples=len(tr),heldout_samples=len(va),skipped_train=skip1,skipped_heldout=skip2,
                      note='Held out entire second game; incomplete games provide policy targets only; value not trained.')
        if len(tr)<4 or len(va)<2:
            result.update(status='insufficient_data');atomic_json(out/'status.json',result);print(json.dumps(result),flush=True);return
        source=Path('nn_checkpoints/experiments/search_budget_20261022/model.pt')
        baseline=NetResNet.load(source).eval();candidate=copy.deepcopy(baseline)
        result['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
        result['before']=metrics(baseline,va)
        for param in candidate.parameters():param.requires_grad=False
        for param in candidate.p_fc2.parameters():param.requires_grad=True
        # Keep BN statistics fixed; train only last policy linear layer.
        candidate.eval();optimizer=torch.optim.Adam(candidate.p_fc2.parameters(),lr=1e-5)
        rng=np.random.default_rng(20261028)
        for iteration in range(100):
            batch=[tr[i] for i in rng.integers(0,len(tr),size=min(16,len(tr)))]
            xs,src,dst,pi,mask,z,vw=_make_batch(batch,'cpu')
            logits,_=candidate(xs);scores=(logits.gather(1,src)+logits.gather(1,dst)).masked_fill(mask==0,-1e30)
            with torch.no_grad():
                orig,_=baseline(xs);os=(orig.gather(1,src)+orig.gather(1,dst)).masked_fill(mask==0,-1e30)
                old=torch.softmax(os,1)
            lp=F.log_softmax(scores,1)
            ce=-(pi*lp).sum(1).mean()
            # Cross-entropy to prior is KL up to a constant; retain existing policy.
            retain=-(old*lp).sum(1).mean()
            loss=ce+.3*retain
            optimizer.zero_grad();loss.backward();optimizer.step()
        candidate.save(out/'candidate.pt');result['after']=metrics(candidate,va);result['status']='evaluating'
        result['training']=dict(steps=100,learning_rate=1e-5,trainable='policy final linear layer',baseline_retention_weight=.3)
        atomic_json(out/'status.json',result);print('trained',json.dumps(result,ensure_ascii=False),flush=True)
        # New openings, same MCTS budget and both colors; no auto-promotion.
        matches2=[]
        for i in range(4):
            side=1+i%2;models=[candidate,baseline] if side==1 else [baseline,candidate]
            rec=play(models,1400000+i//2,[64,64],[.995,.995],opening=4,max_moves=240)
            rec['candidate_side']=side;matches2.append(rec);atomic_pickle(out/f'eval_{i+1}.pkl',rec)
            print(f"eval {i+1}/4 winner={rec['winner']} reason={rec['reason']} plies={len(rec['steps'])}",flush=True)
        result['match']=dict(wins=sum(r['winner']==r['candidate_side'] for r in matches2),
                             losses=sum(r['winner']==3-r['candidate_side'] for r in matches2),
                             unfinished=sum(r['winner'] is None for r in matches2))
        result.update(status='complete',completed=stamp())
        atomic_json(out/'result.json',result);atomic_json(out/'status.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException:
        result.update(status='failed',error=traceback.format_exc());atomic_json(out/'status.json',result);raise

if __name__=='__main__':main()
