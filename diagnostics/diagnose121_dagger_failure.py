"""Audit failed expert correction; learning, search transfer, fixed-batch capacity.
Training/validation groups ONLY; no repeated use of heldout test or arena seeds.
Probe checkpoints are diagnostic only and never registered/deployed.
"""
import copy,hashlib,json,sys,time
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import rules121 as r
from az121 import Net121,Neural121,encode,action_id
from curriculum121 import replay,atomic
from reanalyze121_policy import batch,features,logits_from_features
from engine_clysto121 import ClystoDerived121

ROOT=Path('nn_checkpoints/experiments/dagger121_expert_correction_20261030')
OUT=Path('nn_checkpoints/experiments/diagnose121_dagger_failure_20261030')

@torch.no_grad()
def policy_metrics(model,part):
    ce=hits=0;prob=entropy=0.
    for offset in range(0,len(part),32):
        cells,ids,mask,pi=batch(part[offset:offset+32],'clean')
        lp=logits_from_features(model,cells).gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)
        ce+=float(-(pi*lp).sum());hits+=int((lp.argmax(1)==pi.argmax(1)).sum())
        prob+=float((pi*lp.exp()).sum());entropy+=float(-(lp.exp()*lp).sum())
    return dict(states=len(part),ce=ce/len(part),accuracy=hits/len(part),expert_probability=prob/len(part),entropy=entropy/len(part))

@torch.no_grad()
def full_metrics(model,part):
    ce=hits=0
    for offset in range(0,len(part),16):
        group=part[offset:offset+16];xs=torch.stack([s['x'] for s in group]);logits,_=model(xs)
        for row,s in zip(logits,group):
            logp=row[s['actions']].log_softmax(0);ce-=float(logp[s['target']]);hits+=int(int(logp.argmax())==s['target'])
    return dict(ce=ce/len(part),accuracy=hits/len(part))

def main():
    torch.set_num_threads(1);torch.manual_seed(301046);start=time.monotonic();OUT.mkdir(exist_ok=False)
    def status(stage,**extra):atomic(OUT/'status.json',dict(stage=stage,elapsed=time.monotonic()-start,**extra))
    config=json.loads((ROOT/'config.json').read_text());verified=0
    for name,sha in config['source_sha256'].items():
        assert hashlib.sha256((ROOT/'source'/name).read_bytes()).hexdigest()==sha
    for name,sha in config['data_sha256'].items():assert hashlib.sha256((ROOT/'data_sources'/name).read_bytes()).hexdigest()==sha
    base=Net121.load(config['model']);candidate=Net121.load(ROOT/'candidate.pt')
    assert hashlib.sha256(Path(config['model']).read_bytes()).hexdigest()==config['model_sha256']
    for k,v in base.state_dict().items():
        if not k.startswith(('src.','dst.','bias.')):assert torch.equal(v,candidate.state_dict()[k])
    cache=torch.load(ROOT/'cached_data.pt',weights_only=True);split=json.loads((ROOT/'split.json').read_text())
    annotations=[json.loads(s) for s in (ROOT/'annotations.jsonl').read_text().splitlines()]
    games=[json.loads(s) for s in (ROOT/'expert_rollout.jsonl').read_text().splitlines()]
    reconstructed={'train':[],'validation':[]};offsets=Counter()
    for game,record in zip(games,annotations):
        part=record['split'];assert game['pair'] in split[part]
        g=replay(game['opening']);labels={a['ply']:a for a in record['annotations']}
        for path in game['paths']:
            if g.ply in labels:
                a=labels[g.ply];assert a['player']==g.player and g.player!=game['side'] and a['played_path']==path
                assert r.valid_path(g.board,g.player,a['expert_path'])
                actions=[action_id(s,p[-1],g.player,base.canonical) for s,p in r.get_all_moves(g.board,g.player)]
                target=actions.index(action_id(a['expert_path'][0],a['expert_path'][-1],g.player,base.canonical))
                cached=cache['sets'][part][offsets[part]];offsets[part]+=1
                assert actions==cached['actions'] and cached['clean'][target]==1. and sum(cached['clean'])==1.
                if part in reconstructed:
                    x=torch.from_numpy(encode(g.board,g.player,g.ply,g.repetitions,canonical=base.canonical))
                    with torch.no_grad():expected=features(base,x.unsqueeze(0))[0]
                    torch.testing.assert_close(expected,cached['features'],atol=2e-6,rtol=1e-6)
                    reconstructed[part].append(dict(x=x,actions=actions,target=target,board=g.board.copy(),player=g.player,ply=g.ply,
                                                   repetitions=g.repetitions.copy(),expert_path=a['expert_path'],endgame=game['plies']-g.ply<=16))
                verified+=1
            g.play(path)
        assert g.result()==(game['winner'],game['reason']) and g.ply==game['plies']
    result=dict(audit=dict(rollout_games=len(games),annotations=verified,source_data_hashes_match=True,nonpolicy_parameters_identical=True,legal_labels_and_cached_action_indices_match=True),policy={},search_transfer={},capacity={})
    for name,model in [('baseline',base),('candidate',candidate)]:
        result['policy'][name]={}
        for domain,sets in [('correction',cache['sets']),('retention',cache['retention'])]:
            for part in ('train','validation'):result['policy'][name][domain+'_'+part]=policy_metrics(model,sets[part])
    status('search_transfer');before=[];after=[];teacher=[];rawbefore=[];rawafter=[]
    for i,s in enumerate(reconstructed['validation']):
        status('search_transfer',completed=i,total=len(reconstructed['validation']))
        target=(s['expert_path'][0],s['expert_path'][-1]);teacher.append(target)
        for model,choices,raw in [(base,before,rawbefore),(candidate,after,rawafter)]:
            path,info=Neural121(model,32).think(s['board'],s['player'],s['ply'],s['repetitions']);choices.append((path[0],path[-1]))
            j=int(np.argmax(info['priors']));p=info['paths'][j];raw.append((p[0],p[-1]))
    n=len(teacher)
    result['search_transfer']=dict(states=n,raw_baseline_agreement=sum(a==t for a,t in zip(rawbefore,teacher))/n,
       raw_candidate_agreement=sum(a==t for a,t in zip(rawafter,teacher))/n,
       mcts_baseline_agreement=sum(a==t for a,t in zip(before,teacher))/n,mcts_candidate_agreement=sum(a==t for a,t in zip(after,teacher))/n,
       changed_raw=sum(a!=b for a,b in zip(rawbefore,rawafter)),changed_mcts=sum(a!=b for a,b in zip(before,after)),
       corrected_mcts=sum(a!=t and b==t for a,b,t in zip(before,after,teacher)),lost_mcts=sum(a==t and b!=t for a,b,t in zip(before,after,teacher)),
       note='Development validation proxy, not an independent playing-strength result.')
    # Repeated timed teacher labels can vary with completed search depth.
    status('expert_label_stability');expert=ClystoDerived121(seconds=.04,max_depth=10);stability=[]
    picked=np.random.default_rng(301046).choice(n,size=min(20,n),replace=False)
    for i in picked:
        s=reconstructed['validation'][int(i)];choices=[]
        for _ in range(3):
            p,info=expert.think(s['board'],s['player'],s['ply'],s['repetitions']);choices.append((p[0],p[-1]))
        stability.append(dict(index=int(i),original=teacher[int(i)],repeated=choices,all_repeats_same=len(set(choices))==1))
    result['label_stability']=dict(states=len(stability),all_repeats_same=sum(x['all_repeats_same'] for x in stability),
       repeats_matching_original=sum(t==tuple(row['original']) for row in stability for t in row['repeated']),queries=3*len(stability),details=stability)
    # Fixed 64 training states, 16 per true side/endgame stratum; no test use.
    rng=np.random.default_rng(301046);strata={}
    for s in reconstructed['train']:strata.setdefault((s['player'],s['endgame']),[]).append(s)
    tiny=[strata[k][int(i)] for k in sorted(strata) for i in rng.choice(len(strata[k]),size=16,replace=False)]
    for arm in ('heads_only','full_policy_cnn'):
        model=copy.deepcopy(base);model.requires_grad_(False)
        for name,p in model.named_parameters():
            if arm=='heads_only':p.requires_grad_(name.startswith(('src.','dst.','bias.')))
            else:p.requires_grad_(not name.startswith('value.'))
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=.0001)
        local=np.random.default_rng(301046);history=[dict(step=0,**full_metrics(model,tiny))]
        for step in range(800):
            group=[tiny[int(i)] for i in local.integers(len(tiny),size=16)];xs=torch.stack([s['x'] for s in group]);logits,_=model(xs)
            loss=torch.stack([-row[s['actions']].log_softmax(0)[s['target']] for row,s in zip(logits,group)]).mean()
            opt.zero_grad();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1));opt.step()
            if (step+1)%100==0:
                history.append(dict(step=step+1,gradient_norm=norm,**full_metrics(model,tiny)));status('capacity_probe',arm=arm,step=step+1)
        result['capacity'][arm]=dict(states=len(tiny),history=history,validation=full_metrics(model,reconstructed['validation']),
                                   note='Tiny training-batch fitting diagnostic only; full CNN update also changes value features; do NOT deploy this probe.')
    result['seconds']=time.monotonic()-start
    atomic(OUT/'result.json',result);status('complete');print(json.dumps({k:v for k,v in result.items() if k!='label_stability'}))
if __name__=='__main__':main()
