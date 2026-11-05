"""只读复盘既有固定续弈分支；不训练、不搜索、不生成新对局。"""
import argparse,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import rules121 as r
from az121 import Net121,encode,action_id
from curriculum121 import replay,atomic
from checkpoint121_save import REGISTRY,digest

ROOT=Path('nn_checkpoints/experiments/targeted121_complete_cycle_20261030')
RECHECK=Path('nn_checkpoints/experiments/recheck121_counterfactual_20261031')

def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sign(x):return 1 if x>1e-8 else -1 if x < -1e-8 else 0

def tip_distance(board,player):
    q0,y0=r.COORDS[120 if player==1 else 0]
    return max(max(abs(q-q0),abs(y-y0),abs(q+y-q0-y0)) for i,(q,y) in enumerate(r.COORDS) if board[i]==player)

def features(board,player):
    return dict(goal=sum(board[i]==player for i in r.GOAL[player]),home=sum(board[i]==player for i in r.HOME[player]),farthest=tip_distance(board,player))

def branch_score(winner,player):return 0. if winner==0 else 1. if winner==player else -1.

def validate_game(row):
    g=replay(row['opening']);assert g.ply==row['start_ply']
    for path in row['paths']:g.play(path)
    assert g.ply==row['plies'] and g.result()==(row['winner'],row['reason'])
    assert row['reason']!='max_ply'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--recheck',type=Path,default=RECHECK);args=ap.parse_args();root=args.root;recheck=args.recheck;args.out.mkdir(exist_ok=False,parents=True);torch.set_num_threads(1)
    entry=json.loads(REGISTRY.read_text());modelpath=Path(entry['checkpoint']);assert digest(modelpath)==entry['sha256'];model=Net121.load(modelpath);model.requires_grad_(False)
    seeds=json.loads((root/'seeds.json').read_text());cfg=json.loads((recheck/'config.json').read_text());fixed=rows(recheck/'games.jsonl');saved=json.loads((recheck/'result.json').read_text());branches=defaultdict(dict)
    for row in fixed:
        validate_game(row);branches[row['seed_id']][(row['repeat'],row['role'])]=row
    details=[]
    with torch.no_grad():
        for seed_id in cfg['seed_ids']:
            s=seeds[seed_id];g=replay(s['opening']);assert g.ply==s['ply'] and g.player==s['player'];p=g.player;pair=branches[seed_id]
            for role in ('expert','played'):
                a,b=pair[0,role],pair[1,role];assert a['paths']==b['paths'] and a['winner']==b['winner'];assert a['opening'][:-1]==s['opening']
            choices={role:pair[0,role]['opening'][-1] for role in ('expert','played')}
            assert choices['played']==s['played_path'];assert all(r.valid_path(g.board,p,path) for path in choices.values())
            moves=r.get_all_moves(g.board,p);ids=[action_id(start,path[-1],p,model.canonical) for start,path in moves]
            logits,rootv=model(torch.from_numpy(encode(g.board,p,g.ply,g.repetitions,model.canonical)).unsqueeze(0));legal=logits[0,ids];prob=legal.softmax(0);choice=sorted(range(len(ids)),key=lambda i:-float(legal[i]))[0];raw=moves[choice][1];arms={}
            for role,path in choices.items():
                child=replay(s['opening']+[path]);l,v=model(torch.from_numpy(encode(child.board,child.player,child.ply,child.repetitions,model.canonical)).unsqueeze(0));idx=ids.index(action_id(path[0],path[-1],p,model.canonical));a=pair[0,role]
                arms[role]=dict(path=path,probability=float(prob[idx]),policy_rank=1+sum(float(x)>float(legal[idx]) for x in legal),root_view_child_value=-float(v[0]),child_terminal=child.result(),features=features(child.board,p),winner=a['winner'],reason=a['reason'],end_ply=a['plies'],root_view_outcome=branch_score(a['winner'],p))
            outcome=arms['expert']['root_view_outcome']-arms['played']['root_view_outcome'];vdiff=arms['expert']['root_view_child_value']-arms['played']['root_view_child_value'];pdiff=arms['expert']['probability']-arms['played']['probability']
            details.append(dict(seed_id=seed_id,source_game=s['game_id'],source_split=s['split'],player=p,ply=g.ply,category=s['category'],board=g.board,root_features=features(g.board,p),root_value=float(rootv[0]),raw_policy_path=raw,branches=arms,outcome_difference=outcome,value_difference=vdiff,policy_difference=pdiff,distinct=choices['expert'][0]!=choices['played'][0] or choices['expert'][-1]!=choices['played'][-1],value_sign_correct=sign(vdiff)==sign(outcome),policy_sign_correct=sign(pdiff)==sign(outcome)))
    non_tied=[x for x in details if x['outcome_difference']];positive=[x for x in non_tied if x['outcome_difference']>0];negative=[x for x in non_tied if x['outcome_difference']<0]
    summary=dict(roots=len(details),original_source_games=len({x['source_game'] for x in details}),replayed_games=len(fixed),repeats=2,expert_better=len(positive),expert_worse=len(negative),outcome_tied=len(details)-len(non_tied),non_tied_value_correct=sum(x['value_sign_correct'] for x in non_tied),non_tied_policy_correct=sum(x['policy_sign_correct'] for x in non_tied),positive_value_correct=sum(x['value_sign_correct'] for x in positive),positive_policy_correct=sum(x['policy_sign_correct'] for x in positive),negative_value_correct=sum(x['value_sign_correct'] for x in negative),negative_policy_correct=sum(x['policy_sign_correct'] for x in negative),immediate_terminal_children=sum(x['branches'][role]['child_terminal'][0] is not None for x in details for role in ('expert','played')))
    assert summary['expert_better']==saved['fixed_positive'] and summary['expert_worse']==saved['fixed_negative']
    for label,subset in [('positive',positive),('negative',negative)]:summary[label+'_seed_ids']=[x['seed_id'] for x in subset]
    # These are factual archived NN turns, not counterfactual regenerated play.
    games=rows(root/'data_sources/dev_games.jsonl');annotations=rows(root/'data_sources/dev_annotations.jsonl');stage=defaultdict(Counter)
    for i,(game,ann) in enumerate(zip(games,annotations)):
        validate_game(game);assert ann['game_id']==i;g=replay(game['opening']);by_ply={a['ply']:a for a in ann['annotations']}
        for path in game['paths']:
            if g.ply in by_ply:
                a=by_ply[g.ply];assert a['played_path']==path and a['player']==g.player and g.player!=game['side'];phase='opening_0_19' if g.ply<20 else 'middle_20_59' if g.ply<60 else 'late_60_plus';c=stage[phase];c['nn_turns']+=1;c['games_with_turn_marker_'+str(i)]=1
                distinct=(path[0],path[-1])!=(a['expert_path'][0],a['expert_path'][-1]);c['teacher_disagreement']+=distinct
                if distinct:c['teacher_reduces_farthest']+=tip_distance(r.apply_move(g.board,a['expert_path']),g.player)<tip_distance(r.apply_move(g.board,path),g.player)
            g.play(path)
    stage_result={k:{**{a:b for a,b in v.items() if not a.startswith('games_with_turn_marker_')},'source_games':sum(a.startswith('games_with_turn_marker_') for a in v)} for k,v in stage.items()}
    inputs=[root/'seeds.json',root/'data_sources/dev_games.jsonl',root/'data_sources/dev_annotations.jsonl',recheck/'config.json',recheck/'result.json',recheck/'games.jsonl',REGISTRY,modelpath]
    atomic(args.out/'result.json',dict(roadmap_step=3,scope='事后机制复盘；既有选中过的开发/训练案例，不是新验收。固定续弈结果不是最优博弈标签。不训练、不搜索、不对局。',model_sha256=digest(modelpath),value_head=model.value_head,summary=summary,archived_nn_stages=stage_result,details=details))
    atomic(args.out/'provenance.json',dict(inputs={str(p):digest(p) for p in inputs},source_sha256=digest(Path(__file__)),model_unchanged=digest(modelpath)==entry['sha256']))
    print(json.dumps(summary,ensure_ascii=False));print(json.dumps(stage_result,ensure_ascii=False))
if __name__=='__main__':main()
