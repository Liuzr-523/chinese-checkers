"""既有七棋例回归，仅诊断，不参与本轮新根选择、训练或模型验收。"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import rules121 as r
from az121 import Net121,encode,action_id
from curriculum121 import replay,atomic
from checkpoint121_save import digest
SEEDS=Path('docs/experiments/skill121_review_20261105/inputs/targeted/seeds.json')
GAMES=Path('docs/experiments/skill121_review_20261105/inputs/recheck/games.jsonl')
POS={9,13,47};NEG={12,31,40,44}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(1);m=Net121.load(a.model);m.requires_grad_(False);seeds=json.loads(SEEDS.read_text());by={(x['seed_id'],x['role']):x for x in map(json.loads,GAMES.read_text().splitlines()) if x['repeat']==0};cases=[]
    with torch.no_grad():
        for i in sorted(POS|NEG):
            g=replay(seeds[i]['opening']);rear=by[i,'expert']['opening'][-1];front=by[i,'played']['opening'][-1];paths=[p for _,p in r.get_all_moves(g.board,g.player)];ids=[action_id(p[0],p[-1],g.player,m.canonical) for p in paths];l,_=m(torch.from_numpy(encode(g.board,g.player,g.ply,g.repetitions,m.canonical)).unsqueeze(0));logits=l[0,ids];ia=ids.index(action_id(rear[0],rear[-1],g.player,m.canonical));ib=ids.index(action_id(front[0],front[-1],g.player,m.canonical));gap=float(logits[ia]-logits[ib]);expected=1 if i in POS else -1;prob=logits.softmax(0)
            cases.append(dict(seed_id=i,player=g.player,ply=g.ply,rear_path=rear,front_path=front,expected=expected,logit_gap=gap,correct=expected*gap>0,rear_probability=float(prob[ia]),front_probability=float(prob[ib]),rear_rank=1+int((logits>logits[ia]).sum()),front_rank=1+int((logits>logits[ib]).sum()),raw_policy_path=paths[int(logits.argmax())]))
    atomic(a.out,dict(scope='历史回归，不是新开发、不作模型选择/升级门槛',n=7,correct=sum(x['correct'] for x in cases),rear_positive_correct=sum(x['correct'] for x in cases if x['expected']==1),front_positive_correct=sum(x['correct'] for x in cases if x['expected']==-1),model_sha256=digest(a.model),source_sha256={str(p):digest(p) for p in (SEEDS,GAMES,Path(__file__))},cases=cases))
if __name__=='__main__':main()
