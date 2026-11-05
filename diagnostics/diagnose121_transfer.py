"""固定模型事后只读策略迁移诊断；不查询教师、不训练、不选择模型。"""
import argparse,json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import rules121 as r
from az121 import Net121,encode,action_id
from opening121 import OpeningBook
from train_opening121 import book_samples
from curriculum121 import batch,replay

def row_metrics(scores,actions,positive):
    """名次按严格大于目标最高分计，保留合法动作顺序处理首选同分。"""
    v=np.asarray(scores,dtype=float);actions=list(actions);positive=set(positive)
    if not positive or not positive.issubset(actions):raise ValueError('目标须非空且合法')
    idx=[i for i,a in enumerate(actions) if a in positive];best=float(v[idx].max());other=[v[i] for i,a in enumerate(actions) if a not in positive]
    prob=np.exp(v-v.max());prob/=prob.sum();chosen=actions[int(v.argmax())];ordered=np.sort(v)
    return dict(chosen=chosen,hit=chosen in positive,target_probability=float(prob[idx].sum()),target_rank=1+int((v>best).sum()),target_margin=best-float(max(other)) if other else None,argmax_margin=float(ordered[-1]-ordered[-2]) if len(v)>1 else None,legal_count=len(actions),target_count=len(positive))

def geometry(identity):
    raw,p=identity.split(':');p=int(p);board=list(map(int,raw));sign=1 if p==1 else -1;ys=[sign*r.COORDS[i][1] for i,v in enumerate(board) if v==p]
    return dict(exited_home=sum(v==p and i not in r.HOME[p] for i,v in enumerate(board)),mean_forward_y=float(np.mean(ys)),forward_y_hist=[ys.count(y) for y in range(-8,9)])

def reduce_rows(rows):
    if not rows:return dict(n=0)
    return dict(n=len(rows),hits=sum(x['hit'] for x in rows),mean_target_probability=float(np.mean([x['target_probability'] for x in rows])),mean_target_rank=float(np.mean([x['target_rank'] for x in rows])),median_target_rank=float(np.median([x['target_rank'] for x in rows])),mean_target_margin=float(np.mean([x['target_margin'] for x in rows if x['target_margin'] is not None])),mean_argmax_margin=float(np.mean([x['argmax_margin'] for x in rows if x['argmax_margin'] is not None])))

def stratify(rows):
    return dict(overall=reduce_rows(rows),players={str(p):reduce_rows([x for x in rows if x['player']==p]) for p in (1,2)},stages={str(t):reduce_rows([x for x in rows if x['own_move']==t]) for t in range(1,11)})

def compare(new,old):
    a={x['decision_id']:x for x in new};b={x['decision_id']:x for x in old};assert a.keys()==b.keys();out=[]
    for k,x in a.items():
        y=b[k];out.append(dict(decision_id=k,player=x['player'],own_move=x['own_move'],choice_changed=x['chosen']!=y['chosen'],probability_delta=x['target_probability']-y['target_probability'],rank_delta=x['target_rank']-y['target_rank'],margin_delta=(x['target_margin']-y['target_margin']) if x['target_margin'] is not None and y['target_margin'] is not None else None,old_hit=y['hit'],new_hit=x['hit']))
    def count(ss):
        return dict(n=len(ss),choice_changed=sum(x['choice_changed'] for x in ss),probability_up=sum(x['probability_delta']>1e-8 for x in ss),probability_up_same_choice=sum(x['probability_delta']>1e-8 and not x['choice_changed'] for x in ss),probability_up_still_miss=sum(x['probability_delta']>1e-8 and not x['new_hit'] for x in ss),gained_hits=sum(x['new_hit'] and not x['old_hit'] for x in ss),lost_hits=sum(x['old_hit'] and not x['new_hit'] for x in ss))
    return dict(overall=count(out),players={str(p):count([x for x in out if x['player']==p]) for p in (1,2)},stages={str(t):count([x for x in out if x['own_move']==t]) for t in range(1,11)},cases=out)

@torch.no_grad()
def evaluate(model,samples,meta):
    rows=[]
    for off in range(0,len(samples),32):
        ss=samples[off:off+32];x,*_=batch(ss);logits,_=model(x)
        for s,row in zip(ss,logits):rows.append(dict(**meta[s['decision_id']],**row_metrics(row[s['actions']].numpy(),s['actions'],s['positive'])))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path('nn_checkpoints/experiments/fresh121_projection_20261105'));ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(1);a.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((a.source/'config.json').read_text());labels=json.loads((a.source/'development.json').read_text());choices=json.loads((a.source/'development_choices.json').read_text());sets,_=book_samples(OpeningBook());sets['fresh_development']=[];meta={}
    for name,ss in sets.items():
        for s in ss:
            player=int(s['decision_id'].split(':')[1]);ply=round(float(np.asarray(s['x'])[4].max())*240);meta[s['decision_id']]=dict(decision_id=s['decision_id'],player=player,own_move=ply//2+1)
    for row in labels:
        g=replay(row['prefix']);actions=[action_id(s,p[-1],g.player,True) for s,p in r.get_all_moves(g.board,g.player)];pos=action_id(row['preferred'][0],row['preferred'][-1],g.player,True);identity=row['decision_id'];sets['fresh_development'].append(dict(x=encode(g.board,g.player,g.ply,g.repetitions,canonical=True),actions=actions,target=actions.index(pos),positive=[pos],decision_id=identity,z=0.));meta[identity]=dict(decision_id=identity,player=g.player,own_move=row['own_move'])
    paths=dict(v3=Path(cfg['model']),control=a.source/'control_step2400.pt',projected=a.source/'projected_step2400.pt');results={};manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()}
    for f in [Path(__file__),Path('tests/test_diagnose121_transfer.py'),a.source/'config.json',a.source/'development.json',a.source/'development_choices.json',Path('opening121_book.json')]:manifest[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    for name,path in paths.items():
        model=Net121.load(path);results[name]={part:evaluate(model,ss,meta) for part,ss in sets.items()};saved={x['decision_id']:x for x in choices[name]}
        for row in results[name]['fresh_development']:
            stored=saved[row['decision_id']];path=stored['choice'];expected=action_id(path[0],path[-1],row['player'],True)
            if row['chosen']!=expected:raise ValueError('事后forward首选与冻结评测不一致，不重评教师')
            row.update(saved_chosen_gap=stored['gap'],saved_chosen_score=stored['score'])
    distribution={}
    for part,ss in sets.items():
        rows=[dict(**meta[s['decision_id']],**geometry(s['decision_id'])) for s in ss]
        def stats(xx):return dict(n=len(xx),mean_exited_home=float(np.mean([x['exited_home'] for x in xx])),mean_forward_y=float(np.mean([x['mean_forward_y'] for x in xx])),mean_forward_y_hist=np.mean([x['forward_y_hist'] for x in xx],axis=0).tolist()) if xx else dict(n=0)
        distribution[part]=dict(overall=stats(rows),players={str(p):stats([x for x in rows if x['player']==p]) for p in (1,2)},late_moves7_10=stats([x for x in rows if x['own_move']>=7]))
    summary={name:{part:stratify(rows) for part,rows in parts.items()} for name,parts in results.items()};comparisons={new+'_vs_'+old:{part:compare(results[new][part],results[old][part]) for part in sets} for new,old in [('control','v3'),('projected','v3'),('projected','control')]}
    output=dict(scope='事后只读机制诊断；无训练/教师查询/对局/模型选择；既有fresh64已是开发历史，不是重新独立验收。',target_definitions=dict(train='原开局库合法推荐集合',validation='原开局库集合，含已知转方轨道交叉，只诊断',fresh_development='冻结depth5教师单一建议动作；不是完整best-set，别的专家同分动作不算此单建议命中'),rank_definition='严格高于目标最大logit的合法动作数+1；首选同分仍按合法动作原顺序',manifest=manifest,summary=summary,distribution=distribution,comparisons=comparisons,cases=results)
    (a.out/'transfer.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(summary=summary,distribution=distribution,comparisons={k:{p:v['overall'] for p,v in vv.items()} for k,vv in comparisons.items()}),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
