"""WDL概率校准：Brier三类平方和、NLL、根方points误差与十箱ECE。"""
import math
from collections import defaultdict
import numpy as np
import torch
from wdl121 import sample_class


def summarize(records):
    if not records:return dict(n=0,brier3=None,nll=None,expectedpoint_mse=None,ece10=None,accuracy=None,bins=[])
    brier=nll=mse=correct=0.;bins=[[] for _ in range(10)]
    for row in records:
        p=np.asarray(row['probabilities'],dtype=float);label=row['label']
        if p.shape!=(3,) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1.,atol=1e-6):raise ValueError('非法WDL概率')
        if type(label) is not int or label not in (0,1,2):raise ValueError('非法WDL label')
        target=np.eye(3)[label];brier+=float(((p-target)**2).sum());nll-=math.log(max(float(p[label]),1e-300))
        mse+=(float(p[0]+.5*p[1])-(1.,.5,0.)[label])**2
        hit=int(int(p.argmax())==label);correct+=hit;confidence=float(p.max());bins[min(int(confidence*10),9)].append((confidence,hit))
    n=len(records);calibration=[];ece=0.
    for index,items in enumerate(bins):
        count=len(items);confidence=float(np.mean([x[0] for x in items])) if count else None;accuracy=float(np.mean([x[1] for x in items])) if count else None
        if count:ece+=count/n*abs(confidence-accuracy)
        calibration.append(dict(lower=index/10,upper=(index+1)/10,n=count,confidence=confidence,accuracy=accuracy))
    return dict(n=n,brier3=brier/n,nll=nll/n,expectedpoint_mse=mse/n,ece10=ece,accuracy=correct/n,bins=calibration)


def metrics_from_probabilities(samples,probabilities):
    if len(samples)!=len(probabilities):raise ValueError('样本与概率数量不一致')
    records=[]
    for s,p in zip(samples,probabilities):
        label=sample_class(s)
        if label is None or s.get('wdl_weight',s.get('value_weight',1.))<=0:continue
        remaining=s.get('terminal_ply',s.get('game_plies'))
        late=bool(s.get('late',remaining is not None and 0<=remaining-s.get('ply',0)<=16))
        records.append(dict(probabilities=np.asarray(p).tolist(),label=label,player=s['player'],game_id=str(s.get('game_id',s.get('group','unknown'))),late=late,ply=s.get('ply')))
    games=defaultdict(list)
    for row in records:games[row['game_id']].append(row)
    per_game={key:summarize(rows) for key,rows in sorted(games.items())}
    keys=('brier3','nll','expectedpoint_mse','ece10','accuracy')
    macro={key:float(np.mean([row[key] for row in per_game.values()])) if per_game else None for key in keys}
    return dict(overall=summarize(records),by_player={str(p):summarize([row for row in records if row['player']==p]) for p in (1,2)},late=summarize([row for row in records if row['late']]),by_game=per_game,game_macro=dict(n_games=len(per_game),**macro),records=records,definitions=dict(classes=['win','draw','loss'],brier3='三类预测概率减onehot后的平方和；不除以3',expectedpoint_mse='(pwin+0.5*pdraw-真实根方points)^2',ece10='max概率置信度十等宽箱，各箱|平均置信度-argmax命中率|按样本数加权；概率1归最后箱',late='显式late或真实终局前含终局的16个绝对ply；无终局ply不推测'))


@torch.no_grad()
def wdl_metrics(model,samples):
    probabilities=[];was=model.training;flags=[p.requires_grad for p in model.parameters()];model.eval();model.requires_grad_(False)
    try:
        for off in range(0,len(samples),32):
            ss=samples[off:off+32]
            probabilities.extend(model.predict_wdl(torch.from_numpy(np.stack([s['x'] for s in ss])).float()).cpu().numpy().tolist())
    finally:
        model.train(was)
        for parameter,flag in zip(model.parameters(),flags):parameter.requires_grad_(flag)
    return metrics_from_probabilities(samples,probabilities)
