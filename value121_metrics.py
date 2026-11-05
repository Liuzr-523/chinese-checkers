"""终局价值的固定开发评测；先验仅从训练样本计算，不选择模型。"""
from collections import defaultdict
import math
import numpy as np
import torch


def _player(sample):
    value=sample.get('player')
    if value is None:
        value=1 if float(torch.as_tensor(sample['x'])[3].sum())>0 else 2
    if value not in (1,2):raise ValueError('player必须为1或2')
    return int(value)


def _target(sample):
    z=float(sample['z'])
    if not math.isfinite(z) or abs(z)>1:raise ValueError('价值目标须有限且在[-1,1]')
    if float(sample.get('value_weight',1))<=0:raise ValueError('价值评测不接受无真实价值标签的样本')
    return z


def _late(sample):
    if 'late16' in sample:return bool(sample['late16'])
    if 'terminal_distance' in sample:return 0<=int(sample['terminal_distance'])<=16
    if 'remaining' in sample:return 0<=int(sample['remaining'])<=16
    raise ValueError('缺少late16/terminal_distance/remaining标记，不从编码猜测终局距离')


def _correlation(a,b):
    if len(a)<2 or np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def _summarize(rows):
    if not rows:return dict(n=0,groups=0,point=None,group_macro=None,group_macro_mse=None,zero_group_macro_mse=None,prior_group_macro_mse=None,players={})
    pred=np.array([x['prediction'] for x in rows]);z=np.array([x['z'] for x in rows]);groups=defaultdict(list)
    for row in rows:groups[row['group']].append(row)
    point=dict(mse=float(np.mean((pred-z)**2)),zero_mse=float(np.mean(z*z)),train_player_prior_mse=float(np.mean([(x['prior']-x['z'])**2 for x in rows])),sign_accuracy=float(np.mean(np.sign(pred)==np.sign(z))),correlation=_correlation(pred,z),prediction_mean=float(pred.mean()),prediction_std=float(pred.std()),target_mean=float(z.mean()),target_std=float(z.std()))
    macro={name:float(np.mean([np.mean([({'mse':x['prediction'],'zero_mse':0.,'train_player_prior_mse':x['prior']}[name]-x['z'])**2 for x in rr]) for rr in groups.values()])) for name in ('mse','zero_mse','train_player_prior_mse')}
    return dict(n=len(rows),groups=len(groups),point=point,group_macro=macro,group_macro_mse=macro['mse'],zero_group_macro_mse=macro['zero_mse'],prior_group_macro_mse=macro['train_player_prior_mse'])


def _stratify(rows):
    out=_summarize(rows);out['players']={str(p):_summarize([x for x in rows if x['player']==p]) for p in (1,2)};return out


@torch.no_grad()
def value_metrics(model,samples,train_samples):
    """CPU canonical样本。group宏平均先对每组全部样本求MSE，再等权平均组。

    必需元数据group、game_id、late16(或terminal_distance/remaining)；player可由身份平面推断。
    late16门槛采用组宏平均；点平均同时独立提供。所有预测用于报告，不改权重。
    """
    samples=[s for s in samples if float(s.get('value_weight',1))>0];train_samples=[s for s in train_samples if float(s.get('value_weight',1))>0]
    if not samples or not train_samples:raise ValueError('开发和训练样本均不可为空')
    train_by_player={p:[_target(s) for s in train_samples if _player(s)==p] for p in (1,2)}
    if any(not v for v in train_by_player.values()):raise ValueError('训练缺少一方，拒绝用开发标签补先验')
    priors={p:float(np.mean(v)) for p,v in train_by_player.items()};rows=[];was_training=model.training;model.eval()
    try:
        for offset in range(0,len(samples),32):
            chunk=samples[offset:offset+32];x=torch.stack([torch.as_tensor(s['x'],dtype=torch.float32) for s in chunk]);_,value=model(x)
            predictions=value.detach().cpu().reshape(-1).tolist()
            if len(predictions)!=len(chunk):raise ValueError('价值输出数量与样本不一致')
            for s,pred in zip(chunk,predictions):
                if not math.isfinite(pred):raise ValueError('非有限价值预测')
                if 'group' not in s or 'game_id' not in s:raise ValueError('缺少group/game_id，不能核验宏平均层次')
                p=_player(s);rows.append(dict(group=str(s['group']),game_id=str(s['game_id']),player=p,late16=_late(s),z=_target(s),prediction=float(pred),prior=priors[p]))
    finally:model.train(was_training)
    out=_stratify(rows);out['late16']=_stratify([x for x in rows if x['late16']]);out['train_player_prior']={str(p):v for p,v in priors.items()};out['prior_source']='仅训练样本按走棋方的点目标均值，未使用开发胜者比例'
    groups=defaultdict(list);games=defaultdict(list)
    for row in rows:groups[row['group']].append(row);games[(row['group'],row['game_id'])].append(row)
    out['per_group']={k:_stratify(v) for k,v in sorted(groups.items())};out['per_game']=[dict(group=g,game_id=i,metrics=_stratify(v)) for (g,i),v in sorted(games.items())];out['predictions']=rows
    return out


def value_gate(candidate,before):
    """纯value门槛；全部/late/分方均按group宏平均，旧能力和policy由trainer合并。"""
    c,b=candidate,before;macro=c['group_macro'];late=c['late16'];oldlate=b['late16']
    checks=dict(development_gain05=macro['mse']<=b['group_macro']['mse']-.05,development_better_zero=macro['mse']<macro['zero_mse'],development_better_train_prior=macro['mse']<macro['train_player_prior_mse'])
    for p in ('1','2'):
        cp=late['players'][p];bp=oldlate['players'][p];checks['late16_player'+p]=bool(cp['n'] and bp['n'] and cp['group_macro_mse']<=bp['group_macro_mse']+1e-8)
    checks['late16_better_zero']=bool(late['n'] and late['group_macro_mse']<late['zero_group_macro_mse']);checks['late16_better_train_prior']=bool(late['n'] and late['group_macro_mse']<late['prior_group_macro_mse'])
    return all(checks.values()),checks


def gate(candidate,before):
    """可选兼容包装：development为value_metrics，并合并旧能力三指标。"""
    _,checks=value_gate(candidate['development'],before['development'])
    checks.update(old_policy=candidate['retention']['policy_ce']<=before['retention']['policy_ce']+.05,old_value=candidate['retention']['value_mse']<=before['retention']['value_mse']+.05,normal_policy=candidate['normal']['policy_ce']<=before['normal']['policy_ce']+.05)
    return all(checks.values()),checks
