"""同根两任务落子次序的策略排序诊断；不把pair指标写成实际胜率。"""
import math
import numpy as np
import torch
import rules121 as r
from curriculum121 import replay
from az121 import encode,action_id


def prepare_rows(raw_rows):
    """仅接收预声明eligible；不改split/label，不用NN选择根。"""
    out=[]
    for row in raw_rows:
        if not row['eligible']:continue
        g=replay(row['opening'])
        if list(g.board)!=list(row['board']) or g.player!=row['player'] or g.ply!=row['ply']:raise ValueError('冻结前缀与根棋盘/方/手数不符')
        rear,front=row['rear_path'],row['front_path']
        if not r.valid_path(g.board,g.player,rear) or not r.valid_path(g.board,g.player,front):raise ValueError('冻结候选不合法')
        a=action_id(rear[0],rear[-1],g.player,True);b=action_id(front[0],front[-1],g.player,True)
        if a==b:raise ValueError('两候选端点相同，不能作偏好对')
        d4=float(row['score4_rear'])-float(row['score4_front']);d5=float(row['score5_rear'])-float(row['score5_front'])
        if not math.isfinite(d4) or not math.isfinite(d5) or d4*d5<=0 or abs(d5)<2 or row['label']!=(1 if d5>0 else -1):raise ValueError('eligible深度一致性/标签不符')
        actions=[action_id(s,p[-1],g.player,True) for s,p in r.get_all_moves(g.board,g.player)]
        if a not in actions or b not in actions:raise ValueError('候选未出现在合法端点集合')
        out.append({**row,'x':encode(g.board,g.player,g.ply,g.repetitions,canonical=True),'actions':actions,'rear_action':a,'front_action':b,'teacher_gap':d5,'z':0.,'value_weight':0.})
    return out


def pair_metrics(rear_logit,front_logit,teacher_gap):
    """tie算未辨明；regret为pair未判对时abs教师gap，不是实际全合法走法损失。"""
    if not all(math.isfinite(float(x)) for x in (rear_logit,front_logit,teacher_gap)) or teacher_gap==0:raise ValueError('非有限或无方向pair')
    gap=float(rear_logit)-float(front_logit);y=1 if teacher_gap>0 else -1;correct=y*gap>0
    return dict(logit_gap=gap,label=y,direction='rear' if y>0 else 'front',correct=correct,tie=gap==0,pair_ce=float(np.logaddexp(0.,-y*gap)),score_regret=0. if correct else abs(float(teacher_gap)),oriented_margin=y*gap)


def summarize(rows):
    n=len(rows)
    def part(ss):return dict(n=len(ss),accuracy=sum(x['correct'] for x in ss)/len(ss),pair_ce=float(np.mean([x['pair_ce'] for x in ss])),score_regret_mean=float(np.mean([x['score_regret'] for x in ss])),mean_oriented_margin=float(np.mean([x['oriented_margin'] for x in ss]))) if ss else dict(n=0,accuracy=None,pair_ce=None,score_regret_mean=None,mean_oriented_margin=None)
    out=part(rows);out['directions']={s:part([x for x in rows if x['direction']==s]) for s in ('rear','front')};out['balanced_accuracy']=float(np.mean([x['accuracy'] for x in out['directions'].values()])) if all(x['n'] for x in out['directions'].values()) else None
    out['ties']=sum(x['tie'] for x in rows)
    out['all_legal_argmax']=dict(in_pair=sum(x['argmax_in_pair'] for x in rows),teacher_preferred=sum(x['argmax_teacher_preferred'] for x in rows),n=n)
    return out


@torch.no_grad()
def order_metrics(model,rows):
    rows=list(rows)
    if rows and 'x' not in rows[0]:rows=prepare_rows(rows)
    else:rows=[x for x in rows if x.get('eligible',True)]
    records=[];was=model.training;flags=[p.requires_grad for p in model.parameters()];model.eval();model.requires_grad_(False)
    try:
        for start in range(0,len(rows),32):
            ss=rows[start:start+32];x=torch.stack([torch.as_tensor(s['x'],dtype=torch.float32) for s in ss]);logits,_=model(x)
            for row,l in zip(ss,logits):
                a,b=row['rear_action'],row['front_action'];stats=pair_metrics(float(l[a]),float(l[b]),row['teacher_gap']);actions=row['actions'];chosen=actions[int(l[actions].argmax())];preferred=a if row['teacher_gap']>0 else b
                records.append(dict(group=row['group'],split=row['split'],player=row['player'],ply=row['ply'],rear_action=a,front_action=b,rear_logit=float(l[a]),front_logit=float(l[b]),teacher_gap=row['teacher_gap'],argmax_action=chosen,argmax_in_pair=chosen in (a,b),argmax_teacher_preferred=chosen==preferred,**stats))
    finally:
        model.train(was)
        for p,flag in zip(model.parameters(),flags):p.requires_grad_(flag)
    out=summarize(records);out['players']={str(p):summarize([x for x in records if x['player']==p]) for p in (1,2)};out['cases']=records;out['score_regret_definition']='pair未判对（包括logit tie）计abs(depth5教师分差)，不代表all-legal实际选择的分差或胜率';return out


def order_gate(candidate,before):
    """仅pair开发门槛；旧能力三项和value参数哈希由runner合并。"""
    c,b=candidate,before;coverage=c['n']>=8 and all(c['directions'][s]['n']>=2 for s in ('rear','front'));ba=c['balanced_accuracy'];bb=b['balanced_accuracy']
    checks=dict(development_coverage=coverage,balanced_accuracy75=ba is not None and ba>=.75,balanced_gain125=ba is not None and bb is not None and ba>=bb+.125,rear_accuracy60=c['directions']['rear']['accuracy'] is not None and c['directions']['rear']['accuracy']>=.6,front_accuracy60=c['directions']['front']['accuracy'] is not None and c['directions']['front']['accuracy']>=.6,pair_ce_nonworse=c['pair_ce'] is not None and b['pair_ce'] is not None and c['pair_ce']<=b['pair_ce'],score_regret_nonworse=c['score_regret_mean'] is not None and b['score_regret_mean'] is not None and c['score_regret_mean']<=b['score_regret_mean'])
    return all(checks.values()),checks
