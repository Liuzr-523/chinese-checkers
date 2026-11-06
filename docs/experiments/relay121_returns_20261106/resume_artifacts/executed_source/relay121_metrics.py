"""真实续弈偏好的pair排序指标；points和胜局tempo分开，不造搜索分数。"""
import math
import numpy as np
import torch
import rules121 as r
from az121 import encode,action_id
from curriculum121 import replay
from relay121_targets import stable_preference


def _distance(index,player):
    q,y=r.COORDS[index];a,b=r.COORDS[120 if player==1 else 0]
    return max(abs(q-a),abs(y-b),abs(q+y-a-b))


def _mechanism(row,g):
    rear,front=row['rear_path'],row['front_path'];player=g.player
    if not all(r.valid_path(g.board,player,p) for p in (rear,front)):raise ValueError('候选非法')
    maximum=max(_distance(i,player) for i,v in enumerate(g.board) if v==player)
    if len(rear)!=2 or rear[-1] not in r.NEIGHBORS[rear[0]] or _distance(rear[0],player)!=maximum or _distance(rear[-1],player)>=maximum:raise ValueError('rear必须为最后方棋子的前进相邻小步')
    if len(front)<3 or _distance(front[0],player)>=maximum or _distance(front[-1],player)>=_distance(front[0],player):raise ValueError('front必须是非最后子已有前进连跳')
    witness=row['mechanism_witness'];chain=witness['best_new_chain'];child=r.apply_move(g.board,rear)
    old_ends={p[-1] for p in r.legal_paths(g.board,rear[0]) if _distance(p[-1],player)<maximum}
    if len(chain)<3 or chain[0]!=rear[-1] or not r.valid_path(child,player,chain) or chain[-1] in old_ends or _distance(chain[-1],player)>=_distance(chain[0],player):raise ValueError('没有新增前进连跳落点证据')
    advance=_distance(chain[0],player)-_distance(chain[-1],player)
    if witness['new_chain_advance']!=advance or witness['rear_start_tip_distance']!=maximum:raise ValueError('接力证据数值不符')


def _verify_return(game,row,choice,depth):
    g=replay(row['opening']);path=row[choice+'_path']
    if game['opening']!=row['opening'] or game['start_ply']!=g.ply or game['root_player']!=g.player or game['root_ply']!=g.ply:raise ValueError('分支根不一致')
    if game.get('group',row['group'])!=row['group'] or game.get('depth',depth)!=depth or game.get('choice',choice)!=choice or game.get('forced_path',path)!=path:raise ValueError('分支group/depth/choice/forced metadata不一致')
    paths=game['paths'];infos=game['search_info']
    if not paths or paths[0]!=path or len(infos)!=len(paths):raise ValueError('forced首步或逐步审计缺失')
    for i,(move,info) in enumerate(zip(paths,infos)):
        if info['path']!=move or info['player']!=g.player or info['ply']!=g.ply:raise ValueError('逐步路径/方/ply审计不一致')
        if i==0:
            if info.get('forced') is not True:raise ValueError('首步须明确forced')
        elif info.get('forced') is not False or info['requested_depth']!=depth or info['depth_reached']!=depth or info['aborted']:raise ValueError('续弈深度未完成')
        g.play(move)
    if g.ply!=game['plies'] or g.result()!=(game['winner'],game['reason']):raise ValueError('归档真实终局不一致')
    if game['own_moves']!=(g.ply-row['ply']+1)//2:raise ValueError('落子数不含forced或计算错误')


def prepare_rows(raw_rows):
    out=[]
    for row in raw_rows:
        if not row['eligible']:continue
        g=replay(row['opening'])
        if g.board!=row['board'] or g.player!=row['player'] or g.ply!=row['ply']:raise ValueError('根前缀与输入不一致')
        _mechanism(row,g);pref=stable_preference(row['returns'],g.player)
        if not pref['eligible'] or pref['label']!=row['label']:raise ValueError('最终label与稳定真实续弈方向不一致')
        for depth in (3,4):
            branches=row['returns'].get(str(depth),row['returns'].get(depth))
            for choice in ('rear','front'):_verify_return(branches[choice],row,choice,depth)
        a=action_id(row['rear_path'][0],row['rear_path'][-1],g.player,True);b=action_id(row['front_path'][0],row['front_path'][-1],g.player,True)
        actions=[action_id(s,p[-1],g.player,True) for s,p in r.get_all_moves(g.board,g.player)]
        if a==b or a not in actions or b not in actions:raise ValueError('候选端点不构成两合法动作')
        out.append({**row,'x':encode(g.board,g.player,g.ply,g.repetitions,canonical=True),'actions':actions,'rear_action':a,'front_action':b,'_relay_preference':pref,'value_weight':0.,'z':0.})
    return out


def pair_case(rear_logit,front_logit,preference):
    if not math.isfinite(float(rear_logit)) or not math.isfinite(float(front_logit)) or not preference['eligible']:raise ValueError('无效偏好或logit')
    gap=float(rear_logit)-float(front_logit);label=preference['label'];prediction=1 if gap>0 else -1 if gap<0 else 0
    regrets={}
    for depth in ('3','4'):
        p=preference['per_depth'][depth];wrong=prediction!=p['label'];point=p['point_difference']
        if point is None:raise ValueError('eligible偏好不可包含截断/未完成分支')
        # 同point时实际point损失为0；tempo只在两方都胜时适用。
        tempo_applicable=p['kind']=='tempo'
        regrets[depth]=dict(point_regret=abs(point) if wrong else 0.,winning_tempo_regret=abs(p['own_moves_rear']-p['own_moves_front']) if tempo_applicable and wrong else 0. if tempo_applicable else None)
    return dict(label=label,direction='rear' if label==1 else 'front',kind=preference['kind'],prediction=prediction,tie=prediction==0,correct=prediction==label,logit_gap=gap,oriented_margin=label*gap,pair_ce=float(np.logaddexp(0.,-label*gap)),per_depth=regrets)


def summarize(rows):
    def scalar(ss):return dict(n=len(ss),accuracy=sum(x['correct'] for x in ss)/len(ss) if ss else None,pair_ce=float(np.mean([x['pair_ce'] for x in ss])) if ss else None)
    out=scalar(rows);out['directions']={s:scalar([x for x in rows if x['direction']==s]) for s in ('rear','front')};out['balanced_accuracy']=float(np.mean([v['accuracy'] for v in out['directions'].values()])) if all(v['n'] for v in out['directions'].values()) else None;out['ties']=sum(x['tie'] for x in rows);out['per_depth']={}
    for depth in ('3','4'):
        tempo=[x['per_depth'][depth]['winning_tempo_regret'] for x in rows if x['per_depth'][depth]['winning_tempo_regret'] is not None]
        out['per_depth'][depth]=dict(n=len(rows),point_regret_mean=float(np.mean([x['per_depth'][depth]['point_regret'] for x in rows])) if rows else None,winning_tempo_n=len(tempo),winning_tempo_regret_mean=float(np.mean(tempo)) if tempo else None)
    out['all_legal_argmax']=dict(n=len(rows),in_pair=sum(x.get('argmax_in_pair',False) for x in rows),stable_preferred=sum(x.get('argmax_preferred',False) for x in rows));return out


@torch.no_grad()
def relay_metrics(model,rows):
    rows=list(rows)
    if rows and '_relay_preference' not in rows[0]:rows=prepare_rows(rows)
    records=[];was=model.training;flags=[p.requires_grad for p in model.parameters()];model.eval();model.requires_grad_(False)
    try:
        for off in range(0,len(rows),32):
            ss=rows[off:off+32];logits,_=model(torch.stack([torch.as_tensor(s['x'],dtype=torch.float32) for s in ss]))
            for s,l in zip(ss,logits):
                a,b=s['rear_action'],s['front_action'];stats=pair_case(float(l[a]),float(l[b]),s['_relay_preference']);chosen=s['actions'][int(l[s['actions']].argmax())];preferred=a if s['label']==1 else b
                records.append(dict(group=s['group'],split=s['split'],player=s['player'],ply=s['ply'],rear_logit=float(l[a]),front_logit=float(l[b]),rear_action=a,front_action=b,argmax_action=chosen,argmax_in_pair=chosen in (a,b),argmax_preferred=chosen==preferred,**stats))
    finally:
        model.train(was)
        for p,flag in zip(model.parameters(),flags):p.requires_grad_(flag)
    out=summarize(records);out['players']={str(p):summarize([x for x in records if x['player']==p]) for p in (1,2)};out['cases']=records;out['tempo_gate_scope']={d:'applicable' if out['per_depth'][d]['winning_tempo_n'] else 'not_applicable_no_both_win_pairs' for d in ('3','4')};return out


def relay_gate(candidate,before):
    c,b=candidate,before;ba=c['balanced_accuracy'];bb=b['balanced_accuracy']
    checks=dict(development_coverage=c['n']>=8 and all(c['directions'][s]['n']>=2 for s in ('rear','front')),balanced_accuracy75=ba is not None and ba>=.75,balanced_gain125=ba is not None and bb is not None and ba>=bb+.125,pair_ce_nonworse=c['pair_ce'] is not None and b['pair_ce'] is not None and c['pair_ce']<=b['pair_ce'])
    for side in ('rear','front'):checks[side+'_accuracy60']=c['directions'][side]['accuracy'] is not None and c['directions'][side]['accuracy']>=.6
    for depth in ('3','4'):
        x,y=c['per_depth'][depth],b['per_depth'][depth];checks['depth'+depth+'_point_regret_nonworse']=x['point_regret_mean'] is not None and y['point_regret_mean'] is not None and x['point_regret_mean']<=y['point_regret_mean']
        no_tempo=x['winning_tempo_n']==y['winning_tempo_n']==0
        checks['depth'+depth+'_tempo_regret_nonworse']=no_tempo or (x['winning_tempo_n']==y['winning_tempo_n'] and x['winning_tempo_n']>0 and x['winning_tempo_regret_mean'] is not None and y['winning_tempo_regret_mean'] is not None and x['winning_tempo_regret_mean']<=y['winning_tempo_regret_mean'])
    return all(checks.values()),checks
