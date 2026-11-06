"""两配置完整续弈的局部动作偏好；截断不作和棋或价值标签。"""

WIN_REASONS={'all_pieces_in_goal','anti_block_goal_full'}
DRAW_REASONS={'equal_turn_goal_draw','repetition','no_legal_moves'}


def _outcome(game,player):
    winner=game.get('winner');reason=game.get('reason','')
    if reason=='max_ply':return None,'max_ply'
    if winner is None:return None,'unfinished'
    if winner not in (0,1,2):return None,'invalid_winner'
    if (winner==0 and reason not in DRAW_REASONS) or (winner in (1,2) and reason not in WIN_REASONS):return None,'not_genuine_terminal'
    start=game.get('start_ply');end=game.get('plies')
    if not isinstance(start,int) or not isinstance(end,int) or end<=start:raise ValueError('续弈手数非法或未包含forced动作')
    if game.get('root_ply',start)!=start or game.get('root_player',player)!=player:raise ValueError('续弈根metadata不一致')
    if 'opening' in game and len(game['opening'])!=start:raise ValueError('opening应为未走forced的rootprefix')
    if 'paths' in game and len(game['paths'])!=end-start:raise ValueError('paths须包含forced首步和全部后续')
    own=(end-start+1)//2
    if 'own_moves' in game and game['own_moves']!=own:raise ValueError('own_moves必须包含forced一次')
    return dict(point=.5 if winner==0 else float(winner==player),own_moves=own,winner=winner),None


def compare_returns(rear_game,front_game,root_player):
    if root_player not in (1,2):raise ValueError('根方须为1或2')
    if rear_game.get('start_ply')!=front_game.get('start_ply'):raise ValueError('两分支必须来自同一root ply')
    a,ra=_outcome(rear_game,root_player);b,rb=_outcome(front_game,root_player)
    out=dict(label=0,direction=0,kind='none',point_difference=None,own_moves_rear=a['own_moves'] if a else None,own_moves_front=b['own_moves'] if b else None,reason='')
    if ra or rb:out['reason']='invalid_return:'+str(ra or rb);return out
    difference=a['point']-b['point'];out['point_difference']=difference
    if difference!=0:out.update(label=1 if difference>0 else -1,kind='point',reason='genuine_terminal_points')
    elif a['winner']==root_player and b['winner']==root_player and abs(a['own_moves']-b['own_moves'])>=2:
        out.update(label=1 if a['own_moves']<b['own_moves'] else -1,kind='tempo',reason='both_win_own_moves_gap_at_least2')
    else:out['reason']='same_outcome_no_tempo_preference'
    out['direction']=out['label'];return out


def stable_preference(returns,player):
    bydepth={}
    for depth in (3,4):
        pair=returns.get(str(depth),returns.get(depth))
        if pair is None:raise ValueError('缺少预声明depth3或4续弈')
        bydepth[str(depth)]=compare_returns(pair['rear'],pair['front'],player)
    labels=[x['label'] for x in bydepth.values()];eligible=labels[0]!=0 and labels[0]==labels[1];kinds={x['kind'] for x in bydepth.values()}
    return dict(label=labels[0] if eligible else 0,direction=labels[0] if eligible else 0,eligible=eligible,kind=next(iter(kinds)) if eligible and len(kinds)==1 else 'mixed' if eligible else 'none',per_depth=bydepth,reason='stable_nonzero_direction' if eligible else 'depths_tie_or_disagree_or_invalid')
