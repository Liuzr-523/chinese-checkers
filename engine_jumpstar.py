"""Experimental bridge to the public JumpStar play service.

The local game has only 81 of JumpStar's 121 cells and different terminal rules.
Never silently replace a move; the adapter raises RuleMismatch instead.
The remote endpoint is used by the official public game; calls send board positions.
"""
import json
import os
import subprocess
import time
from pathlib import Path
import Chinese_Checker as cc

HERE=Path(__file__).resolve().parent
DEFAULT_ENGINE=HERE/'nn_checkpoints/third_party/jumpstar/build/cczero'
URL='https://chinesecheckers.ai/api/champion-move'

class RuleMismatch(RuntimeError):
    pass


def _star_coordinates():
    coords={(q,r) for q in range(-4,5) for r in range(-4,5)
            if max(abs(q),abs(r),abs(q+r))<=4}
    for d in range(1,5):
        for q in range(d,5):
            a,b=q,-4-d
            for _ in range(6):
                coords.add((a,b));a,b=-b,a+b
    return sorted(coords,key=lambda c:(c[1],c[0]))

STAR_COORDS=_star_coordinates()
STAR_IDS={xy:i for i,xy in enumerate(STAR_COORDS)}
LOCAL_TO_STAR=[STAR_IDS[(x-4,y-8)] for x,y in cc.POS_TO_XY]
STAR_TO_LOCAL={v:k for k,v in enumerate(LOCAL_TO_STAR)}
assert len(STAR_COORDS)==121 and len(STAR_TO_LOCAL)==81
assert LOCAL_TO_STAR[:10]==list(range(10)) and LOCAL_TO_STAR[-10:]==list(range(111,121))


def encode(board):
    if len(board)!=81 or any(x not in (0,1,2) for x in board):raise ValueError('invalid local board')
    cells=['.']*121
    for local,piece in enumerate(board):
        if piece:cells[LOCAL_TO_STAR[local]]=str(piece-1)
    return ''.join(cells)


def reference(board,player,ply,engine_path=DEFAULT_ENGINE):
    proc=subprocess.run([str(engine_path),'position-info','--cells',encode(board),'--player',str(player-1),
                         '--ply',str(ply),'--rules','strict'],capture_output=True,text=True,timeout=10,check=True)
    return json.loads(proc.stdout)


def remote_move(board,player,ply,simulations=512):
    payload=dict(cells=encode(board),player=player-1,playerCount=2,player_count=2,
                 ruleProfile='CCZ-121-Strict-LG-v2',rule_profile='CCZ-121-Strict-LG-v2',
                 ply=ply,simulations=simulations)
    # curl uses the system trust store; do not disable TLS verification.
    proc=subprocess.run(['curl','-fsS','--max-time','60','-H','Content-Type: application/json',
                         '-H','Origin: https://chinesecheckers.ai','--data-binary','@-',URL],
                        input=json.dumps(payload),capture_output=True,text=True,timeout=65,check=True)
    data=json.loads(proc.stdout)
    if not isinstance(data.get('move'),dict):raise RuntimeError('JumpStar response has no move')
    return data


def checked_path(board,player,path121,official):
    legal={(m['from'],m['to']) for m in official['legal']}
    if len(path121)<2 or (path121[0],path121[-1]) not in legal:
        raise RuleMismatch('走法不符合 JumpStar 官方规则')
    if any(p not in STAR_TO_LOCAL for p in path121):
        raise RuleMismatch('走法进入本项目棋盘以外的侧营')
    path=[STAR_TO_LOCAL[p] for p in path121]
    if (path[0],path[-1]) not in {(s,p[-1]) for s,p in cc.get_all_moves(board,player)}:
        raise RuleMismatch('走法不符合本项目规则')
    # Verify each hop, including the actual route rather than merely the endpoint.
    current=board.copy()
    if current[path[0]]!=player:raise RuleMismatch('起点棋子不属于当前方')
    if len(path)==2 and path[1] in cc.get_step_moves(path[0],current):
        goal = set(range(71,81) if player==1 else range(10))
        if path[0] in goal and path[1] not in goal:raise RuleMismatch('离开目标营地，违反 JumpStar 规则')
        return path
    goal = set(range(71,81) if player==1 else range(10))
    entered_goal = path[0] in goal
    visited={path[0]}
    for a,b in zip(path,path[1:]):
        if b in visited or b not in cc.get_jump_moves(a,current):raise RuleMismatch('连跳路径不合法')
        if entered_goal and b not in goal:raise RuleMismatch('连跳离开目标营地，违反 JumpStar 规则')
        entered_goal = entered_goal or b in goal
        current[a]=0;current[b]=player;visited.add(b)
    return path


class JumpStarEngine:
    """Explicitly experimental: cannot represent side arms or full strict rules."""
    name='JumpStar 在线（规则兼容实验）'
    def __init__(self,simulations=512,engine_path=DEFAULT_ENGINE):
        if not Path(engine_path).is_file():raise FileNotFoundError('需先编译 JumpStar 官方裁判')
        self.simulations=simulations;self.engine_path=Path(engine_path);self.ply=0
    def set_ply(self, ply):
        self.ply = int(ply)

    def think(self,board,player):
        info=reference(board,player,self.ply,self.engine_path)
        if info['terminal']['terminal']:raise RuleMismatch('JumpStar 规则已终局')
        start=time.monotonic();data=remote_move(board,player,self.ply,self.simulations)
        move=data['move'];path=checked_path(board,player,move.get('path') or [move['from'],move['to']],info)
        return path,dict(engine=self.name,remote=data,seconds=time.monotonic()-start)
