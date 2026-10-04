"""
人机对战（mediated 版）：状态存 game_state.json，每调用一次做一步。
网格序号 0~80 格式表示走法。红●=你，青●=AI。

用法：
  python3 play_turn.py new 2      新局，人=P2(后手,71-80)；AI 先手则自动走
  python3 play_turn.py new 1      新局，人=P1(先手,0-9)
  python3 play_turn.py moves      列出你的合法走法（序号 → 序号）
  python3 play_turn.py do 77 69    你走 77→69（连跳传完整路径：12 23 40 57）
  python3 play_turn.py board      只显示棋盘
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
from engine import AlphaBetaIDSEngine

cc = engine.cc
AI = AlphaBetaIDSEngine(time_limit_sec=1.5, max_depth=8, eval_version="v2")
STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "game_state.json")

RED = '\033[91m'
CYAN = '\033[96m'
RST = '\033[0m'


def load():
    with open(STATE_PATH) as f:
        return json.load(f)


def save(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def fmt_path(path):
    return ' → '.join(str(p) for p in path)


def step_kind(path):
    n = len(path)
    if n == 2:
        return '单步'
    if n == 3:
        return '单跳'
    return f'连跳{n - 2}'


def show_board(board, human):
    ai = 3 - human
    rows = {}
    for pos in range(81):
        x, y = cc.POS_TO_XY[pos]
        rows.setdefault(y, {})[x] = pos
    max_x = max(cc.POS_TO_XY[p][0] for p in range(81))
    lines = []
    for y in sorted(rows.keys()):
        xs = rows[y]
        offset = (max_x - max(xs.keys())) // 2
        line = ' ' * (offset * 3)
        for x in sorted(xs.keys()):
            pos = xs[x]
            if board[pos] == human:
                line += f'{RED}●{RST} '
            elif board[pos] == ai:
                line += f'{CYAN}●{RST} '
            else:
                line += f'{pos:2d} '
        lines.append(line.rstrip())
    return '\n'.join(lines)


def print_ai_move(path, info):
    depth = info.get('depth_reached', '?')
    nodes = info.get('nodes_searched', '?')
    score = info.get('score', 0)
    try:
        score = round(float(score), 1)
    except (TypeError, ValueError):
        score = info.get('score', '?')
    print(f"AI(青) 走：[{step_kind(path)}] {fmt_path(path)}  "
          f"(深度{depth}, {nodes} 节点, 分 {score})")


def print_human_moves(moves):
    if not moves:
        print("你无棋可走，AI 获胜。")
        return
    print(f"你的棋子(红)可走（共 {len(moves)} 种）：")
    for i, (pos, path) in enumerate(moves, 1):
        print(f"  {i:3d}. [{step_kind(path):4s}] {fmt_path(path)}")


def do_ai_turn(state):
    """AI 走一步并落盘。返回 path 或 None（无棋可走）。"""
    board = state['board']
    ap = state['ai_player']
    path, info = AI.think(board, ap)
    if path is None:
        state['winner'] = 3 - ap
        save(state)
        return None, info
    state['board'] = cc.apply_move(board, path)
    state['history'].append([ap, path])
    w = cc.is_game_over(state['board'])
    if w:
        state['winner'] = w
    save(state)
    return path, info


def do_human_turn(state, path):
    """人走给定 path 并落盘。返回是否触发终局。"""
    state['board'] = cc.apply_move(state['board'], path)
    state['history'].append([state['human_player'], path])
    w = cc.is_game_over(state['board'])
    if w:
        state['winner'] = w
    save(state)
    return w is not None


def parse_path_args(argv):
    """把命令行参数解析成 path 整数列表，支持 '77 69' 和 '77→69' '77->69'。"""
    joined = ' '.join(argv)
    for sep in ['→', '->', '—']:
        joined = joined.replace(sep, ' ')
    parts = [p for p in joined.split() if p != '']
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None


def find_matching_move(moves, path):
    for pos, mpath in moves:
        if list(mpath) == list(path):
            return mpath
    return None


def cmd_new(human):
    board = cc.init_board()
    state = {
        'board': list(board),
        'human_player': human,
        'ai_player': 3 - human,
        'winner': None,
        'history': [],
    }
    save(state)
    print(f"新局开始：你=P{human}({'先手, 0-9' if human == 1 else '后手, 71-80'}), "
          f"AI=P{state['ai_player']}")
    print()
    print(show_board(state['board'], human))
    print()
    # 若 AI 先手，AI 先走
    if state['ai_player'] == 1:
        path, info = do_ai_turn(state)
        if path is None:
            print("AI 无棋可走。")
            return
        print_ai_move(path, info)
        if state.get('winner'):
            print(f"\n对局结束：P{state['winner']} 获胜！")
            return
        print()
        print(show_board(state['board'], human))
        print()
    # 列出人的合法走法
    moves = cc.get_all_moves(state['board'], human)
    print_human_moves(moves)


def cmd_moves():
    state = load()
    if state.get('winner'):
        print(f"对局已结束：P{state['winner']} 获胜。")
        return
    moves = cc.get_all_moves(state['board'], state['human_player'])
    print_human_moves(moves)


def cmd_do(argv):
    state = load()
    if state.get('winner'):
        print(f"对局已结束：P{state['winner']} 获胜。")
        return
    path = parse_path_args(argv)
    if not path or len(path) < 2:
        print("走法格式：python3 play_turn.py do 77 69  （连跳：12 23 40 57）")
        return
    moves = cc.get_all_moves(state['board'], state['human_player'])
    matched = find_matching_move(moves, path)
    if matched is None:
        print(f"非法走法：{fmt_path(path)}。用 `python3 play_turn.py moves` 查合法走法。")
        return
    # 人走
    print(f"你(红) 走：[{step_kind(matched)}] {fmt_path(matched)}")
    over = do_human_turn(state, matched)
    if over:
        print()
        print(show_board(state['board'], state['human_player']))
        print(f"\n对局结束：P{state['winner']} 获胜！")
        return
    # AI 走
    path2, info = do_ai_turn(state)
    if path2 is None:
        print("AI 无棋可走，你获胜。")
        return
    print_ai_move(path2, info)
    if state.get('winner'):
        print()
        print(show_board(state['board'], state['human_player']))
        print(f"\n对局结束：P{state['winner']} 获胜！")
        return
    print()
    print(show_board(state['board'], state['human_player']))
    print()
    moves = cc.get_all_moves(state['board'], state['human_player'])
    print_human_moves(moves)


def cmd_board():
    state = load()
    print(show_board(state['board'], state['human_player']))
    if state.get('winner'):
        print(f"\n对局已结束：P{state['winner']} 获胜。")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == 'new':
        if len(sys.argv) < 3 or sys.argv[2] not in ('1', '2'):
            print("用法：python3 play_turn.py new 1|2  （1=先手, 2=后手）")
            return
        cmd_new(int(sys.argv[2]))
    elif cmd == 'moves':
        cmd_moves()
    elif cmd == 'do':
        cmd_do(sys.argv[2:])
    elif cmd == 'board':
        cmd_board()
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
