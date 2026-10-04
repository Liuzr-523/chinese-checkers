"""
人机对战：你 vs 最强 AI（v2 启发函数 + 迭代加深 Alpha-Beta 搜索）。
使用方法：python3 play_vs_ai.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
from engine import AlphaBetaIDSEngine

cc = engine.cc
AI = AlphaBetaIDSEngine(time_limit_sec=1.5, max_depth=8, eval_version="v2")
AI.name = "v2-AI"


def show_board(board, human):
    """
    星型布局打印棋盘，编号 0~80（对应你图片上的序号）。
    红●=你的棋子，青●=AI 的棋子，空位=格子号。
    human 传你执的方（1 或 2），颜色按此动态着色，先/后手都正确。
    """
    ai = 3 - human
    rows = {}
    for pos in range(81):
        x, y = cc.POS_TO_XY[pos]
        rows.setdefault(y, {})[x] = pos

    you_color = '\033[91m'   # 红 = 你
    ai_color  = '\033[96m'   # 青 = AI
    reset     = '\033[0m'
    max_x = max(cc.POS_TO_XY[p][0] for p in range(81))

    lines = []
    for y in sorted(rows.keys()):
        xs = rows[y]
        offset = (max_x - max(xs.keys())) // 2
        line = '  ' + ' ' * (offset * 4)
        for x in sorted(xs.keys()):
            pos = xs[x]
            if board[pos] == human:
                line += f'{you_color}●{reset}   '
            elif board[pos] == ai:
                line += f'{ai_color}●{reset}   '
            else:
                line += f'{pos:2d}  '
        lines.append(line.rstrip())

    return '\n'.join(lines)


def choose_first():
    while True:
        ans = input("你要先手(P1, 蓝顶●=红)还是后手(P2, 橙底●=青)? [1/2] ").strip()
        if ans == '1':
            return 1
        if ans == '2':
            return 2
        print("  输入 1 或 2")


def ask_ai_move(board, ai_player):
    print(f"  AI 思考中…")
    path, info = AI.think(board, ai_player)
    pos_list = '→'.join(str(p) for p in path)
    step_type = '单步' if len(path) == 2 else (f'单跳' if len(path) == 3 else f'连跳{len(path)-2}')
    print(f"  AI 走：[{step_type}] {pos_list}")
    return path


def ask_human_move(board, player):
    moves = cc.get_all_moves(board, player)
    if not moves:
        print("  你无棋可走，AI 获胜。")
        return None

    # 先让玩家选棋子
    while True:
        my_positions = sorted([p for p in range(81) if board[p] == player])
        print("  你的棋子位置：", ' '.join(f'[{p}]' for p in my_positions))
        sel = input("  选一颗棋子编号（或按回车看全部可走列表）：").strip()

        if sel == '':
            # 展示全部合法走法
            for i, (pos, path) in enumerate(moves, 1):
                step_type = '单步' if len(path) == 2 else ('单跳' if len(path) == 3 else f'连跳{len(path)-2}')
                print(f"    {i:3d}. [{step_type:4s}] " + '→'.join(str(p) for p in path))
            while True:
                idx_s = input(f"  选择走法编号 (1~{len(moves)})：").strip()
                if idx_s.isdigit() and 1 <= int(idx_s) <= len(moves):
                    return moves[int(idx_s) - 1][1]
                print("    编号无效")

        if not sel.isdigit():
            print("  请输入数字")
            continue
        piece_pos = int(sel)
        if piece_pos not in my_positions:
            print("  那个格子不是你的棋子")
            continue

        # 展示选中棋子的所有可走路径
        piece_moves = [(i, m) for i, m in enumerate(moves) if m[0] == piece_pos]
        if not piece_moves:
            print("  这颗棋子现在走不了，换一颗")
            continue
        print(f"  {piece_pos} 号棋子的所有走法：")
        for local_i, (global_i, (pos, path)) in enumerate(piece_moves, 1):
            step_type = '单步' if len(path) == 2 else ('单跳' if len(path) == 3 else f'连跳{len(path)-2}')
            print(f"    {local_i:2d}. [{step_type:4s}] " + '→'.join(str(p) for p in path))
        while True:
            sub_s = input(f"  选择这颗棋子的走法编号 (1~{len(piece_moves)})：").strip()
            if sub_s.isdigit() and 1 <= int(sub_s) <= len(piece_moves):
                return piece_moves[int(sub_s) - 1][1][1]
            print("    编号无效")


def main():
    print("=" * 50)
    print("  中国跳棋 · 人机对战（你 vs v2-AI）")
    print("  棋盘编号对照你贴的图（0 蓝顶 ~ 80 橙底）")
    print("  红● = 你的棋子，青● = AI 的棋子")
    print("=" * 50)

    human = choose_first()
    ai = 3 - human
    names = {human: '你', ai: 'AI'}

    board = cc.init_board()
    step = 0
    while True:
        current = 1 if step % 2 == 0 else 2
        print()
        print(f"----- 第 {step//2+1} 回合 · P{current} ({names[current]}) -----")
        print(show_board(board, human))

        winner = cc.is_game_over(board)
        if winner:
            print(f"\n  🎉 对局结束：P{winner} ({names[winner]}) 获胜！")
            return

        if current == human:
            path = ask_human_move(board, human)
            if path is None:
                return
            pos_list = '→'.join(str(p) for p in path)
            print(f"  你走：{pos_list}")
        else:
            path = ask_ai_move(board, ai)

        board = cc.apply_move(board, path)
        step += 1

        if step % 20 == 0:
            # 偶尔报告进度，防止太长
            in_p1 = sum(1 for p in range(71,81) if board[p]==1)
            in_p2 = sum(1 for p in range(0,10)  if board[p]==2)
            print(f"  [第 {step//2} 手统计] 你的棋子进目标区：{in_p1 if human==1 else in_p2}/10；AI：{in_p1 if ai==1 else in_p2}/10")


if __name__ == '__main__':
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\n  对局结束（退出）")
