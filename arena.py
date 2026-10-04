"""
自对弈擂台：让两个 Engine 互相对战，记录棋谱，统计胜率。

对应 Katago 的：
  - self-play（自对弈产出训练数据）
  - rating match（新旧模型对战胜率对比）
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
from engine import Engine, RandomEngine, GreedyEngine, MinimaxEngine, AlphaBetaEngine, AlphaBetaIDSEngine

cc = engine.cc  # 规则模块，从 engine.py 复用，不重复加载


class GameRecord:
    """一局对战的完整记录（将来的训练数据雏形）"""

    def __init__(self):
        self.moves = []      # [(player, path), ...] 每一步谁走了哪条路
        self.winner = None    # 1, 2, 或 None（超步数平局）
        self.boards = []      # 每步走之前的棋盘快照

    def __repr__(self):
        return f"GameRecord(winner={self.winner}, steps={len(self.moves)})"


def play_game(engine_a, engine_b, max_steps=500):
    """
    跑一局对战。
    engine_a 执 P1（先手），engine_b 执 P2。
    返回 GameRecord。
    """
    board = cc.init_board()
    record = GameRecord()
    record.boards.append(board.copy())

    engines = {1: engine_a, 2: engine_b}

    for step in range(max_steps):
        player = 1 if step % 2 == 0 else 2
        engine = engines[player]

        path, info = engine.think(board, player)
        if path is None:
            # 无棋可走，判对方胜
            record.winner = 2 if player == 1 else 1
            break

        record.moves.append((player, path))
        board = cc.apply_move(board, path)
        record.boards.append(board.copy())

        winner = cc.is_game_over(board)
        if winner:
            record.winner = winner
            break
    else:
        record.winner = None  # 超步数，平局

    return record


def calc_gap(final_board, winner, winner_engine, max_steps=300):
    """
    赛后补算差距值：输家停手，赢家独自继续走，
    直到 10 颗棋子全部进入目标区。
    返回赢家额外需要的步数（越小说明双方差距越小）。
    """
    board = final_board.copy()
    gap_steps = 0

    if winner == 1:
        target_zone = set(range(71, 81))
    else:
        target_zone = set(range(0, 10))

    for _ in range(max_steps):
        path, info = winner_engine.think(board, winner)
        if path is None:
            break
        board = cc.apply_move(board, path)
        gap_steps += 1

        in_target = 0
        for pos in range(81):
            if board[pos] == winner and pos in target_zone:
                in_target += 1
        if in_target == 10:
            break

    return gap_steps


def fight(engine_a, engine_b, n_games=10, max_steps=500):
    """
    跑 n_games 局，双方轮流先手（保证公平）。
    返回统计结果字典。
    """
    results = {"a_wins": 0, "b_wins": 0, "draws": 0, "games": [], "gaps": []}

    for i in range(n_games):
        # 偶数局 A 先手，奇数局 B 先手
        if i % 2 == 0:
            first, second = engine_a, engine_b
        else:
            first, second = engine_b, engine_a

        record = play_game(first, second, max_steps)
        results["games"].append(record)

        # 判断谁赢：先手=1，后手=2
        if i % 2 == 0:
            a_player, b_player = 1, 2
        else:
            a_player, b_player = 2, 1

        gap = None
        if record.winner == a_player:
            results["a_wins"] += 1
            gap = calc_gap(record.boards[-1], record.winner, engine_a)
        elif record.winner == b_player:
            results["b_wins"] += 1
            gap = calc_gap(record.boards[-1], record.winner, engine_b)
        else:
            results["draws"] += 1

        results["gaps"].append(gap)

        winner_str = f"P{record.winner}" if record.winner else "平局"
        gap_str = f"，差距 {gap} 步" if gap is not None else ""
        print(f"  第 {i+1}/{n_games} 局：{len(record.moves)} 步 → {winner_str}{gap_str}")

    return results


def print_result(engine_a, engine_b, results):
    """打印对战结果"""
    total = results["a_wins"] + results["b_wins"] + results["draws"]
    print(f"\n{'='*50}")
    print(f"对战结果：{engine_a.name} vs {engine_b.name}  ({total} 局)")
    print(f"{'='*50}")
    print(f"  {engine_a.name} 胜：{results['a_wins']}  ({results['a_wins']/total*100:.0f}%)")
    print(f"  {engine_b.name} 胜：{results['b_wins']}  ({results['b_wins']/total*100:.0f}%)")
    print(f"  平局：{results['draws']}  ({results['draws']/total*100:.0f}%)")

    # 每局详情 + 步数统计
    all_steps = []
    a_steps = []
    b_steps = []
    for i, record in enumerate(results["games"]):
        winner_str = f"P{record.winner}" if record.winner else "平局"
        steps = len(record.moves)
        all_steps.append(steps)

        # 判断这局谁赢，记录胜者用了几步
        if i % 2 == 0:
            a_player, b_player = 1, 2
        else:
            a_player, b_player = 2, 1

        if record.winner == a_player:
            a_steps.append(steps)
        elif record.winner == b_player:
            b_steps.append(steps)

        print(f"  第{i+1}局：{steps}步 → {winner_str}")

    # 步数效率统计
    print(f"\n步数效率统计：")
    avg_all = sum(all_steps) / len(all_steps) if all_steps else 0
    print(f"  全部对局平均步数：{avg_all:.0f} 步")
    if a_steps:
        print(f"  {engine_a.name} 胜局平均步数：{sum(a_steps)/len(a_steps):.0f} 步（{len(a_steps)} 局）")
    if b_steps:
        print(f"  {engine_b.name} 胜局平均步数：{sum(b_steps)/len(b_steps):.0f} 步（{len(b_steps)} 局）")
    if all_steps:
        print(f"  最短一局：{min(all_steps)} 步 / 最长一局：{max(all_steps)} 步")

    # 差距值统计（赛后补算：输家停手，赢家还需几步全归位）
    gaps = results.get("gaps", [])
    valid_gaps = [g for g in gaps if g is not None]
    if valid_gaps:
        print(f"\n差距值统计（赢家补走步数，越小=双方越接近）：")
        print(f"  平均差距：{sum(valid_gaps)/len(valid_gaps):.0f} 步")
        print(f"  最小差距：{min(valid_gaps)} 步 / 最大差距：{max(valid_gaps)} 步")

    # 棋子数守恒校验（防止规则 bug 导致棋子丢失）
    print(f"\n棋子数守恒校验：")
    all_ok = True
    for i, record in enumerate(results["games"]):
        for j, board in enumerate(record.boards):
            p1 = sum(1 for p in board if p == 1)
            p2 = sum(1 for p in board if p == 2)
            if p1 != 10 or p2 != 10:
                print(f"  ❌ 第{i+1}局第{j}步：P1={p1} P2={p2}")
                all_ok = False
    if all_ok:
        print(f"  ✅ 全部 {total} 局全程棋子数 = 20（没吃子）")


if __name__ == "__main__":
    # 关卡 5：逐个特征验证 — 当前测试：后排(W=3) + 对手(W=0.05) + 终局加速(W=1)
    engine_a = AlphaBetaIDSEngine(time_limit_sec=1.2, max_depth=8, eval_version="v2")
    engine_b = AlphaBetaIDSEngine(time_limit_sec=1.2, max_depth=8, eval_version="v0")
    results = fight(engine_a, engine_b, n_games=10, max_steps=400)
    print_result(engine_a, engine_b, results)
