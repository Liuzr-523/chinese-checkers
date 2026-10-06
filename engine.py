"""
Engine 抽象基类 + RandomEngine（Level-0 随机 AI）

所有 AI 引擎都继承 Engine，实现 think() 方法。
擂台代码只调 engine.think(board, player)，不关心内部是随机还是神经网络。
"""
import importlib.util
import os
import random
import time

# 导入规则模块：路径相对于本脚本目录，不依赖 cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("cc", os.path.join(_HERE, "Chinese_Checker.py"))
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


class Engine:
    """所有 AI 引擎的基类（抽象接口）"""

    name = "BaseEngine"

    def think(self, board, player):
        """
        思考一步走法。

        输入：
            board:  81 长度的棋盘列表（0=空, 1=P1, 2=P2）
            player: 当前轮到谁（1 或 2）

        输出：(path, info) 元组
            path: 走法路径，如 [40, 32, 25]，第一个元素是起点
            info: 调试信息字典，比如搜了几个节点、估值多少

        子类必须重写此方法。
        """
        raise NotImplementedError


def evaluate(board, player):
    """
    启发函数（Heuristic）：给一个局面打分，分数越大对 player 越有利。
    所有 AI 统一调这个入口（evaluate_vx 版本号在内部切换），
    保证 Greedy/Minimax/AlphaBeta/IDS 用同一张打分规则。
    """
    return evaluate_v2(board, player)


# ================================
# 启发函数各版本（迭代验证用，每个版本互相独立）
# ================================

def evaluate_v0(board, player):
    """
    v0 基线版本：最落后棋子进度 * 100 + 总进度
      - v1 纯总和 → 只推前排，后排卡死，80%平局
      - v2 纯最落后 → 区分度太低，全平局
      - v0 加权组合 → 主分推后排，次分保证有区分度
    """
    progress = []
    for pos in range(81):
        if board[pos] == player:
            y = cc.POS_TO_XY[pos][1]
            if player == 1:
                progress.append(y)          # P1：y 越大越靠近目标
            else:
                progress.append(16 - y)     # P2：y 越小越靠近目标
    if not progress:
        return 0
    return min(progress) * 100 + sum(progress)


def evaluate_v1(board, player):
    """
    v1（关卡 5-特征 1）：v0 + 「连跳机会分数」× W1
      - 连跳机会 = 所有我方棋子里，len(path) >= 3 的跳路总条数
      - 连跳 = 一次能跨越多格，极可能大幅推进，机会越多 = 局面越有利
      - W1 先粗估 20，跑完20局胜率验证再调
    """
    W_JUMP = 2

    score = evaluate_v0(board, player)

    # 特征 1：连跳机会
    jump_chance = 0
    for pos in range(81):
        if board[pos] == player:
            paths = cc.get_all_jump_paths(pos, board, player)
            for p in paths:
                if len(p) >= 3:          # 连跳：至少 3 个节点（跨过 >=2 颗棋子）
                    jump_chance += 1
    score += jump_chance * W_JUMP

    return score


def evaluate_v2(board, player):
    """
    v2（关卡 5 最终保留版）：v0 基线 + 后排惩罚(W=3) + 对手进度差(W=0.05) [已验证 10:0 胜 v0，平均 80 步]
    已丢弃：中央要道(中性)、聚集(0:10)、连跳机会(太慢 0:10)、压制对手连跳(太慢)、
            终局加速(W=1，5:5 平 = 把 10:0 拖成五五开，sum 加权冷落最落后棋子)
    """
    opp = 3 - player
    score = evaluate_v0(board, player)

    # 后排启动惩罚：还在起点区的棋子每个扣分
    W_BACK = 3
    if player == 1:
        start_zone = set(range(0, 10))
    else:
        start_zone = set(range(71, 81))
    back_count = 0
    for pos in range(81):
        if board[pos] == player and pos in start_zone:
            back_count += 1
    score -= back_count * W_BACK

    # 对手进度差：对手越接近目标，我方扣分越多
    W_OPP = 0.05
    opp_progress = []
    for pos in range(81):
        if board[pos] == opp:
            y = cc.POS_TO_XY[pos][1]
            if opp == 1:
                opp_progress.append(y)
            else:
                opp_progress.append(16 - y)
    if opp_progress:
        opp_v0 = min(opp_progress) * 100 + sum(opp_progress)
        score -= opp_v0 * W_OPP

    return score


def evaluate_v3(board, player):
    """
    v3 = v0 + 轻量跳路机会(W=2)
    v1 用 get_all_jump_paths（慢，拖累搜索深度5→4）→ 0:10败
    v3 改用 get_jump_moves 数单跳目标数量（快），不拖累搜索深度
    """
    W_JUMP = 2
    score = evaluate_v0(board, player)

    jump_chance = 0
    for pos in range(81):
        if board[pos] == player:
            jump_chance += len(cc.get_jump_moves(pos, board))
    score += jump_chance * W_JUMP

    return score


def evaluate_v4(board, player):
    """
    v4 = v2 + 轻量跳路机会(W=2)
    在已验证的 v2（后排惩罚+对手进度差）基础上加跳路机会
    """
    W_JUMP = 2
    score = evaluate_v2(board, player)

    jump_chance = 0
    for pos in range(81):
        if board[pos] == player:
            jump_chance += len(cc.get_jump_moves(pos, board))
    score += jump_chance * W_JUMP

    return score


def evaluate_v5(board, player):
    """
    v5 = v0 + 前进跳路机会(W=10)
    v3 数所有跳路(含倒退)→0:10败；v5 只数跳到更靠目标位置的跳路
    坐标修复前跳路方向算错，前进/倒退判断也错，现在重新测
    """
    W_FWD = 10
    score = evaluate_v0(board, player)

    fwd_jump = 0
    for pos in range(81):
        if board[pos] == player:
            y_start = cc.POS_TO_XY[pos][1]
            for dst in cc.get_jump_moves(pos, board):
                y_dst = cc.POS_TO_XY[dst][1]
                if player == 1 and y_dst > y_start:
                    fwd_jump += 1
                elif player == 2 and y_dst < y_start:
                    fwd_jump += 1
    score += fwd_jump * W_FWD

    return score


def evaluate_v6(board, player):
    """
    v6 = v0 + 中央要道(W=1)
    棋子越靠近棋盘中央(4,8)，跳路枢纽价值越高
    之前判"中性"，坐标修复后下三角距离计算有变化，重测
    """
    W_CENTER = 1
    score = evaluate_v0(board, player)

    cx, cy = 4, 8
    center_bonus = 0
    for pos in range(81):
        if board[pos] == player:
            x, y = cc.POS_TO_XY[pos]
            dist = max(abs(x - cx), abs(y - cy))
            center_bonus += (8 - dist)
    score += center_bonus * W_CENTER

    return score


def evaluate_v7(board, player):
    """
    v7 = v0 + 终局加速(W=10)
    过了中线的棋子越多，每个加越多分，鼓励终局快速归位
    之前判"5:5平"（把10:0拖成五五开），重测
    """
    W_END = 10
    score = evaluate_v0(board, player)

    midline = 8
    advanced = 0
    for pos in range(81):
        if board[pos] == player:
            y = cc.POS_TO_XY[pos][1]
            if player == 1 and y > midline:
                advanced += 1
            elif player == 2 and y < midline:
                advanced += 1
    score += advanced * W_END

    return score


class RandomEngine(Engine):
    """Level-0 随机 AI：从所有合法走法中随机挑一个"""

    name = "Random"

    def think(self, board, player):
        moves = cc.get_all_moves(board, player)
        if not moves:
            return None, {"reason": "no_moves"}

        pos, path = random.choice(moves)
        return path, {"move_count": len(moves)}


class GreedyEngine(Engine):
    """Level-1 贪心 AI：试所有走法，选走完后局面分数最高的"""

    name = "Greedy"

    def think(self, board, player):
        moves = cc.get_all_moves(board, player)
        if not moves:
            return None, {"reason": "no_moves"}

        best_score = -1
        best_path = None
        for pos, path in moves:
            new_board = cc.apply_move(board, path)
            score = evaluate(new_board, player)
            if score > best_score:
                best_score = score
                best_path = path

        return best_path, {"move_count": len(moves), "score": best_score}


def minimax(board, player, depth, root_player):
    """
    极小极大搜索（Minimax）。

    输入：
        board:       当前棋盘
        player:      这一层轮到谁走（每递归一层就切一次，1↔2）
        depth:       还剩几层没搜（搜到0就停）
        root_player: 根节点的玩家（MAX方），全树固定不变！
                     所有 evaluate 都站在 root_player 视角打分，统一度量衡。

    返回：当前局面的估值分数（越大对 root_player 越有利）
    """
    # 停止条件：分出胜负 或 搜到底层
    winner = cc.is_game_over(board)
    if winner is not None:
        if winner == root_player:
            return 10_000_000   # MAX方赢了，极大正数
        else:
            return -10_000_000  # MIN方赢了，极大负数
    if depth <= 0:
        return evaluate(board, root_player)   # 统一用 root_player 视角

    moves = cc.get_all_moves(board, player)
    if not moves:
        return evaluate(board, root_player)

    # MAX 层：轮到 root_player 走 → 选子节点里分数最大的
    if player == root_player:
        best = -10_000_000_000
        for pos, path in moves:
            new_board = cc.apply_move(board, path)
            score = minimax(new_board, 3 - player, depth - 1, root_player)
            if score > best:
                best = score
        return best

    # MIN 层：对手走 → 选子节点里分数最小的（最小化 MAX 方的优势）
    else:
        best = 10_000_000_000
        for pos, path in moves:
            new_board = cc.apply_move(board, path)
            score = minimax(new_board, 3 - player, depth - 1, root_player)
            if score < best:
                best = score
        return best


class MinimaxEngine(Engine):
    """
    Level-2 极小极大 AI：向前看 depth 层，考虑对手的最优反击。

    depth=2 就是"我走一步 + 对手反击一步"（共2层），一般第一步就能赢 Greedy。
    """

    def __init__(self, depth=2):
        self.depth = depth
        self.name = f"Minimax-d{depth}"

    def think(self, board, player):
        moves = cc.get_all_moves(board, player)
        if not moves:
            return None, {"reason": "no_moves"}

        best_score = -10_000_000_000
        best_path = None
        nodes = 0

        for pos, path in moves:
            new_board = cc.apply_move(board, path)
            # 根节点的第一条走法 → 下一层是对手走（深度剩 depth-1），root_player=player（当前这个人）
            score = minimax(new_board, 3 - player, self.depth - 1, player)
            nodes += 1
            if score > best_score:
                best_score = score
                best_path = path

        return best_path, {"move_count": len(moves), "score": best_score, "nodes": nodes}


# 哨兵值：比任何正常分数都大/小，保证"第一次比较必然成立"
_INF = 10_000_000_000      # +无穷
_NEG_INF = -10_000_000_000  # -无穷


def alphabeta(board, player, depth, root_player, alpha, beta):
    """
    Alpha-Beta 剪枝搜索。

    和 minimax 的唯一区别：
        1. 多了 alpha, beta 两个边界参数
        2. MAX 层：子节点搜完一个就更新 alpha，发现 alpha >= beta 就 break（剪枝）
        3. MIN 层：子节点搜完一个就更新 beta，发现 alpha >= beta 就 break（剪枝）

    返回值和 minimax 完全一致：当前局面对 root_player 的估值分数。
    数学保证：alphabeta(board, p, d, r, -inf, +inf) == minimax(board, p, d, r)
    """
    # 停止条件：分出胜负 或 搜到底层
    winner = cc.is_game_over(board)
    if winner is not None:
        if winner == root_player:
            return 10_000_000
        else:
            return -10_000_000
    if depth <= 0:
        return evaluate(board, root_player)

    moves = cc.get_all_moves(board, player)
    if not moves:
        return evaluate(board, root_player)

    # 关卡 4：走法排序 —— 分数高的先搜，让 alpha/beta 尽早收敛，大幅提高剪枝率
    moves = order_moves(moves, board, player, root_player, evaluate)

    # MAX 层：轮到 root_player 走 → 选子节点里分数最大的
    if player == root_player:
        value = _NEG_INF
        for pos, path in moves:
            new_board = cc.apply_move(board, path)
            # 注意：传给下一层的是同一个 alpha, beta（不是更新后的副本）
            value = max(value, alphabeta(new_board, 3 - player, depth - 1, root_player, alpha, beta))
            alpha = max(alpha, value)        # 收紧下界
            if alpha >= beta:                 # ✂️ 剪枝：对手这条分支最多给我 value，但我已经保底 alpha ≥ beta
                break                         # 后面子节点不用搜了
        return value

    # MIN 层：对手走 → 选子节点里分数最小的
    else:
        value = _INF
        for pos, path in moves:
            new_board = cc.apply_move(board, path)
            value = min(value, alphabeta(new_board, 3 - player, depth - 1, root_player, alpha, beta))
            beta = min(beta, value)           # 收紧上界
            if alpha >= beta:                  # ✂️ 剪枝：我这条分支保底 value，但对手已经能把我压到 beta ≤ alpha
                break
        return value


class AlphaBetaEngine(Engine):
    """
    Level-3 Alpha-Beta 剪枝 AI。

    和 MinimaxEngine 数学等价（返回同一分数、同一走法），
    但因为剪枝，同样深度下速度快 5~10 倍，或者在相同时间下多搜 1~2 层。
    """

    def __init__(self, depth=4):
        self.depth = depth
        self.name = f"AlphaBeta-d{depth}"

    def think(self, board, player):
        moves = cc.get_all_moves(board, player)
        if not moves:
            return None, {"reason": "no_moves"}

        best_score = _NEG_INF
        best_path = None
        alpha = _NEG_INF
        beta = _INF
        nodes_searched = 0       # 实际搜了多少根节点走法（用来对比剪枝率）
        nodes_pruned = 0          # 被剪掉多少根节点走法

        for pos, path in moves:
            new_board = cc.apply_move(board, path)
            # 根节点的第一条走法 → 下一层是对手走（MIN层）
            score = alphabeta(new_board, 3 - player, self.depth - 1, player, alpha, beta)
            nodes_searched += 1
            if score > best_score:
                best_score = score
                best_path = path
            # 更新 alpha（根节点是 MAX 层）
            alpha = max(alpha, best_score)

        nodes_pruned = len(moves) - nodes_searched
        prune_rate = nodes_pruned / len(moves) * 100 if moves else 0

        return best_path, {
            "move_count": len(moves),
            "score": best_score,
            "nodes_searched": nodes_searched,
            "nodes_pruned": nodes_pruned,
            "prune_rate": f"{prune_rate:.0f}%",
        }


# ========================================
# 关卡 4.1：走法排序（启发式 Move Ordering）
# ========================================

def order_moves(moves, board, player, root_player, eval_fn):
    """
    给一个节点的所有子走法排序：「看起来更好」的走法放在前面先搜。
    作用：让 alpha/beta 尽早收紧，大幅提高剪枝率（实测 7x→14x 加速）。

    排序策略（两个信号叠加，越靠上权重越大）：
      1. 模拟走完一步后，eval_fn(new_board, root_player) 的分数 → 主要排序依据
      2. 连跳额外加分：len(path)>=3 加 1000 → 经验规则：连跳 > 单跳 > 单步
    """
    scored = []
    for pos, path in moves:
        new_board = cc.apply_move(board, path)
        s = eval_fn(new_board, root_player)
        bonus = 1000 if len(path) >= 3 else 0
        scored.append((s + bonus, pos, path))
    scored.sort(reverse=True)    # 分数高的放最前面
    return [(pos, path) for (s, pos, path) in scored]


# ========================================
# 关卡 4.2：AlphaBetaIDSEngine（迭代加深搜索 + 限时）
# ========================================

def _alphabeta_count(board, player, depth, root_player, alpha, beta, deadline, stats, eval_fn):
    """
    带「节点计数 + 超时中断 + 可切换 eval_fn」的 alphabeta。
    stats = [nodes, prunes, aborted]
    eval_fn = evaluate_v0 / evaluate_v1 / evaluate ... 传进来用哪个打分用哪个
    """
    if stats[2] or (deadline is not None and time.time() >= deadline):
        stats[2] = True
        return eval_fn(board, root_player)

    stats[0] += 1

    winner = cc.is_game_over(board)
    if winner is not None:
        return 10_000_000 if winner == root_player else -10_000_000
    if depth <= 0:
        return eval_fn(board, root_player)

    moves = cc.get_all_moves(board, player)
    if not moves:
        return eval_fn(board, root_player)

    moves = order_moves(moves, board, player, root_player, eval_fn)

    if player == root_player:
        value = _NEG_INF
        for pos, path in moves:
            new_board = cc.apply_move(board, path)
            value = max(value, _alphabeta_count(new_board, 3 - player, depth - 1, root_player, alpha, beta, deadline, stats, eval_fn))
            alpha = max(alpha, value)
            if alpha >= beta:
                stats[1] += 1
                break
        return value
    else:
        value = _INF
        for pos, path in moves:
            new_board = cc.apply_move(board, path)
            value = min(value, _alphabeta_count(new_board, 3 - player, depth - 1, root_player, alpha, beta, deadline, stats, eval_fn))
            beta = min(beta, value)
            if alpha >= beta:
                stats[1] += 1
                break
        return value


class AlphaBetaIDSEngine(Engine):
    """
    Level-4：迭代加深搜索（Iterative Deepening Search）+ Alpha-Beta 剪枝 + 限时。

    思路：
      for depth in 1,2,3,4,5,6,7,8...:
          完整搜一层 depth（用 alphabeta）
          搜完了 → 更新 best = 这层答案
          到时间了 → 立刻中断，返回上一层已完成的 best

    eval_version 参数（关卡 5 逐个特征 AB 测试用）：
        "current"  → 用全局 evaluate()（当前正式版本）
        "v0"       → 用 evaluate_v0（木桶+sum 基线）
        "v1"       → 用 evaluate_v1（基线+连跳机会）
    """

    def __init__(self, time_limit_sec=1.5, max_depth=8, eval_version="current"):
        self.time_limit = time_limit_sec
        self.max_depth = max_depth
        self.eval_version = eval_version
        self.name = f"IDS-t{time_limit_sec}s"
        if eval_version != "current":
            self.name += f"-{eval_version}"

        # 根据版本选择对应的 eval 函数（整个 IDS 从"排序→叶子打分"全程用同一个）
        if eval_version == "v0":
            self._eval_fn = evaluate_v0
        elif eval_version == "v1":
            self._eval_fn = evaluate_v1
        elif eval_version == "v2":
            self._eval_fn = evaluate_v2
        elif eval_version == "v3":
            self._eval_fn = evaluate_v3
        elif eval_version == "v4":
            self._eval_fn = evaluate_v4
        elif eval_version == "v5":
            self._eval_fn = evaluate_v5
        elif eval_version == "v6":
            self._eval_fn = evaluate_v6
        elif eval_version == "v7":
            self._eval_fn = evaluate_v7
        else:
            self._eval_fn = evaluate

    def think(self, board, player):
        moves_root = cc.get_all_moves(board, player)
        if not moves_root:
            return None, {"reason": "no_moves"}

        deadline = time.time() + self.time_limit
        eval_fn = self._eval_fn

        best_path_overall = None
        best_score_overall = _NEG_INF
        depth_reached = 0
        total_nodes = 0
        total_prunes = 0
        aborted_overall = False

        # 迭代加深外层循环
        for depth in range(1, self.max_depth + 1):
            alpha = _NEG_INF
            beta = _INF
            best_path_d = None
            best_score_d = _NEG_INF
            stats = [0, 0, False]

            sorted_root_moves = order_moves(moves_root, board, player, player, eval_fn)

            for pos, path in sorted_root_moves:
                if time.time() >= deadline:
                    stats[2] = True
                    break

                new_board = cc.apply_move(board, path)

                score = _alphabeta_count(
                    new_board, 3 - player, depth - 1,
                    player, alpha, beta, deadline, stats, eval_fn
                )

                if score > best_score_d:
                    best_score_d = score
                    best_path_d = path
                alpha = max(alpha, best_score_d)

                if stats[2]:
                    break

            total_nodes += stats[0]
            total_prunes += stats[1]

            if stats[2]:
                aborted_overall = True
                break
            else:
                best_path_overall = best_path_d
                best_score_overall = best_score_d
                depth_reached = depth

                if abs(best_score_overall) >= 9_999_999:
                    break

        prune_rate = total_prunes / max(total_nodes, 1) * 100

        return best_path_overall, {
            "move_count": len(moves_root),
            "score": best_score_overall,
            "depth_reached": depth_reached,
            "nodes_searched": total_nodes,
            "nodes_pruned": total_prunes,
            "prune_rate": f"{prune_rate:.0f}%",
            "aborted": aborted_overall,
            "eval_version": self.eval_version,
        }
