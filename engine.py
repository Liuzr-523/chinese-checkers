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
# 评估查表化：预计算 81 格位置分（每方视角一张表）
#   - 进度表 _PROG：P1 视角 = y 坐标（越大越靠 P2 目标区）；P2 视角 = 16-y（越小越靠 P1 目标区）
#   - 起步区表 _BACK：P1 在 0-9，P2 在 71-80（还在起点的棋子受后排惩罚）
# 让 evaluate 从"81 格大循环 + 二维索引 + 三角判断"变成"一次查表 + 简单加法"。
# ================================
_PROG = (
    None,
    tuple(cc.POS_TO_XY[pos][1] for pos in range(81)),
    tuple(16 - cc.POS_TO_XY[pos][1] for pos in range(81)),
)
_BACK = (
    None,
    tuple(pos < 10 for pos in range(81)),
    tuple(pos >= 71 for pos in range(81)),
)


# ================================
# 启发函数各版本（迭代验证用，每个版本互相独立）
# ================================

def evaluate_v0(board, player):
    """
    v0 基线版本：最落后棋子进度 * 100 + 总进度
      - v1 纯总和 → 只推前排，后排卡死，80%平局
      - v2 纯最落后 → 区分度太低，全平局
      - v0 加权组合 → 主分推后排，次分保证有区分度
    查表版：与旧实现数值完全等价（同一张 _PROG 表，一次遍历）。
    """
    prog = _PROG[player]
    m = 17            # 进度最大 16，17 表示该方已无棋子
    s = 0
    for pos in range(81):
        if board[pos] == player:
            v = prog[pos]
            if v < m:
                m = v
            s += v
    if m == 17:
        return 0
    return m * 100 + s


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
    查表版：合并为一次 81 格循环，与旧实现数值完全等价（同一张 _PROG/_BACK 表）。
    """
    opp = 3 - player
    prog = _PROG[player]
    opp_prog = _PROG[opp]
    back_zone = _BACK[player]

    my_min = 17
    my_sum = 0
    back = 0
    op_min = 17
    op_sum = 0
    for pos in range(81):
        piece = board[pos]
        if piece:
            if piece == player:
                v = prog[pos]
                if v < my_min:
                    my_min = v
                my_sum += v
                if back_zone[pos]:
                    back += 1
            else:
                v = opp_prog[pos]
                if v < op_min:
                    op_min = v
                op_sum += v

    if my_min == 17:           # 我方已无棋子
        score = 0
    else:
        score = my_min * 100 + my_sum - back * 3
    if op_min < 17:            # 对手还有棋子
        score -= (op_min * 100 + op_sum) * 0.05

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
    moves = order_moves(moves, board, player, root_player)

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

def order_moves(moves, board, player, root_player, tt_best=-1, killers=None, history=None):
    """
    给一个节点的所有子走法排序：「看起来更好」的走法放在前面先搜。
    作用：让 alpha/beta 尽早收紧，大幅提高剪枝率。

    排序优先级（IDS 系，带 TT/killer/history）：
      1. TT 最优走法（上一轮搜索证明最好的那步）→ 最高优先
      2. Killer 走法（本层刚剪过枝的强着）→ 次高
      3. History 分数（累计的剪枝经验，越晚层越可信）+ 连跳奖励
    均为查表，零 apply_move，是"隐性浪费"的根治方案。

    回退分支：MinimaxEngine 等无辅助信息的调用点保持旧的「完整评估排序」行为。
    """
    if history is None and tt_best < 0 and killers is None:
        scored = []
        for pos, path in moves:
            new_board = cc.apply_move(board, path)
            s = evaluate_v2(new_board, root_player)
            if len(path) >= 3:
                s += 1000
            scored.append((s, pos, path))
        scored.sort(reverse=True)
        return [(pos, path) for (_, pos, path) in scored]

    scored = []
    for pos, path in moves:
        code = pos * 81 + path[-1]
        if code == tt_best:
            prio = 30
        elif killers is not None and code in killers:
            prio = 20
        else:
            prio = 10
        s = history.get((player, code), 0) if history else 0
        if len(path) >= 3:
            s += 1000
        scored.append((prio, s, pos, path))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [(pos, path) for (_, _, pos, path) in scored]


# ========================================
# 关卡 4.2：AlphaBetaIDSEngine（迭代加深搜索 + 限时）
# ========================================

# ---- Zobrist 哈希（置换表用）----
# 每个 (位置, 棋子) 一个 64 位随机数 + 一个轮次翻转位。
# 局面哈希 = 各子上随机数的异或（可增量更新：走一步只改起终点 2 个格子 + 翻转轮次位）。
_ZOB = [[random.getrandbits(64) for _ in range(3)] for _ in range(81)]
_ZOB_TURN = random.getrandbits(64)


def _zobrist_init(board, player):
    """初始局面哈希：P1 轮走不带轮次位，P2 轮走带轮次位"""
    h = 0
    for pos in range(81):
        pc = board[pos]
        if pc:
            h ^= _ZOB[pos][pc]
    if player == 2:
        h ^= _ZOB_TURN
    return h


def _zobrist_move(h, path, player):
    """走一步后的增量哈希：起点拿走、终点放入（中间被跳的棋子不动），轮次翻转"""
    h ^= _ZOB[path[0]][player]
    h ^= _ZOB[path[-1]][player]
    return h ^ _ZOB_TURN


# 置换表条目 flag：值是精确值 / 下界(fail-high) / 上界(fail-low)
_EXACT, _LOWER, _UPPER = 0, 1, 2


def _record_cut(killer_table, ply, player, pos, path, history, depth):
    """记录"剪枝走法"：killer 两槽（本层强着）+ history 计数（全局累计，越深权重越大）。"""
    code = pos * 81 + path[-1]
    ks = killer_table[ply]
    if ks[0] != code:
        ks[1] = ks[0]
        ks[0] = code
    key = (player, code)
    history[key] = history.get(key, 0) + depth * depth

def _alphabeta_count(board, player, depth, root_player, alpha, beta, deadline, stats, eval_fn,
                     tt, h, killer_table, history, ply):
    """
    带「节点计数 + 超时中断 + 可切换 eval_fn + Zobrist 置换表 + Killer/History」的 alphabeta。
    stats = [nodes, prunes, aborted]
    eval_fn = evaluate_v0 / evaluate_v1 / evaluate ... 传进来用哪个打分用哪个
    tt     = 置换表 dict：hash -> (depth, score, flag, best_move)。一次 think 内 root 不变，
             表格里所有分数都是 root_player 视角，可安全复用；拆表省 30~70% 节点。
    h      = 当前局面的 Zobrist 哈希（增量 xorb，避免重算 81 格）
    killer_table = 按层(ply)的杀手锏两槽表，history = 全局剪枝经验计数
    ply    = 距根部的层数（0 起），Killer 表按它索引
    """
    stats[0] += 1
    if stats[2]:
        return eval_fn(board, root_player)

    # 置换表命中：同局面重复出现且存的深度足够，直接复用搜索结果（省整棵子树）
    entry = tt.get(h)
    if entry is not None:
        _, e_score, e_flag, e_best = entry
        if entry[0] >= depth:
            if e_flag == _EXACT:
                return e_score
            if e_flag == _LOWER and e_score >= beta:
                return e_score
            if e_flag == _UPPER and e_score <= alpha:
                return e_score
    else:
        e_best = -1

    if deadline is not None and time.time() >= deadline:
        stats[2] = True
        return eval_fn(board, root_player)

    winner = cc.is_game_over(board)
    if winner is not None:
        return 10_000_000 if winner == root_player else -10_000_000
    if depth <= 0:
        return eval_fn(board, root_player)

    moves = cc.get_all_moves(board, player)
    if not moves:
        return eval_fn(board, root_player)

    moves = order_moves(moves, board, player, root_player, e_best, killer_table[ply], history)

    if player == root_player:
        value = _NEG_INF
        best_move = -1
        for pos, path in moves:
            new_board = cc.apply_move(board, path)
            v = _alphabeta_count(new_board, 3 - player, depth - 1, root_player, alpha, beta,
                                 deadline, stats, eval_fn, tt, _zobrist_move(h, path, player),
                                 killer_table, history, ply + 1)
            if v > value:
                value = v
                best_move = pos * 81 + path[-1]
            alpha = max(alpha, value)
            if alpha >= beta:
                stats[1] += 1
                _record_cut(killer_table, ply, player, pos, path, history, depth)
                break
    else:
        value = _INF
        best_move = -1
        for pos, path in moves:
            new_board = cc.apply_move(board, path)
            v = _alphabeta_count(new_board, 3 - player, depth - 1, root_player, alpha, beta,
                                 deadline, stats, eval_fn, tt, _zobrist_move(h, path, player),
                                 killer_table, history, ply + 1)
            if v < value:
                value = v
                best_move = pos * 81 + path[-1]
            beta = min(beta, value)
            if alpha >= beta:
                stats[1] += 1
                _record_cut(killer_table, ply, player, pos, path, history, depth)
                break

    # 存储：剪枝中断的返回值只是边界（非精确），按 flag 区分，查询时只作边界剪枝用
    if value <= alpha:
        tt[h] = (depth, value, _UPPER, best_move)
    elif value >= beta:
        tt[h] = (depth, value, _LOWER, best_move)
    else:
        tt[h] = (depth, value, _EXACT, best_move)
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

        # 置换表：一次 think 内 root 不变（表里分数都是 root 视角），迭代加深各层共享复用
        tt = {}
        root_hash = _zobrist_init(board, player)
        # killer 表按 ply 索引（深度 2 倍足够），history 全局累计剪枝经验
        killer_table = [[-1, -1] for _ in range(2 * self.max_depth + 2)]
        history = {}

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

            root_entry = tt.get(root_hash)
            root_best = root_entry[3] if root_entry is not None else -1
            sorted_root_moves = order_moves(moves_root, board, player, player,
                                            root_best, killer_table[0], history)

            for pos, path in sorted_root_moves:
                if time.time() >= deadline:
                    stats[2] = True
                    break

                new_board = cc.apply_move(board, path)

                score = _alphabeta_count(
                    new_board, 3 - player, depth - 1,
                    player, alpha, beta, deadline, stats, eval_fn,
                    tt, _zobrist_move(root_hash, path, player),
                    killer_table, history, 1
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
