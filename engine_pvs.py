"""PVS（主变窗搜索）+ 空着裁剪 迭代加深引擎——对标 clysto 的搜索结构。

与 AlphaBetaIDSEngine 的差异只有两处（均来自 clysto game.cpp 的做法）：
  1. PVS 主变窗：每节点第一个孩子全窗搜索，后续孩子先用单格窗口试探，
     只有真正可能改进时才回搜全窗。数学上与普通 alpha-beta 等价（结果不变，纯省节点）。
  2. 空着裁剪（NULL_MOVE_R=2）：让当前方轮空一手、再多搜 R 层，
     若结果仍越过窗口边界则剪掉整棵子树。跳棋中"轮到走"几乎总是有利
     （=竞速中的先手 tempo），该启发比国际象棋更可靠，clysto 同参数实测有效。

评估函数、走法排序（TT/killer/history）、迭代加深、超时中断、root 视角打分
全部复用 engine.py 原实现，不改动原类（A/B 可随时回退）。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import (AlphaBetaIDSEngine, _NEG_INF, _INF, _EXACT, _LOWER, _UPPER,
                    _zobrist_init, _zobrist_move, _ZOB_TURN,
                    order_moves, _record_cut)

import engine

cc = engine.cc


def _pvs_count(board, player, depth, root_player, alpha, beta, deadline, stats, eval_fn,
               tt, h, killer_table, history, ply, null_move_r):
    """PVS + 空着裁剪版的 _alphabeta_count（root 视角 min/max 框架，fail-soft）。

    与 _alphabeta_count 的差异：
      - 空着裁剪：进子树前先让当前方"轮空"探一下，越界即剪
      - PVS：非首个孩子先 (alpha,alpha+1)/(beta-1,beta) 单格窗试探，改进才回搜
      - TT flag 用入窗 alpha_orig/beta_orig 判断（原版用的是循环中被改过的 alpha）
      - 超时中断时不写 TT（原版会把中断值写进表）
    其余逻辑逐行对应原版。
    """
    stats[0] += 1
    if stats[2]:
        return eval_fn(board, root_player)

    # 置换表命中
    entry = tt.get(h)
    if entry is not None:
        _, e_score, e_flag, e_best = entry
        if entry[0] >= depth:
            if e_flag == _EXACT:
                return e_score
            elif e_flag == _LOWER and e_score >= beta:
                return e_score
            elif e_flag == _UPPER and e_score <= alpha:
                return e_score
            if alpha >= beta:
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

    # ===== 空着裁剪：当前方轮空，对手连走 R 手，值仍越界则剪 =====
    if null_move_r > 0 and depth - 1 - null_move_r > 0:
        if player == root_player:
            # 最大方轮空：探"对手连走 R 手后我方还能不能 ≥ beta"
            v = _pvs_count(board, 3 - player, depth - 1 - null_move_r, root_player,
                           beta - 1, beta, deadline, stats, eval_fn,
                           tt, h ^ _ZOB_TURN, killer_table, history, ply + 1, null_move_r)
        else:
            # 最小方轮空：探"我方白得 R 手后还能不能 ≤ alpha"
            v = _pvs_count(board, 3 - player, depth - 1 - null_move_r, root_player,
                           alpha, alpha + 1, deadline, stats, eval_fn,
                           tt, h ^ _ZOB_TURN, killer_table, history, ply + 1, null_move_r)
        if stats[2]:
            return eval_fn(board, root_player)
        if player == root_player and v >= beta:
            stats[1] += 1
            return v
        if player != root_player and v <= alpha:
            stats[1] += 1
            return v

    moves = cc.get_all_moves(board, player)
    if not moves:
        return eval_fn(board, root_player)

    moves = order_moves(moves, board, player, root_player, e_best, killer_table[ply], history)

    alpha_orig, beta_orig = alpha, beta

    if player == root_player:
        # ===== 最大节点：PVS 主变窗 =====
        value = _NEG_INF
        best_move = -1
        for i, (pos, path) in enumerate(moves):
            if stats[2]:
                break
            new_board = cc.apply_move(board, path)
            if i == 0:
                v = _pvs_count(new_board, 3 - player, depth - 1, root_player, alpha, beta,
                               deadline, stats, eval_fn, tt, _zobrist_move(h, path, player),
                               killer_table, history, ply + 1, null_move_r)
            else:
                v = _pvs_count(new_board, 3 - player, depth - 1, root_player, alpha, alpha + 1,
                               deadline, stats, eval_fn, tt, _zobrist_move(h, path, player),
                               killer_table, history, ply + 1, null_move_r)
                if stats[2]:
                    break
                if alpha < v < beta:      # 单格窗失败高发 → 真有可能改进，回搜全窗
                    v = _pvs_count(new_board, 3 - player, depth - 1, root_player, alpha, beta,
                                   deadline, stats, eval_fn, tt, _zobrist_move(h, path, player),
                                   killer_table, history, ply + 1, null_move_r)
                    if stats[2]:
                        break
            if v > value:
                value = v
                best_move = pos * 81 + path[-1]
                if value > alpha:
                    alpha = value
            if alpha >= beta:
                stats[1] += 1
                _record_cut(killer_table, ply, player, pos, path, history, depth)
                break
    else:
        # ===== 最小节点：PVS 主变窗（镜像） =====
        value = _INF
        best_move = -1
        for i, (pos, path) in enumerate(moves):
            if stats[2]:
                break
            new_board = cc.apply_move(board, path)
            if i == 0:
                v = _pvs_count(new_board, 3 - player, depth - 1, root_player, alpha, beta,
                               deadline, stats, eval_fn, tt, _zobrist_move(h, path, player),
                               killer_table, history, ply + 1, null_move_r)
            else:
                v = _pvs_count(new_board, 3 - player, depth - 1, root_player, beta - 1, beta,
                               deadline, stats, eval_fn, tt, _zobrist_move(h, path, player),
                               killer_table, history, ply + 1, null_move_r)
                if stats[2]:
                    break
                if alpha < v < beta:      # 单格窗失败低发 → 可能改进，回搜全窗
                    v = _pvs_count(new_board, 3 - player, depth - 1, root_player, alpha, beta,
                                   deadline, stats, eval_fn, tt, _zobrist_move(h, path, player),
                                   killer_table, history, ply + 1, null_move_r)
                    if stats[2]:
                        break
            if v < value:
                value = v
                best_move = pos * 81 + path[-1]
                if value < beta:
                    beta = value
            if alpha >= beta:
                stats[1] += 1
                _record_cut(killer_table, ply, player, pos, path, history, depth)
                break

    if stats[2]:                       # 超时中断：值不可信，不写 TT
        return eval_fn(board, root_player)

    if value <= alpha_orig:
        tt[h] = (depth, value, _UPPER, best_move)
    elif value >= beta_orig:
        tt[h] = (depth, value, _LOWER, best_move)
    else:
        tt[h] = (depth, value, _EXACT, best_move)
    return value


class PVSIDSEngine(AlphaBetaIDSEngine):
    """AlphaBetaIDSEngine + PVS + 空着裁剪（继承 think 骨架与 eval 版本选择，只换搜索内核）。"""

    def __init__(self, time_limit_sec=1.5, max_depth=8, eval_version="v2", null_move_r=2):
        super().__init__(time_limit_sec=time_limit_sec, max_depth=max_depth,
                         eval_version=eval_version)
        self.null_move_r = null_move_r
        self.name = self.name.replace("IDS", "PVS")
        if null_move_r != 2:
            self.name += f"-R{null_move_r}"

    def think(self, board, player):
        moves_root = cc.get_all_moves(board, player)
        if not moves_root:
            return None, {"reason": "no_moves"}

        deadline = time.time() + self.time_limit
        eval_fn = self._eval_fn
        null_r = self.null_move_r

        tt = {}
        root_hash = _zobrist_init(board, player)
        killer_table = [[-1, -1] for _ in range(2 * self.max_depth + 2)]
        history = {}

        best_path_overall = None
        best_score_overall = _NEG_INF
        depth_reached = 0
        total_nodes = 0
        total_prunes = 0
        aborted_overall = False

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

            for i, (pos, path) in enumerate(sorted_root_moves):
                if time.time() >= deadline:
                    stats[2] = True
                    break

                new_board = cc.apply_move(board, path)
                if i == 0:
                    score = _pvs_count(
                        new_board, 3 - player, depth - 1,
                        player, alpha, beta, deadline, stats, eval_fn,
                        tt, _zobrist_move(root_hash, path, player),
                        killer_table, history, 1, null_r)
                else:
                    score = _pvs_count(
                        new_board, 3 - player, depth - 1,
                        player, alpha, alpha + 1, deadline, stats, eval_fn,
                        tt, _zobrist_move(root_hash, path, player),
                        killer_table, history, 1, null_r)
                    if not stats[2] and alpha < score < beta:
                        score = _pvs_count(
                            new_board, 3 - player, depth - 1,
                            player, alpha, beta, deadline, stats, eval_fn,
                            tt, _zobrist_move(root_hash, path, player),
                            killer_table, history, 1, null_r)

                if not stats[2] and score > best_score_d:
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
