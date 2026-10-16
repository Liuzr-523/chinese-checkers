"""NN 走法排序的 AlphaBetaIDS 引擎：V2 叶子评估 + NN policy 根结点排序。

思路：alpha-beta 的剪枝效率极度依赖"好棋先搜"。V2 用 TT/killer/history 排序，
这里在根结点改用 NN policy 概率降序（TT 最优仍排第一），让搜索先展开 NN 认为
最有戏的走法，同时间内实际搜得更深。深层层内仍用原 TT/killer/history 排序
（NN 前向太贵，只花在根上）。
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
from engine import (AlphaBetaIDSEngine, _NEG_INF, _INF, _alphabeta_count,
                    _zobrist_init, _zobrist_move)
import alphazero_nn as aznn

cc = engine.cc


class NNAlphaBetaIDSEngine(AlphaBetaIDSEngine):
    def __init__(self, nn_pt_path, time_limit_sec=1.5, max_depth=8, eval_version="v2"):
        super().__init__(time_limit_sec=time_limit_sec, max_depth=max_depth,
                         eval_version=eval_version)
        from nn_torch import NetTorch
        self.model = NetTorch.load(nn_pt_path).eval()
        self.name = f"NNord-{self.name}"

    @torch.no_grad()
    def _nn_root_order(self, board, player, moves_root, tt_best_code):
        """返回按 NN policy 概率降序的走法列表；TT 最优（迭代加深上一层答案）排最前。
        moves_root: [(src, path)]（engine 格式）；tt_best 编码 = src*81+dst（同 order_moves）。"""
        x = torch.from_numpy(aznn.encode_board(board, player)).unsqueeze(0)
        pl, _ = self.model(x)
        indexed = [(s, p[-1], p) for (s, p) in moves_root]
        probs = aznn.policy_logits_to_move_probs(pl[0].numpy(), indexed)
        order = sorted(range(len(moves_root)), key=lambda i: -probs[i])
        if tt_best_code >= 0:
            for i, (s, p) in enumerate(moves_root):
                if s * 81 + p[-1] == tt_best_code:
                    order.remove(i)
                    order.insert(0, i)
                    break
        return [moves_root[i] for i in order]

    def think(self, board, player):
        moves_root = cc.get_all_moves(board, player)
        if not moves_root:
            return None, {"reason": "no_moves"}

        deadline = time.time() + self.time_limit
        eval_fn = self._eval_fn

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
            # 与原版唯一差异：根结点用 NN policy 排序替代 order_moves
            sorted_root_moves = self._nn_root_order(board, player, moves_root, root_best)

            for pos, path in sorted_root_moves:
                if time.time() >= deadline:
                    stats[2] = True
                    break
                new_board = cc.apply_move(board, path)
                score = _alphabeta_count(
                    new_board, 3 - player, depth - 1,
                    player, alpha, beta, deadline, stats, eval_fn,
                    tt, _zobrist_move(root_hash, path, player),
                    killer_table, history, 1)
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
