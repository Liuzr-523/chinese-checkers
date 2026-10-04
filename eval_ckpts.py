"""批量评估所有 params_bc_ckpt_*.pkl vs V0，每个评 4 局。
记录胜率，输出排名和最佳路径。
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import engine
from engine import AlphaBetaIDSEngine
import alphazero_nn as nn_mod
from alphazero_nn import NetParams, policy_logits_to_move_probs, forward

cc = engine.cc

HERE = Path(__file__).parent / "nn_checkpoints"


class NNEasyEngine:
    name = "NN-Easy"

    def __init__(self, params):
        self.params = params

    def think(self, board, player):
        moves = cc.get_all_moves(board, player)
        if not moves:
            return None, {}
        indexed = [(s, p[-1], p) for (s, p) in moves]
        s = nn_mod.encode_board(board, player)
        pl, _ = forward(self.params, s, {})
        logits = np.array([pl[:81][s] + pl[81:][d] for s, d, _ in indexed], dtype=np.float64)
        probs = policy_logits_to_move_probs(pl, [(s, d, p) for s, d, p in indexed])
        i = int(np.argmax(probs))
        return list(indexed[i][2]), {}


def play_one_game(nn_eng, v0_eng, nn_first, max_moves=200):
    board = cc.init_board()
    move_count = 0
    while True:
        w = cc.is_game_over(board)
        if w is not None:
            return w, move_count
        player = 1 if (move_count % 2 == 0) else 2
        if (player == 1 and nn_first) or (player == 2 and not nn_first):
            path, _ = nn_eng.think(board, player)
        else:
            path, _ = v0_eng.think(board, player)
        if path is None:
            return 3 - player, move_count
        board = cc.apply_move(board, path)
        move_count += 1
        if move_count >= max_moves:
            break
    s1 = engine.evaluate_v2(board, 1)
    s2 = engine.evaluate_v2(board, 2)
    return (1 if s1 >= s2 else 2), move_count


def eval_ckpt(ckpt_path, v0_eng, n_games=4, max_moves=200):
    params = NetParams.load(ckpt_path)
    nn_eng = NNEasyEngine(params)
    wins = 0
    losses = 0
    moves = []
    for g in range(n_games):
        nn_first = (g % 2 == 0)
        winner, m = play_one_game(nn_eng, v0_eng, nn_first, max_moves=max_moves)
        moves.append(m)
        nn_player = 1 if nn_first else 2
        if winner == nn_player:
            wins += 1
        elif winner == 3 - nn_player:
            losses += 1
    return wins, losses, moves


def main():
    v0 = AlphaBetaIDSEngine(time_limit_sec=0.5, max_depth=4, eval_version="v0")
    ckpts = sorted(HERE.glob("params_bc_ckpt_*.pkl"))
    print(f"[EVAL-CKPT] 找到 {len(ckpts)} 个 ckpt, 每个评 4 局", flush=True)
    results = []
    t0 = time.time()
    for ck in ckpts:
        t = time.time()
        wins, losses, moves = eval_ckpt(ck, v0, n_games=4, max_moves=200)
        elapsed = time.time() - t
        step_str = ck.stem.replace("params_bc_ckpt_", "")
        print(f"[EVAL-CKPT] step {step_str}: {wins}胜 {losses}负  平均步数 {sum(moves)/len(moves):.0f}  "
              f"用时 {elapsed:.0f}s", flush=True)
        results.append((step_str, wins, losses, moves, ck))
    results.sort(key=lambda r: (-r[1], r[2]))
    print("\n[EVAL-CKPT] === 排名 ===", flush=True)
    for i, (st, w, l, mv, ck) in enumerate(results, 1):
        print(f"  #{i}: step {st}  {w}:{l}  {ck.name}", flush=True)
    best = results[0]
    best_name = HERE / "params_bc_chosen_by_eval.pkl"
    if best_name.exists():
        best_name.unlink()
    import shutil
    shutil.copyfile(best[4], best_name)
    print(f"\n[EVAL-CKPT] 最佳: step {best[0]} {best[1]}:{best[2]}")
    print(f"[EVAL-CKPT] 已复制到 {best_name}")
    print(f"[EVAL-CKPT] 总用时 {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
