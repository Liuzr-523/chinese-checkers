"""AlphaZero 式 MCTS 自对弈数据生成（单进程）。

双方都是同一个 NN+MCTS（每步按访问分布 π 采样保证多样性，标准 AlphaZero 做法）。
输出训练样本 {s, moves, pi(MCTS访问分布), z}：完赛局 z=±1，超步数判平 z=0（训练时 vw=0 只训 policy）。

用法:
  python3 az_selfplay.py --ckpt params_xxx.pkl --n_games 25 --seed 1 --out az_r1_0.pkl
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import engine
from alphazero_nn import NetParams, mcts_think, encode_board, all_moves_indexed

cc = engine.cc
HERE = Path(__file__).parent / "nn_checkpoints"


def play_game(params, rng, n_sim=24, max_moves=200):
    """一局自对弈。返回 samples list + winner（None=超步数）。"""
    board = cc.init_board()
    steps = []
    move_count = 0
    while move_count < max_moves:
        w = cc.is_game_over(board)
        if w is not None:
            break
        player = 1 if move_count % 2 == 0 else 2
        v_est, (moves, pi_probs), (best_s, best_path) = mcts_think(
            params, board, player, n_simulations=n_sim)
        move_tuples = [(m[0], m[1], tuple(m[2])) for m in moves]
        p = np.asarray(pi_probs, dtype=np.float64)
        s_sum = p.sum()
        if s_sum < 1e-12:
            i = int(rng.integers(len(moves)))
        else:
            i = int(rng.choice(len(moves), p=p / s_sum))
        steps.append({"player": player, "s": encode_board(board, player),
                      "moves": move_tuples, "pi": p / max(s_sum, 1e-12)})
        board = cc.apply_move(board, moves[i][2])
        move_count += 1
    w = cc.is_game_over(board)
    samples = []
    for st in steps:
        z = 0.0 if w is None else (1.0 if st["player"] == w else -1.0)
        samples.append({"s": st["s"], "player": st["player"], "moves": st["moves"],
                        "pi": st["pi"].astype(np.float32), "z": z})
    return samples, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_games", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_sim", type=int, default=24)
    ap.add_argument("--max_moves", type=int, default=200)
    a = ap.parse_args()
    np.random.seed(a.seed)  # mcts_think 的 Dirichlet 噪声用全局 np.random
    rng = np.random.default_rng(a.seed)
    params = NetParams.load(HERE / a.ckpt)
    out_path = HERE / a.out
    all_samples = []
    t0 = time.time()
    for g in range(a.n_games):
        samples, w = play_game(params, rng, n_sim=a.n_sim, max_moves=a.max_moves)
        all_samples += samples
        el = time.time() - t0
        print(f"[SP] 局 {g+1}/{a.n_games} winner={w} 步数={len(samples)} "
              f"累计样本 {len(all_samples)} 用时 {el:.0f}s", flush=True)
        with open(out_path, "wb") as f:
            pickle.dump(all_samples, f)
    print(f"[SP] 完成 {len(all_samples)} 样本 → {out_path}", flush=True)


if __name__ == "__main__":
    main()
