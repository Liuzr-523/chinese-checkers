"""DAgger 纠错数据生成（两阶段）。

阶段1 play: 当前最佳 NN 跟 V0 下 N 局，记录每个 NN 轮到的局面(board, player, moves)。
阶段2 label: 把这些局面切给多个进程，让 clysto 对每个局面给一步纠正走法，
             生成 (s, moves, pi_cly, z) BC 样本 —— 直接解决 exposure bias。

用法:
  python3 dagger_gen.py play --ckpt params_ft_v2_ckpt_01200.pkl --n_games 50 --out states_XX.pkl --seed S
  python3 dagger_gen.py label --in states_XX.pkl --out dgrad_XX.pkl --shard i --nshards k --cly_time 0.5
"""
import argparse
import os
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/liuzirui/Desktop/py/chinese-checkers")
import numpy as np
import engine
from engine import AlphaBetaIDSEngine
from alphazero_nn import NetParams, NNEasyEngine, encode_board
import arena_vs_clysto as ac

cc = engine.cc
HERE = Path("/Users/liuzirui/Desktop/py/chinese-checkers/nn_checkpoints")


def play_games(ckpt, n_games, out_path, seed, max_moves=200, temperature=0.7,
               n_opening=0, opp="v0d4"):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    params = NetParams.load(HERE / ckpt)
    # 温度采样：argmax 确定性会导致多局轨迹完全重复，收集不到多样 on-policy 局面
    nn = NNEasyEngine(params, use_mcts=False, temperature=temperature, seed=seed)
    if opp == "v0d4":
        # 固定深度对手：与评估尺子一致，采集的局面分布匹配实战分布
        v0 = AlphaBetaIDSEngine(time_limit_sec=3600, max_depth=4, eval_version="v0")
        v0.name = "V0d4"
    else:
        v0 = AlphaBetaIDSEngine(time_limit_sec=0.5, max_depth=4, eval_version="v0")
        v0.name = "V0"

    rng = random.Random(seed)
    states = []
    t0 = time.time()
    for g in range(n_games):
        board = cc.init_board()
        move_count = 0
        # 随机开局（不计入 NN 局面样本）：打破确定性重复，覆盖更多中盘局面
        for _ in range(n_opening):
            player = 1 if move_count % 2 == 0 else 2
            mvs = cc.get_all_moves(board, player)
            if not mvs:
                break
            board = cc.apply_move(board, rng.choice(mvs)[1])
            move_count += 1
        nn_player = 1 if g % 2 == 0 else 2
        game_states = []
        while move_count < max_moves:
            w = cc.is_game_over(board)
            if w is not None:
                break
            player = 1 if move_count % 2 == 0 else 2
            eng = nn if player == nn_player else v0
            path, _ = eng.think(board, player)
            if path is None:
                break
            if player == nn_player:
                all_moves = cc.get_all_moves(board, player)
                move_tuples = [(m[0], m[1][-1], tuple(m[1])) for m in all_moves]
                game_states.append({"board": list(board), "player": player,
                                    "moves": move_tuples,
                                    "nn_move": (path[0], path[-1]),
                                    "winner": None})
            board = cc.apply_move(board, path)
            move_count += 1
        w = cc.is_game_over(board)
        for st in game_states:
            st["winner"] = w
        states += game_states
        print(f"[DAGGER-play] 局 {g+1}/{n_games} winner={w} "
              f"累计局面 {len(states)} 用时 {time.time()-t0:.0f}s", flush=True)
    with open(out_path, "wb") as f:
        pickle.dump(states, f)
    print(f"[DAGGER-play] 完成 {len(states)} 局面 → {out_path}", flush=True)


def label_states(in_path, out_path, shard, nshards, cly_time, pi_top=0.95):
    states = pickle.load(open(in_path, "rb"))
    mine = states[shard::nshards]
    cly = ac.ClystoEngine(timelimit=cly_time)
    out = []
    t0 = time.time()
    try:
        for i, st in enumerate(mine):
            if st["winner"] is None:
                continue
            path, _ = cly.think(st["board"], st["player"])
            if path is None:
                continue
            move_tuples = st["moves"]
            n = len(move_tuples)
            chosen = None
            for j, (s, d, _) in enumerate(move_tuples):
                if s == path[0] and d == path[-1]:
                    chosen = j
                    break
            if chosen is None:
                continue
            pi = [(1.0 - pi_top) / max(n - 1, 1)] * n
            pi[chosen] = pi_top
            tot = sum(pi)
            pi = [p / tot for p in pi]
            s = encode_board(st["board"], st["player"])
            z = 1.0 if st["player"] == st["winner"] else -1.0
            out.append({"s": s, "player": st["player"], "moves": move_tuples,
                        "pi": np.array(pi, dtype=np.float32), "z": z})
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f"[DAGGER-label {shard}] {i+1}/{len(mine)} "
                      f"用时 {el:.0f}s 剩 {el/(i+1)*(len(mine)-i-1):.0f}s",
                      flush=True)
                with open(out_path, "wb") as f:
                    pickle.dump(out, f)
    finally:
        cly.quit()
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"[DAGGER-label {shard}] 完成 {len(out)}/{len(mine)} → {out_path}",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["play", "label"])
    ap.add_argument("--ckpt", default="params_ft_v2_ckpt_01200.pkl")
    ap.add_argument("--n_games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--cly_time", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--n_opening", type=int, default=0)
    ap.add_argument("--opp", default="v0d4", choices=["v0", "v0d4"])
    a = ap.parse_args()
    if a.cmd == "play":
        play_games(a.ckpt, a.n_games, HERE / a.out, a.seed,
                   temperature=a.temperature, n_opening=a.n_opening, opp=a.opp)
    else:
        label_states(HERE / a.inp, HERE / a.out, a.shard, a.nshards, a.cly_time)
