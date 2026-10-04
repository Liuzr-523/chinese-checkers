"""生成 V2 argmax + 开局随机 4 步的专家数据。
走法 = V2 argmax（能赢 V0 的水平），开局前 4 步随机打破固定序列保证局面多样。
输出到 replay_v2_argmax.pkl（不覆盖原 replay_v2.pkl）。
"""
import os
import sys
import time
import pickle
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import engine
from engine import AlphaBetaIDSEngine
import self_play_v2 as sp

cc = engine.cc


def main(n_games=150, max_moves=200, time_limit_sec=0.3,
         random_opening=4, pi_top=0.95, out_filename="replay_v2_argmax.pkl"):
    save_dir = Path(__file__).parent / "nn_checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / out_filename

    # 两个 V2 引擎，不设 seed 保证每局开局随机不同
    eng1 = AlphaBetaIDSEngine(time_limit_sec=time_limit_sec, max_depth=6, eval_version="v2")
    eng1.name = "V2arg-P1"
    eng2 = AlphaBetaIDSEngine(time_limit_sec=time_limit_sec, max_depth=6, eval_version="v2")
    eng2.name = "V2arg-P2"

    print(f"[ARG] V2 argmax + 开局随机{random_opening}步, {n_games}局, "
          f"time_limit={time_limit_sec}s, pi_top={pi_top}", flush=True)
    print(f"[ARG] 输出 {out_path}", flush=True)

    all_records = []
    t0 = time.time()
    p1w = 0
    p2w = 0
    for g in range(n_games):
        tg = time.time()
        try:
            rec = sp.self_play_v2_one_game(
                eng1, eng2, max_moves=max_moves, pi_top=pi_top,
                temperature=1.0, use_explore=False,
                random_opening=random_opening)
        except Exception as e:
            print(f"[ARG] 局{g+1} 异常: {e}, 跳过", flush=True)
            continue
        all_records.append(rec)
        if rec["winner"] == 1:
            p1w += 1
        else:
            p2w += 1
        if (g + 1) % 5 == 0 or g == n_games - 1:
            with open(out_path, "wb") as f:
                pickle.dump(all_records, f)
        el = time.time() - t0
        done = g + 1
        rem = (el / done) * (n_games - done)
        print(f"[ARG] 局 {g+1}/{n_games} winner=P{rec['winner']} "
              f"步数={len(rec['steps'])} 用时{time.time()-tg:.1f}s "
              f"P1={p1w} P2={p2w} 累计{el/60:.1f}min 剩{rem/60:.1f}min", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump(all_records, f)
    print(f"[ARG] 完成! {len(all_records)}局, P1={p1w} P2={p2w}, "
          f"用时{(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
