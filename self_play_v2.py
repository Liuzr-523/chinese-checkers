"""
V2 自对弈产棋谱 - 监督预训练数据生成（探索版 + 视角交换增强）

让 AlphaBetaIDSEngine(eval_version="v2") 自对弈 N 局，加 softmax 探索破后手模仿策略，
再对每局做 P1↔P2 视角交换数据增强（winner/player 翻转，expand 时自动重算 board 编码和 z），
最终输出 2N 局均衡数据到 nn_checkpoints/replay_v2_aug.pkl。

pi 用软标签 0.7（V2 选中的走法给 0.7，其余均匀分摊 0.3），NN 学的是"V2 偏好分布"。

用法：
  python3 self_play_v2.py --n_games 100 --time_limit_sec 0.3
  python3 self_play_v2.py --n_games 100 --temperature 1.0   # 调探索强度
"""
from checkpoint_paths import artifact_path, artifact_glob
import os
import sys
import time
import pickle
import argparse
import random
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
from engine import AlphaBetaIDSEngine

cc = engine.cc


def _v2_think_explore(board, player, eng, temperature=1.0):
    """探索版 V2 走法选择。

    对所有 root move，apply 后用 evaluate_v2 给静态评分（player 视角），
    softmax(score/temperature) 采样选一个走法。
    temperature 越高随机性越强（破模仿），越低越贪婪（接近纯 V2）。

    返回 (chosen_path, info_dict)
    """
    all_moves = cc.get_all_moves(board, player)  # [(pos, path), ...]
    if not all_moves:
        return None, {"reason": "no_moves"}

    # 给每个 root move 评分（apply 后从 player 视角看）
    scores = []
    for pos, path in all_moves:
        try:
            new_board = cc.apply_move(board, path)
            s = engine.evaluate_v2(new_board, player)
        except Exception:
            s = -1e9
        scores.append(float(s))

    scores = np.array(scores, dtype=np.float64)
    # softmax(scores / T)：减最大值防溢出
    logits = scores / max(temperature, 1e-6)
    logits = logits - logits.max()
    e = np.exp(logits)
    probs = e / e.sum()
    # 采样
    i = int(np.random.choice(len(all_moves), p=probs))
    chosen_path = list(all_moves[i][1])
    return chosen_path, {
        "move_count": len(all_moves),
        "chosen_idx": i,
        "chosen_prob": float(probs[i]),
        "max_score": float(scores.max()),
        "chosen_score": float(scores[i]),
    }


def self_play_v2_one_game(eng1, eng2, max_moves=200, pi_top=0.7, temperature=1.0,
                          use_explore=True, random_opening=0):
    """用 V2 引擎产一局棋谱，格式与 self_play_one_game_raw 兼容。

    use_explore=True 时用 _v2_think_explore（softmax 探索），
    use_explore=False 时用纯 eng.think（确定性 V2 argmax）。
    random_opening>0 时前 N 步随机选合法走法，打破纯 argmax 的固定序列，
    保证局面多样；之后按 use_explore 走。

    返回 {"winner": 1|2, "steps": [{player, move_path, moves, pi}, ...]}
    """
    board = cc.init_board()
    steps = []
    move_count = 0
    while True:
        w = cc.is_game_over(board)
        if w is not None:
            break
        player = 1 if (move_count % 2 == 0) else 2
        eng = eng1 if player == 1 else eng2
        # 选走法
        if move_count < random_opening:
            # 开局随机：随机选一个合法走法，保证局面多样
            all_moves = cc.get_all_moves(board, player)
            if not all_moves:
                break
            path = list(all_moves[np.random.randint(len(all_moves))][1])
        elif use_explore:
            path, _ = _v2_think_explore(board, player, eng, temperature=temperature)
        else:
            path, _ = eng.think(board, player)
        if path is None:
            break
        src = path[0]
        # 拿当前所有合法走法构造 moves 列表
        # cc.get_all_moves 返回 [(pos, path), ...]
        all_moves = cc.get_all_moves(board, player)
        move_tuples = [(m[0], m[1][-1], tuple(m[1])) for m in all_moves]
        n = len(move_tuples)
        # 找选中的步在 move_tuples 里的索引（先按 path 全等匹配）
        chosen_idx = None
        chosen_path_tuple = tuple(path)
        for i, (s, d, p) in enumerate(move_tuples):
            if s == src and p == chosen_path_tuple:
                chosen_idx = i
                break
        # 兜底：path 全等匹配不到，按 src+dst 匹配（连跳路径可能不同但起终点一样）
        if chosen_idx is None:
            for i, (s, d, p) in enumerate(move_tuples):
                if s == src and d == path[-1]:
                    chosen_idx = i
                    break
        # 实在不行就选第一步
        if chosen_idx is None:
            chosen_idx = 0
        # 软标签 pi：选中的步给 pi_top，其余均匀分摊
        pi = [(1.0 - pi_top) / max(n - 1, 1)] * n
        pi[chosen_idx] = pi_top
        s_total = sum(pi)
        pi = [p / s_total for p in pi]
        steps.append({
            "player": player,
            "move_path": list(path),
            "moves": move_tuples,
            "pi": pi,
        })
        board = cc.apply_move(board, path)
        move_count += 1
        if move_count >= max_moves:
            break

    # 决定 winner
    w = cc.is_game_over(board)
    if w is None:
        s1 = engine.evaluate_v2(board, 1)
        s2 = engine.evaluate_v2(board, 2)
        w = 1 if s1 >= s2 else 2
    return {"winner": w, "steps": steps}


def augment_with_swap(records):
    """对每条 record 生成一个 P1↔P2 视角交换的版本。

    交换操作：winner 1↔2，每个 step.player 1↔2，
    其他不动（move_path/moves/pi 是 player 无关的相对走法，board 在 expand 时按新 player 重算）。

    返回 records + swapped_records 的合并 list（长度翻倍）。
    """
    swapped = []
    for rec in records:
        new_rec = {
            "winner": 3 - rec["winner"],
            "steps": [],
        }
        for step in rec["steps"]:
            new_step = dict(step)
            new_step["player"] = 3 - step["player"]
            new_rec["steps"].append(new_step)
        swapped.append(new_rec)
    return list(records) + swapped


def main(n_games, max_moves, time_limit_sec, save_dir, pi_top, temperature,
         use_explore=True, seed=0, do_augment=True):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    replay_v2_path = artifact_path(Path(save_dir) / "replay_v2.pkl")
    replay_aug_path = artifact_path(Path(save_dir) / "replay_v2_aug.pkl")

    # 设随机种子
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # 两个 V2 引擎实例（一样配置），两边都是 v2 启发式
    # use_explore=True 时其实不会用到 eng.think（用 _v2_think_explore），
    # 但保留实例便于 use_explore=False 切换。
    eng1 = AlphaBetaIDSEngine(time_limit_sec=time_limit_sec, max_depth=6, eval_version="v2")
    eng1.name = "V2-P1"
    eng2 = AlphaBetaIDSEngine(time_limit_sec=time_limit_sec, max_depth=6, eval_version="v2")
    eng2.name = "V2-P2"

    mode = "explore" if use_explore else "deterministic"
    print(f"[V2-BC] 开始自对弈 {n_games} 局, 每局最多 {max_moves} 步, 模式={mode}", flush=True)
    print(f"[V2-BC] V2 time_limit={time_limit_sec}s, max_depth=6, pi_top={pi_top}, "
          f"temperature={temperature}", flush=True)
    print(f"[V2-BC] 原始棋谱保存到 {replay_v2_path}", flush=True)
    if do_augment:
        print(f"[V2-BC] 视角交换增强后保存到 {replay_aug_path}", flush=True)
    # 粗估耗时
    if use_explore:
        est_per_step = 0.05  # explore 版每步很快（apply + evaluate_v2）
    else:
        est_per_step = time_limit_sec + 0.1
    est_per_game = 80 * est_per_step
    est_total = n_games * est_per_game
    print(f"[V2-BC] 粗估单局 {est_per_game:.0f}s, 总耗时 {est_total/60:.0f} 分钟", flush=True)

    all_records = []
    # 断点续传：如果 replay_v2.pkl 已存在且非空，加载继续
    if replay_v2_path.exists():
        try:
            with open(replay_v2_path, "rb") as f:
                all_records = pickle.load(f)
            print(f"[V2-BC] 续传：已有 {len(all_records)} 局，继续追加", flush=True)
        except Exception:
            all_records = []

    t0 = time.time()
    start_idx = len(all_records)
    p1_wins = 0
    p2_wins = 0
    for g in range(start_idx, n_games):
        t_game = time.time()
        try:
            rec = self_play_v2_one_game(
                eng1, eng2, max_moves=max_moves, pi_top=pi_top,
                temperature=temperature, use_explore=use_explore)
        except Exception as e:
            print(f"[V2-BC] 第 {g+1} 局异常: {e}, 跳过", flush=True)
            continue
        all_records.append(rec)
        if rec["winner"] == 1:
            p1_wins += 1
        else:
            p2_wins += 1
        # 每 5 局保存一次（防意外中断丢数据）
        if (g + 1) % 5 == 0 or g == n_games - 1:
            with open(replay_v2_path, "wb") as f:
                pickle.dump(all_records, f)
        elapsed = time.time() - t0
        done = g + 1 - start_idx
        avg = elapsed / max(done, 1)
        remaining = avg * (n_games - g - 1)
        print(f"[V2-BC] 局 {g+1}/{n_games} 完成, winner=P{rec['winner']}, "
              f"步数={len(rec['steps'])}, 用时 {time.time()-t_game:.1f}s, "
              f"P1胜={p1_wins} P2胜={p2_wins}, "
              f"累计 {elapsed/60:.1f}min, 预计剩 {remaining/60:.1f}min", flush=True)

    print(f"[V2-BC] 自对弈完成! 共 {len(all_records)} 局", flush=True)
    print(f"[V2-BC] 胜负分布: P1胜={p1_wins} P2胜={p2_wins}", flush=True)
    print(f"[V2-BC] 自对弈总用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)

    # 视角交换增强
    if do_augment and all_records:
        print(f"[V2-BC] 开始视角交换数据增强...", flush=True)
        aug_records = augment_with_swap(all_records)
        with open(replay_aug_path, "wb") as f:
            pickle.dump(aug_records, f)
        # 增强后胜负分布（应该接近 50/50）
        aug_p1 = sum(1 for r in aug_records if r["winner"] == 1)
        aug_p2 = sum(1 for r in aug_records if r["winner"] == 2)
        print(f"[V2-BC] 增强后 {len(aug_records)} 局已保存到 {replay_aug_path}", flush=True)
        print(f"[V2-BC] 增强后胜负分布: P1胜={aug_p1} P2胜={aug_p2}", flush=True)
        total_steps = sum(len(r["steps"]) for r in aug_records)
        print(f"[V2-BC] 增强后总样本数(展开后) ≈ {total_steps}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games", type=int, default=100)
    parser.add_argument("--max_moves", type=int, default=200)
    parser.add_argument("--time_limit_sec", type=float, default=0.3)
    parser.add_argument("--pi_top", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="softmax 探索温度，高=随机破模仿，低=贪婪")
    parser.add_argument("--use_explore", type=int, default=1,
                        help="1=softmax 探索版, 0=纯 V2 确定性")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_augment", action="store_true",
                        help="跳过视角交换增强")
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "nn_checkpoints"))
    args = parser.parse_args()
    main(n_games=args.n_games, max_moves=args.max_moves,
         time_limit_sec=args.time_limit_sec, save_dir=args.save_dir,
         pi_top=args.pi_top, temperature=args.temperature,
         use_explore=bool(args.use_explore), seed=args.seed,
         do_augment=not args.no_augment)
