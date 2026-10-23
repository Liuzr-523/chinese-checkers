"""用 clysto 的 MTD(f)+开局库 引擎自对弈，生成高质量专家棋谱（"拜师"数据源）。

输出到 nn_checkpoints/replay_clysto.pkl，格式与 self_play_v2_one_game 完全一致：
  rec = {"winner": 1|2, "steps": [
      {"player": 1|2, "move_path": [src,...,dst], "moves": [(src,dst,path),...], "pi": [...]}
  ]}
BC 训练（train_iteration_bc / expand_replay）直接读这个格式。

clysto 走法是确定性的（argmax），所以 pi 用"选中那步 pi_top + 其余均匀"的软标签。
"""
from checkpoint_paths import artifact_path, artifact_glob
import os
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
from engine import AlphaBetaIDSEngine
import arena_vs_clysto as ac   # 复用坐标映射 + ClystoEngine 适配器

cc = engine.cc


def cly_game(cly1, cly2, max_moves=250, pi_top=0.95, random_opening=6):
    """clysto 自对弈一局。cly1 执 P1，cly2 执 P2。返回标准棋谱 dict。

    random_opening>0：前 N 步双方随机走（clysto 开局库会把每局开局锁死在固定
    套路，30 局可能只有 4 种重复棋谱；随机开局打破多样性，中后盘仍由 clysto 接管）。
    """
    board = cc.init_board()
    steps = []
    move_count = 0
    winner = None
    while True:
        w = cc.is_game_over(board)
        if w is not None:
            winner = w
            break
        player = 1 if move_count % 2 == 0 else 2
        cly = cly1 if player == 1 else cly2

        # 开局随机：在合法走法里均匀抽一个，多样性优先（数据量足够后 NN 自己会泛化）
        if move_count < random_opening:
            all_moves0 = cc.get_all_moves(board, player)
            if not all_moves0:
                winner = 3 - player
                break
            path = list(random.choice(all_moves0)[1])
        else:
            path, _ = cly.think(board, player)
            if path is None:
                winner = 3 - player
                break

        # 构造与 self_play_v2 一致的 moves / pi 软标签
        all_moves = cc.get_all_moves(board, player)
        move_tuples = [(m[0], m[1][-1], tuple(m[1])) for m in all_moves]
        n = len(move_tuples)
        chosen_idx = None
        for i, (s, d, _) in enumerate(move_tuples):
            if s == path[0] and d == path[-1]:
                chosen_idx = i
                break
        if chosen_idx is None:      # 理论上必中（think 已校验合法性），兜底用
            chosen_idx = 0
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

    # 没自然完赛 = 双方在 max_moves 内没把 10 子全送进目标区（健康对局一般 ~100 步）。
    # 这种僵局的 z 标签本来就不存在，硬判"胜负"只会污染训练，标记 aborted 交给上层丢弃。
    return {"winner": winner, "steps": steps, "aborted": winner is None}


def _record_step(steps, board, player, path, pi_top, by):
    """把一步记录成标准格式（moves 用当前合法走法，pi 指向所选步）。

    by 标注这步是谁走的："cly"=clysto、"v2"=V2、"random"=随机开局。
    BC 训练只该学 by=="cly" 的步，其余只用于重建棋盘。
    """
    all_moves = cc.get_all_moves(board, player)
    move_tuples = [(m[0], m[1][-1], tuple(m[1])) for m in all_moves]
    n = len(move_tuples)
    chosen_idx = None
    for i, (s, d, _) in enumerate(move_tuples):
        if s == path[0] and d == path[-1]:
            chosen_idx = i
            break
    if chosen_idx is None:
        chosen_idx = 0
    pi = [(1.0 - pi_top) / max(n - 1, 1)] * n
    pi[chosen_idx] = pi_top
    s_total = sum(pi)
    steps.append({
        "player": player,
        "move_path": list(path),
        "moves": move_tuples,
        "pi": [p / s_total for p in pi],
        "by": by,
    })


def cly_game(cly1, cly2, max_moves=250, pi_top=0.95, random_opening=6):
    """clysto 自对弈一局。cly1 执 P1，cly2 执 P2。返回标准棋谱 dict。

    random_opening>0：前 N 步双方随机走（clysto 开局库会把每局开局锁死在固定
    套路，30 局可能只有 4 种重复棋谱；随机开局打破多样性，中后盘仍由 clysto 接管）。
    """
    board = cc.init_board()
    steps = []
    move_count = 0
    winner = None
    while True:
        w = cc.is_game_over(board)
        if w is not None:
            winner = w
            break
        player = 1 if move_count % 2 == 0 else 2
        cly = cly1 if player == 1 else cly2

        # 开局随机：在合法走法里均匀抽一个，多样性优先
        if move_count < random_opening:
            all_moves0 = cc.get_all_moves(board, player)
            if not all_moves0:
                winner = 3 - player
                break
            path = list(random.choice(all_moves0)[1])
            by = "random"
        else:
            path, _ = cly.think(board, player)
            if path is None:
                winner = 3 - player
                break
            by = "cly"
        _record_step(steps, board, player, path, pi_top, by)
        board = cc.apply_move(board, path)
        move_count += 1
        if move_count >= max_moves:
            break

    # 没自然完赛 = 双方在 max_moves 内没把 10 子全送进目标区（健康对局一般 ~100 步）。
    # 这种僵局的 z 标签本来就不存在，硬判"胜负"只会污染训练，标记 aborted 交给上层丢弃。
    return {"winner": winner, "steps": steps, "aborted": winner is None}


def cly_vs_v2_game(cly, v2, cly_player, max_moves=250, pi_top=0.95,
                   random_opening=6):
    """clysto 执 cly_player、V2 执另一边的一局棋谱。

    双方每步都记录（BC 展开需要连续走法重建棋盘），每步带 by 标注来源。
    random_opening>0：前 N 步双方随机走——clysto 和 V2 都是近确定性引擎，
    不加随机开局，多进程并行生成会产出完全相同的棋谱。
    """
    board = cc.init_board()
    steps = []
    move_count = 0
    winner = None
    while True:
        w = cc.is_game_over(board)
        if w is not None:
            winner = w
            break
        player = 1 if move_count % 2 == 0 else 2
        if move_count < random_opening:
            all_moves0 = cc.get_all_moves(board, player)
            if not all_moves0:
                winner = 3 - player
                break
            path = list(random.choice(all_moves0)[1])
            by = "random"
        else:
            if player == cly_player:
                path, _ = cly.think(board, player)
            else:
                path, _ = v2.think(board, player)
            if path is None:
                winner = 3 - player
                break
            by = "cly" if player == cly_player else "v2"
        _record_step(steps, board, player, path, pi_top, by)
        board = cc.apply_move(board, path)
        move_count += 1
        if move_count >= max_moves:
            break
    return {"winner": winner, "steps": steps, "aborted": winner is None}


def main(n_games=30, cly_time=1, max_moves=250, pi_top=0.95, random_opening=6,
         mode="self", out_filename=None, seed=None):
    if seed is not None:
        random.seed(seed)  # 并行生成时各进程传不同种子，保证随机开局不撞车
    save_dir = Path(__file__).parent / "nn_checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    if out_filename is None:
        out_filename = "replay_clysto.pkl" if mode == "self" else "replay_clysto_vs_v2.pkl"
    out_path = artifact_path(save_dir / out_filename)

    cly1 = ac.ClystoEngine(timelimit=cly_time)

    print(f"[CLY-GEN] mode={mode} {n_games} 局, clysto 每步 {cly_time}s, max_moves={max_moves}"
          f", 随机开局 {random_opening} 步", flush=True)
    print(f"[CLY-GEN] 输出 {out_path}", flush=True)

    if mode == "self":
        cly2 = ac.ClystoEngine(timelimit=cly_time)
    else:
        v2 = AlphaBetaIDSEngine(time_limit_sec=0.5, max_depth=6, eval_version="v2")
        v2.name = "V2"

    all_records = []
    t0 = time.time()
    p1w = p2w = 0
    try:
        for g in range(n_games):
            tg = time.time()
            try:
                if mode == "self":
                    rec = cly_game(cly1, cly2, max_moves=max_moves, pi_top=pi_top,
                                   random_opening=random_opening)
                else:
                    # V2 模式：奇偶局换边,保证 clysto 先后手各一半
                    cly_player = 1 if g % 2 == 0 else 2
                    rec = cly_vs_v2_game(cly1, v2, cly_player, max_moves=max_moves,
                                         pi_top=pi_top, random_opening=random_opening)
            except RuntimeError as e:
                print(f"[CLY-GEN] 局{g+1} 异常: {e}, 跳过", flush=True)
                continue
            if rec.get("aborted"):
                print(f"[CLY-GEN] 局{g+1} 超步数未分胜负, 丢弃(收 '{len(rec['steps'])}' 步废谱)",
                      flush=True)
                continue
            all_records.append(rec)
            if rec["winner"] == 1:
                p1w += 1
            elif rec["winner"] == 2:
                p2w += 1
            with open(out_path, "wb") as f:
                pickle.dump(all_records, f)
            done = len(all_records)
            el = time.time() - t0
            rem = (el / done) * (n_games - done) if done else 0
            print(f"[CLY-GEN] 局 {g+1}/{n_games} winner=P{rec['winner']} "
                  f"步数={len(rec['steps'])} 用时{time.time()-tg:.1f}s "
                  f"P1={p1w} P2={p2w} 累计{el/60:.1f}min 剩{rem/60:.1f}min",
                  flush=True)
    finally:
        cly1.quit()
        if mode == "self":
            cly2.quit()

    print(f"[CLY-GEN] 完成! {len(all_records)}局, P1={p1w} P2={p2w}, "
          f"用时{(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games", type=int, default=30)
    parser.add_argument("--cly_time", type=float, default=1)
    parser.add_argument("--max_moves", type=int, default=250)
    parser.add_argument("--pi_top", type=float, default=0.95)
    parser.add_argument("--random_opening", type=int, default=6)
    parser.add_argument("--mode", choices=["self", "v2"], default="self")
    parser.add_argument("--out_filename", default=None)
    parser.add_argument("--seed", type=int, default=None, help="随机开局种子，并行生成时每进程传不同值")
    args = parser.parse_args()
    main(args.n_games, args.cly_time, args.max_moves, args.pi_top,
         args.random_opening, args.mode, args.out_filename, args.seed)