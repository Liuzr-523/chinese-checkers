"""评估 NN vs V2（NN 的老师）。
V2 = AlphaBetaIDSEngine(eval_version="v2", max_depth=6, time_limit=0.3s)
看 NN 能否青出于蓝胜过老师。
"""
from checkpoint_paths import artifact_path, artifact_glob
import os
import sys
import time
import random
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
from engine import AlphaBetaIDSEngine
from alphazero_nn import NetParams, NNEasyEngine

cc = engine.cc


def play_one_game(nn_engine, opp_engine, nn_first, max_moves=200, n_opening=0, rng=None):
    for eng in (nn_engine, opp_engine):
        if hasattr(eng, "reset_history"):
            eng.reset_history()
    board = cc.init_board()
    move_count = 0
    # 随机开局：双方各走 n_opening 步随机合法走法，打破确定性对局的重复。
    for _ in range(n_opening):
        w = cc.is_game_over(board)
        if w is not None:
            break
        player = 1 if (move_count % 2 == 0) else 2
        for eng in (nn_engine, opp_engine):
            if hasattr(eng, "observe_position"):
                eng.observe_position(board, player)
        moves = cc.get_all_moves(board, player)
        if not moves:
            break
        m = rng.choice(moves)
        board = cc.apply_move(board, m[1])
        move_count += 1
    while True:
        w = cc.is_game_over(board)
        if w is not None:
            return w, move_count
        player = 1 if (move_count % 2 == 0) else 2
        for eng in (nn_engine, opp_engine):
            if hasattr(eng, "observe_position"):
                eng.observe_position(board, player)
        if (player == 1 and nn_first) or (player == 2 and not nn_first):
            path, _ = nn_engine.think(board, player)
        else:
            path, _ = opp_engine.think(board, player)
        if path is None:
            return 3 - player, move_count
        board = cc.apply_move(board, path)
        move_count += 1
        if move_count >= max_moves:
            break
    return cc.is_game_over(board), move_count


def main(n_games, use_mcts, n_simulations, max_moves, temperature, seed, opponent, ckpt_path=None, n_opening=0, use_book=0, backend="numpy"):
    save_dir = Path(__file__).parent / "nn_checkpoints"
    if ckpt_path:
        ckpt = artifact_path(ckpt_path)
        if not ckpt.is_absolute():
            ckpt = artifact_path(save_dir / ckpt)
    else:
        ckpt = artifact_path(save_dir / "params_bc_latest.pkl")
    if not ckpt.exists():
        print(f"[EVAL] 参数文件未找到: {ckpt}")
        return
    print(f"[EVAL] 加载 NN 参数 {ckpt}")
    if backend == "torch":
        from nn_torch import NetTorch, NNEasyEngineTorch
        if ckpt.suffix == ".pt":
            try:
                model = NetTorch.load(ckpt)
            except Exception:
                from nn_resnet import NetResNet
                model = NetResNet.load(ckpt)
        else:
            pt = ckpt.with_suffix(".pt")
            model = NetTorch.load(pt) if pt.exists() else NetTorch.from_numpy_params(NetParams.load(ckpt))
        nn_engine = NNEasyEngineTorch(model, temperature=temperature, seed=seed,
                                      use_book=bool(use_book), use_mcts=use_mcts,
                                      n_simulations=n_simulations)
        nn_engine.name = "NN(torch)" + ("+book" if use_book else "")
    else:
        params = NetParams.load(ckpt)
        nn_engine = NNEasyEngine(params, use_mcts=use_mcts, n_simulations=n_simulations,
                                 temperature=temperature, seed=seed, use_book=bool(use_book))
        nn_engine.name = "NN(bc)" + ("+MCTS" if use_mcts else "") + ("+book" if use_book else "")
    if temperature > 0:
        nn_engine.name += f"(T={temperature})"

    if opponent == "v2":
        opp_engine = AlphaBetaIDSEngine(time_limit_sec=0.3, max_depth=6, eval_version="v2")
        opp_engine.name = "V2"
    elif opponent == "v0d4":
        # 固定深度 V0：时间限制放到极大，等价于纯 depth-4 搜索。
        # 结果不受 CPU 负载影响，可复现——限时搜索在并行跑评估时会被拖慢变弱。
        opp_engine = AlphaBetaIDSEngine(time_limit_sec=3600, max_depth=4, eval_version="v0")
        opp_engine.name = "V0d4"
    elif opponent == "v0d5":
        opp_engine = AlphaBetaIDSEngine(time_limit_sec=3600, max_depth=5, eval_version="v0")
        opp_engine.name = "V0d5"
    else:
        opp_engine = AlphaBetaIDSEngine(time_limit_sec=0.5, max_depth=4, eval_version="v0")
        opp_engine.name = "V0"

    print(f"[EVAL] {nn_engine.name} vs {opp_engine.name}, {n_games} 局, 交替先后手")
    print(f"[EVAL] use_mcts={use_mcts}, temperature={temperature}, max_moves={max_moves}, n_opening={n_opening}")
    rng = random.Random(seed if seed is not None else 12345)
    t0 = time.time()
    nn_wins = 0
    opp_wins = 0
    draws = 0
    total_moves = 0
    for g in range(n_games):
        nn_first = (g % 2 == 0)
        side = "P1(先)" if nn_first else "P2(后)"
        t_game = time.time()
        winner, moves = play_one_game(nn_engine, opp_engine, nn_first, max_moves=max_moves,
                                       n_opening=n_opening, rng=rng)
        nn_player = 1 if nn_first else 2
        if winner == nn_player:
            result = "NN胜"
            nn_wins += 1
        elif winner == 3 - nn_player:
            result = f"{opp_engine.name}胜"
            opp_wins += 1
        else:
            result = "未分胜负"
            draws += 1
        total_moves += moves
        print(f"[EVAL] 局 {g+1}/{n_games} NN走{side} → {result}, "
              f"步数={moves}, 用时 {time.time()-t_game:.1f}s, "
              f"累计 NN:{opp_engine.name} = {nn_wins}:{opp_wins} (未分胜负 {draws})", flush=True)
    print(f"[EVAL] === 最终结果 ===")
    print(f"[EVAL] NN:{opp_engine.name} = {nn_wins}:{opp_wins} (未分胜负 {draws})")
    print(f"[EVAL] 平均步数 {total_moves/n_games:.0f}")
    print(f"[EVAL] 总用时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games", type=int, default=10)
    parser.add_argument("--use_mcts", type=int, default=0)
    parser.add_argument("--n_simulations", type=int, default=24)
    parser.add_argument("--max_moves", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--opponent", type=str, default="v2", choices=["v0", "v0d4", "v0d5", "v2"])
    parser.add_argument("--ckpt", type=str, default=None, help="ckpt 文件名或绝对路径；默认 params_bc_latest.pkl")
    parser.add_argument("--n_opening", type=int, default=0, help="随机开局步数（双方各计），打破确定性对局重复")
    parser.add_argument("--use_book", type=int, default=0, help="1=NN 开局优先走 clysto 开局库")
    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch"])
    args = parser.parse_args()
    main(args.n_games, args.use_mcts, args.n_simulations, args.max_moves,
         args.temperature, args.seed, args.opponent, ckpt_path=args.ckpt,
         n_opening=args.n_opening, use_book=args.use_book, backend=args.backend)
