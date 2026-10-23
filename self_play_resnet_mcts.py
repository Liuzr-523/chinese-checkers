"""ResNet + MCTS 自对弈产数据（路线B）。

用 PyTorch ResNet 模型（d11 冠军）做 MCTS 自对弈，产出与 gen_clysto_data.py
相同格式的棋谱，用于后续训练。

关键差异 vs alphazero_nn.self_play_one_game_raw：
- 模型：PyTorch ResNet (NetResNet) 而非 numpy NetParams
- MCTS evaluate：resnet_forward() 替代 numpy forward()
- 随机开局：前 N 步随机走，打破多样性
"""
from checkpoint_paths import artifact_path, artifact_glob
import os
import sys
import time
import pickle
import random
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
from alphazero_nn import (encode_board, policy_logits_to_move_probs,
                          all_moves_indexed, H, W, CIN)
from nn_resnet import NetResNet

cc = engine.cc


# ---------- ResNet 前向桥接 ----------
@torch.no_grad()
def resnet_forward(model, board, player):
    """ResNet 前向，返回 (policy_logits, value)。接口对齐 alphazero_nn.forward()"""
    enc = encode_board(board, player)
    x = torch.from_numpy(enc).unsqueeze(0).float()
    pl, v = model(x)
    return pl[0].numpy(), float(v[0])


# ---------- MCTS（基于 alphazero_nn.mcts_think 逻辑，换 ResNet 前向） ----------
class Node:
    __slots__ = ("board", "player", "P", "children", "N", "W", "Q", "v_est", "expanded")

    def __init__(self, board, player, prior_p):
        self.board = board
        self.player = player
        self.P = prior_p
        self.children = []
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.v_est = 0.0
        self.expanded = False


from mcts_history import position_key, candidate_indices


def mcts_think_resnet(model, board, player, n_simulations=64, c_puct=1.25, *, add_root_noise=False, rng=None, history=None, discount=1.0):
    """ResNet 版 MCTS。返回 (moves_root, pi_vec), (best_s, best_path)"""
    if not 0 < discount <= 1:
        raise ValueError("discount must be in (0, 1]")
    history = set(history or ())
    root_board = list(board)
    moves_root = all_moves_indexed(root_board, player)
    if not moves_root:
        return ([], []), (None, None)

    pl, root_v = resnet_forward(model, root_board, player)
    priors_root = policy_logits_to_move_probs(pl, moves_root)
    # Dirichlet 噪声（探索）
    if add_root_noise:
        rng = rng if rng is not None else np.random
        alpha = 0.6 / max(1, len(moves_root))
        noise = rng.dirichlet([alpha] * len(moves_root)).astype(np.float32)
        priors_root = 0.75 * priors_root + 0.25 * noise

    root = Node(root_board, player, priors_root)
    root.children = []
    for i, (s, d, p) in enumerate(moves_root):
        nxt_board = cc.apply_move(root_board, p)
        nxt_player = 3 - player
        root.children.append((s, d, p, Node(nxt_board, nxt_player, None)))
    root.expanded = True

    def evaluate_node(node):
        w = cc.is_game_over(node.board)
        if w is not None:
            node.v_est = 1.0 if w == node.player else -1.0
            node.children = []
            node.expanded = True
            return node.v_est
        pl, v = resnet_forward(model, node.board, node.player)
        node.v_est = v
        moves = all_moves_indexed(node.board, node.player)
        if not moves:
            node.v_est = -1.0
            node.children = []
            node.expanded = True
            return node.v_est
        probs = policy_logits_to_move_probs(pl, moves)
        node.P = probs
        node.children = []
        for i, (s, d, p) in enumerate(moves):
            nxt_board = cc.apply_move(node.board, p)
            nxt_player = 3 - node.player
            node.children.append((s, d, p, Node(nxt_board, nxt_player, None)))
        node.expanded = True
        return node.v_est

    def traverse():
        path = []
        node = root
        seen = set(history)
        seen.discard(position_key(root.board, root.player))
        repeated = False
        while True:
            key = position_key(node.board, node.player)
            if key in seen and cc.is_game_over(node.board) is None:
                repeated = True
                break
            if not (node.expanded and node.children):
                break
            seen.add(key)
            eligible = candidate_indices(node.children, seen, cc.is_game_over)
            n_parent = node.N
            best_u = -1e9
            best_i = 0
            for i in eligible:
                ch = node.children[i]
                c = ch[3]
                N_i = c.N
                P_i = node.P[i]
                U_i = c.Q + c_puct * P_i * (np.sqrt(max(1, n_parent)) / (1 + N_i))
                if U_i > best_u:
                    best_u = U_i
                    best_i = i
            path.append((node, best_i))
            node = node.children[best_i][3]
        v = 0.0 if repeated else evaluate_node(node)
        cur_v = -v * discount
        for parent, ci in reversed(path):
            child = parent.children[ci][3]
            child.N += 1
            child.W += cur_v
            child.Q = child.W / child.N
            cur_v = -cur_v * discount
        root.N += 1  # Each traversed node is visited exactly once per simulation.

    for _ in range(n_simulations):
        traverse()

    counts = np.array([c[3].N for c in root.children], dtype=np.float64)
    if counts.sum() == 0:
        probs = np.ones(len(root.children)) / len(root.children)
    else:
        probs = counts / counts.sum()

    pi_moves = list(probs)
    best_i = int(np.argmax(counts))
    best_s, _, best_path, _ = root.children[best_i]
    return (moves_root, pi_moves), (best_s, list(best_path))


# ---------- 自对弈 ----------
def self_play_one_game(model, n_simulations=64, max_moves=200, random_opening=6):
    """ResNet + MCTS 自对弈一局。返回标准棋谱 dict（兼容 gen_clysto_data 格式）。"""
    board = cc.init_board()
    steps = []
    history = set()
    move_count = 0
    winner = None
    while True:
        w = cc.is_game_over(board)
        if w is not None:
            winner = w
            break
        player = 1 if move_count % 2 == 0 else 2
        history.add((tuple(board), player))

        if move_count < random_opening:
            all_moves0 = cc.get_all_moves(board, player)
            if not all_moves0:
                winner = 3 - player
                break
            path = list(random.choice(all_moves0)[1])
            by = "random"
        else:
            (moves_root, pi_vec), (best_s, best_path) = mcts_think_resnet(
                model, board, player, n_simulations=n_simulations, add_root_noise=True, history=history)
            if best_path is None:
                winner = 3 - player
                break
            path = best_path
            by = "nn_mcts"

        # 记录（格式对齐 gen_clysto_data）
        all_moves = cc.get_all_moves(board, player)
        move_tuples = [(m[0], m[1][-1], tuple(m[1])) for m in all_moves]
        n = len(move_tuples)
        if move_count < random_opening:
            # 随机步：均匀分布
            pi = [1.0 / n] * n
        else:
            # MCTS 步：用 MCTS 的访问分布
            pi_map = {}
            for i, (s, d, p) in enumerate(moves_root):
                pi_map[(s, tuple(p))] = pi_vec[i]
            pi = [pi_map.get((s, tuple(p)), 1e-8) for (s, d, p) in move_tuples]
            s_total = sum(pi)
            pi = [p / s_total for p in pi]

        steps.append({
            "player": player,
            "move_path": list(path),
            "moves": move_tuples,
            "pi": pi,
            "by": by,
        })
        board = cc.apply_move(board, path)
        move_count += 1
        if move_count >= max_moves:
            break

    winner = cc.is_game_over(board)
    return {"winner": winner, "steps": steps, "aborted": winner is None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_games", type=int, default=30)
    ap.add_argument("--n_sim", type=int, default=64)
    ap.add_argument("--max_moves", type=int, default=200)
    ap.add_argument("--random_opening", type=int, default=6)
    ap.add_argument("--ckpt", type=str, default="params_resnet_champ.pt")
    ap.add_argument("--out", type=str, default="replay_nn_mcts_b.pkl")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    save_dir = Path(__file__).parent / "nn_checkpoints"
    out_path = artifact_path(save_dir / args.out)

    ckpt_path = artifact_path(save_dir / args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"模型不存在: {ckpt_path}")
    model = NetResNet.load(ckpt_path).eval()
    print(f"[NN-MCTS] 加载模型 {ckpt_path.name}", flush=True)

    print(f"[NN-MCTS] {args.n_games} 局, n_sim={args.n_sim}, "
          f"随机开局{args.random_opening}步, 输出 {out_path}", flush=True)

    all_records = []
    t0 = time.time()
    p1w = p2w = 0
    try:
        for g in range(args.n_games):
            tg = time.time()
            rec = self_play_one_game(model, n_simulations=args.n_sim,
                                     max_moves=args.max_moves,
                                     random_opening=args.random_opening)
            if rec.get("aborted"):
                print(f"[NN-MCTS] 局{g+1} 超步数未分胜负, 丢弃({len(rec['steps'])}步)",
                      flush=True)
                continue
            all_records.append(rec)
            if rec["winner"] == 1:
                p1w += 1
            elif rec["winner"] == 2:
                p2w += 1
            with open(out_path, "wb") as f:
                pickle.dump(all_records, f)
            el = time.time() - t0
            rem = (el / len(all_records)) * (args.n_games - len(all_records))
            print(f"[NN-MCTS] 局 {g+1}/{args.n_games} winner=P{rec['winner']} "
                  f"步数={len(rec['steps'])} 用时{time.time()-tg:.1f}s "
                  f"P1={p1w} P2={p2w} 累计{el/60:.1f}min 剩{rem/60:.1f}min",
                  flush=True)
    except KeyboardInterrupt:
        print(f"[NN-MCTS] 中断, 已存 {len(all_records)} 局", flush=True)

    print(f"[NN-MCTS] 完成! {len(all_records)}局, P1={p1w} P2={p2w}, "
          f"用时{(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
