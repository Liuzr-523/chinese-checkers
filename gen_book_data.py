"""把 clysto 开局库变成 BC 训练样本。

从初始局面 BFS 展开所有合法走法，凡是 book 覆盖的局面就生成一条样本：
pi = book 推荐走法（平滑 0.95），z = 0（开局阶段胜负未知，取中性）。

用法: python3 gen_book_data.py [--out replay_book.pkl]
"""
from checkpoint_paths import artifact_path, artifact_glob
import argparse
import pickle
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, "/Users/liuzirui/Desktop/py/chinese-checkers")
import numpy as np
import engine
from alphazero_nn import encode_board
from clysto_book import _load, _hash, _TURN_GREEN_XOR

cc = engine.cc
HERE = Path("/Users/liuzirui/Desktop/py/chinese-checkers/nn_checkpoints")


def book_lookup(board, player, zob, book, map_h2o):
    """命中返回 (src, dst)（我们的坐标），否则 None。"""
    h = _hash(board, player, map_h2o, zob)
    for cand in (h, h ^ _TURN_GREEN_XOR):
        e = book.get(cand)
        if e is not None and board[map_h2o[e[0]]] == player:
            return map_h2o[e[0]], map_h2o[e[1]]
    return None


def main(out_path):
    zob, book, map_h2o = _load()
    samples = []
    seen = set()
    q = deque([(cc.init_board(), 1)])
    seen.add(_hash(cc.init_board(), 1, map_h2o, zob))
    while q:
        board, player = q.popleft()
        bm = book_lookup(board, player, zob, book, map_h2o)
        if bm is None:
            continue
        all_moves = cc.get_all_moves(board, player)
        move_tuples = [(m[0], m[1][-1], tuple(m[1])) for m in all_moves]
        chosen = None
        for j, (s, d, _) in enumerate(move_tuples):
            if s == bm[0] and d == bm[1]:
                chosen = j
                break
        if chosen is None:
            continue
        pi_top = 0.95
        n = len(move_tuples)
        pi = [(1.0 - pi_top) / max(n - 1, 1)] * n
        pi[chosen] = pi_top
        tot = sum(pi)
        pi = [p / tot for p in pi]
        s = encode_board(board, player)
        samples.append({"s": s, "player": player, "moves": move_tuples,
                        "pi": np.array(pi, dtype=np.float32), "z": 0.0})
        # 展开所有合法走法，子局面在 book 里就入队
        for _, p in all_moves:
            nb = cc.apply_move(board, p)
            hh = _hash(nb, 3 - player, map_h2o, zob)
            if hh in seen:
                continue
            seen.add(hh)
            q.append((nb, 3 - player))
    with open(out_path, "wb") as f:
        pickle.dump(samples, f)
    print(f"[BOOK-DATA] {len(samples)} 条样本 → {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="replay_book.pkl")
    a = ap.parse_args()
    main(artifact_path(HERE / a.out))
