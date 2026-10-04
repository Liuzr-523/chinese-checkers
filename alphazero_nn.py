"""
AlphaZero 风格 CNN (纯 numpy 实现) + MCTS(PUCT) + 自对弈 + 训练
棋盘是 17行 * 9格 的三角网格，用 9x17 的 2D 网格编码。

CNN:
  - 输入: (C=4, H=17, W=9)  float32
      ch0: 当前方棋子(0/1)，ch1: 对方棋子(0/1)
      ch2: 空/占用，ch3: 当前方身份(P1=1 P2=-1)
  - 2层 Conv3x3 + ReLU，通道 32
  - policy head: Conv1x1 → flatten → softmax(81)
  - value  head: Conv1x1 → 64ReLU → tanh(1)
所有 (src,dst) 合法棋步映射到 81×81 的 index = src*81+dst（共 6561）。
为了省空间，实际 policy logits 输出 81+81（src 权重和 dst 权重），
合法走法 src→dst 的 logit = w_src[src] + w_dst[dst]。
"""
import os
import sys
import time
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
from engine import AlphaBetaIDSEngine

cc = engine.cc

H, W = 17, 9
CIN = 4
FILTERS = 32
POLICY_OUT = 81  # src + dst 各 81（即 162 维 logits）


# ---------- 工具 ----------
def encode_board(board, player):
    """→ (C=4, H=17, W=9)"""
    x = np.zeros((CIN, H, W), dtype=np.float32)
    opp = 3 - player
    for pos in range(81):
        cx, cy = cc.POS_TO_XY[pos]
        # cx ∈ [0,8], cy ∈ [0,16]
        if board[pos] == player:
            x[0, cy, cx] = 1.0
        elif board[pos] == opp:
            x[1, cy, cx] = 1.0
        if board[pos] != 0:
            x[2, cy, cx] = 1.0
    x[3] = 1.0 if player == 1 else -1.0
    return x


def all_moves_indexed(board, player):
    """返回 [(src, dst, path)]"""
    moves = cc.get_all_moves(board, player)
    result = []
    for src, path in moves:
        dst = path[-1]
        result.append((src, dst, path))
    return result


# ---------- 网络参数 ----------
class NetParams:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        # 两层 Conv 3x3
        def conv_he(ci, co):
            w = rng.standard_normal((co, ci, 3, 3)).astype(np.float32) * np.sqrt(2.0 / (ci * 9)).astype(np.float32)
            b = np.zeros((co,), dtype=np.float32)
            return w, b

        self.w1, self.b1 = conv_he(CIN, FILTERS)
        self.w2, self.b2 = conv_he(FILTERS, FILTERS)

        # policy head: 2层 FC，17*9*FILTERS → 64 → 162
        fc_in = H * W * FILTERS
        self.wp1 = (rng.standard_normal((fc_in, 64)) * np.sqrt(2.0 / fc_in)).astype(np.float32)
        self.bp1 = np.zeros((64,), dtype=np.float32)
        self.wp2 = (rng.standard_normal((64, 162)) * np.sqrt(2.0 / 64)).astype(np.float32)
        self.bp2 = np.zeros((162,), dtype=np.float32)

        # value head
        self.wv1 = (rng.standard_normal((fc_in, 64)) * np.sqrt(2.0 / fc_in)).astype(np.float32)
        self.bv1 = np.zeros((64,), dtype=np.float32)
        self.wv2 = (rng.standard_normal((64, 1)) * np.sqrt(2.0 / 64)).astype(np.float32)
        self.bv2 = np.zeros((1,), dtype=np.float32)

    def as_list(self):
        return [self.w1, self.b1, self.w2, self.b2,
                self.wp1, self.bp1, self.wp2, self.bp2,
                self.wv1, self.bv1, self.wv2, self.bv2]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------- 前向 / 反向 ----------
def _relu(x):
    return np.maximum(x, 0)


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _conv3x3(x, w, b):
    """输入 (C,H,W), 权重 (Co,C,3,3) → (Co,H,W) SAME padding"""
    Ci, Hi, Wi = x.shape
    Co, _, _, _ = w.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="constant")
    out = np.zeros((Co, Hi, Wi), dtype=np.float32)
    # 朴素实现，棋盘小性能可接受
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for i in range(Hi):
            for j in range(Wi):
                patch = xp[:, i:i + 3, j:j + 3].reshape(1, Ci * 9)
                out[:, i, j] = patch @ w.reshape(Co, Ci * 9).T + b
    return out


def _dconv3x3(dy, x, w):
    """d(conv3x3)/dw, d(conv3x3)/dx"""
    Ci, Hi, Wi = x.shape
    Co, _, _, _ = w.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="constant")
    dw = np.zeros_like(w)
    dx_pad = np.zeros_like(xp)
    w_flat = w.reshape(Co, Ci * 9)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for i in range(Hi):
            for j in range(Wi):
                patch = xp[:, i:i + 3, j:j + 3].reshape(Ci * 9, 1)
                dyi = dy[:, i, j][:, None]  # (Co,1)
                dw += (dyi @ patch.T).reshape(Co, Ci, 3, 3)
                dx_patch = (w_flat.T @ dyi).reshape(Ci, 3, 3)
                dx_pad[:, i:i + 3, j:j + 3] += dx_patch
    db = dy.sum(axis=(1, 2))
    dx = dx_pad[:, 1:-1, 1:-1]
    return dw, db, dx


def forward(params, x, cache=None):
    """x: (C,H,W) 单样本。返回 policy_logits(162,), value(标量)。若 cache 是 dict，则回填反向图"""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # conv1
        z1 = _conv3x3(x, params.w1, params.b1)
        h1 = _relu(z1)
        # conv2
        z2 = _conv3x3(h1, params.w2, params.b2)
        h2 = _relu(z2)
        # flatten
        flat = h2.reshape(-1)
        # policy
        pz1 = flat @ params.wp1 + params.bp1
        ph1 = _relu(pz1)
        plogits = ph1 @ params.wp2 + params.bp2  # (162,)
        # value
        vz1 = flat @ params.wv1 + params.bv1
        vh1 = _relu(vz1)
        vpre = vh1 @ params.wv2 + params.bv2  # (1,)
    value = float(np.tanh(vpre[0]))

    if cache is not None:
        cache.update({
            "x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2,
            "flat": flat, "pz1": pz1, "ph1": ph1,
            "vz1": vz1, "vh1": vh1, "vpre": vpre,
        })
    return plogits, value


def backward(params, cache, dplogits, dvalue):
    """返回各参数的梯度列表。
    dplogits: dL/d(plogits), shape (162,)
    dvalue:   dL/d(value), 标量
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # value: tanh
        dvpre = (1.0 - np.tanh(cache["vpre"]) ** 2) * np.array([dvalue], dtype=np.float32)
        dvh1 = dvpre @ params.wv2.T  # (64,)
        dvz1 = dvh1 * (cache["vz1"] > 0)
        dwv2 = np.outer(cache["vh1"], dvpre)
        dbv2 = dvpre.copy()
        dwv1 = np.outer(cache["flat"], dvz1)
        dbv1 = dvz1.copy()
        dflat_v = dvz1 @ params.wv1.T

        # policy: L 是 softmax CE，外面给的是 dL/d(plogits)
        dph1 = dplogits @ params.wp2.T  # (64,)
        dpz1 = dph1 * (cache["pz1"] > 0)
        dwp2 = np.outer(cache["ph1"], dplogits)
        dbp2 = dplogits.copy()
        dwp1 = np.outer(cache["flat"], dpz1)
        dbp1 = dpz1.copy()
        dflat_p = dpz1 @ params.wp1.T

        dflat = dflat_v + dflat_p
        dh2 = dflat.reshape(cache["h2"].shape)
        dz2 = dh2 * (cache["z2"] > 0)
        dw2, db2, dh1 = _dconv3x3(dz2, cache["h1"], params.w2)
        dz1 = dh1 * (cache["z1"] > 0)
        dw1, db1, dx = _dconv3x3(dz1, cache["x"], params.w1)

    # 清洗可能的 NaN/Inf（梯度裁剪前的安全网）
    result = [dw1, db1, dw2, db2,
              dwp1, dbp1, dwp2, dbp2,
              dwv1, dbv1, dwv2, dbv2]
    result = [np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0) for g in result]
    return result


def policy_logits_to_move_probs(plogits, moves):
    """
    plogits: (162,) = [src_81, dst_81]
    moves: [(src, dst, path)]
    返回: probs array 与 moves 对齐（已归一化）；对应 logit = src_logit[src] + dst_logit[dst]
    """
    src_lg = plogits[:81]
    dst_lg = plogits[81:]
    lgs = np.array([src_lg[s] + dst_lg[d] for (s, d, _) in moves], dtype=np.float32)
    lgs = lgs - lgs.max()
    ex = np.exp(lgs)
    s = ex.sum()
    if s < 1e-12:
        return np.ones(len(moves), dtype=np.float32) / len(moves)
    return ex / s


# ---------- MCTS (PUCT) ----------
class Node:
    __slots__ = ("board", "player", "P", "children", "N", "W", "Q", "v_est", "expanded")

    def __init__(self, board, player, prior_p):
        self.board = board
        self.player = player
        self.P = prior_p  # 走法概率 (len=len(children_pending))
        self.children = []  # [(src, dst, path, node)] — expand 时填充
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.v_est = 0.0   # 该结点被评估的 value（终端时用真实z）
        self.expanded = False


def _backup(visited_nodes, v):
    # visited_nodes: [node, index_of_action] × depth
    for node, _ in visited_nodes:
        # 对 node 来说，我们在 node 这一层选了子动作，加的值是“该动作方向的收益”——即下一手 P 视角的 v 反转。
        # 但更简单：在 traverse 里把 v 按 player 视角处理好，这里直接累加。
        v = -v  # 下一层的视角是对方的反
        pass
    # 上面是占位，具体在 traverse 函数内处理备份。


def mcts_think(params, board, player, n_simulations=48, c_puct=1.25):
    """跑 MCTS。返回:
    pi: (src,dst) → prob 的 dict（只含合法走法）
    best_move: (src, path) — 按 π 采样或选 argmax
    """
    root_board = list(board)

    # 1) 根结点先得到根合法走法的 priors 与 v
    moves_root = all_moves_indexed(root_board, player)
    if not moves_root:
        return None, (None, None)
    enc = encode_board(root_board, player)
    cache = {}
    pl, root_v = forward(params, enc, cache)
    priors_root = policy_logits_to_move_probs(pl, moves_root)
    # 加 Dirichlet 噪声（探索）
    alpha = 0.6 / max(1, len(moves_root))
    noise = np.random.dirichlet([alpha] * len(moves_root)).astype(np.float32)
    priors_root = 0.75 * priors_root + 0.25 * noise

    root = Node(root_board, player, priors_root)
    # 立即展开根
    root.children = []
    for i, (s, d, p) in enumerate(moves_root):
        nxt_board = cc.apply_move(root_board, p)
        nxt_player = 3 - player
        root.children.append((s, d, p, Node(nxt_board, nxt_player, None)))  # child.P 等评估再填
    root.expanded = True

    def evaluate_node(node):
        """为 node 计算 v_est，如果还在叶子且未扩展则填充 P 和子（非终端）。返回 v_est。"""
        w = cc.is_game_over(node.board)
        if w is not None:
            # 终端：当前 node.player 视角，如果 w=player 则为 1，否则 -1
            node.v_est = 1.0 if w == node.player else -1.0
            # 终端无子，children 空列表
            node.children = []
            node.expanded = True
            return node.v_est
        enc = encode_board(node.board, node.player)
        pl, v = forward(params, enc)
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
        """一次模拟。沿树选到叶子，扩展并评估，然后回溯更新。"""
        path = []  # [(parent_node, child_index)]
        node = root
        while node.expanded and node.children:
            # PUCT 选择
            n_parent = node.N
            best_u = -1e9
            best_i = 0
            for i, ch in enumerate(node.children):
                c = ch[3]
                N_i = c.N
                P_i = node.P[i]
                U_i = c.Q + c_puct * P_i * (np.sqrt(max(1, n_parent)) / (1 + N_i))
                if U_i > best_u:
                    best_u = U_i
                    best_i = i
            path.append((node, best_i))
            node = node.children[best_i][3]

        # 到叶子：评估
        v = evaluate_node(node)
        # 回溯：沿 path 更新。注意对 node.player 视角 v 是“node 这一方”的价值，给父的 value 是 -v。
        # 所以从 path[-1] 的父（即 path[-1][0]）往 root 更新，对应它的第 path[-1][1] 个孩子的统计 += -v。
        cur_v = -v
        for parent, ci in reversed(path):
            child = parent.children[ci][3]
            child.N += 1
            child.W += cur_v
            child.Q = child.W / child.N
            parent.N += 1
            cur_v = -cur_v

    # 模拟
    for _ in range(n_simulations):
        traverse()

    # 根 π ∝ N^(1/temp)，temp=1
    counts = np.array([c[3].N for c in root.children], dtype=np.float64)
    if counts.sum() == 0:
        probs = np.ones(len(root.children)) / len(root.children)
    else:
        probs = counts / counts.sum()

    # 返回 pi dict + best_move（按 counts 最多的那个）
    pi = {}
    for i, (s, d, p, _) in enumerate(root.children):
        pi[(s, tuple(p))] = probs[i]
    best_i = int(np.argmax(counts))
    best_s, _, best_path, _ = root.children[best_i]

    # 顺便把用于训练的访问分布 π_vec 也返回（对齐 moves_root 顺序）
    pi_moves = list(pi.values())
    return root_v, (moves_root, pi_moves), (best_s, list(best_path))


# ---------- 自对弈 ----------
def self_play_one_game_raw(params, n_simulations, max_moves=200):
    """返回一条完整棋谱（可存入 replay.pkl 的原始格式），不含展开后的 (s,π,z)。
    结构:
      {
        "winner": 1 or 2,
        "steps": [
          {"player": 1, "move_path": [src,...,dst], "moves": [...], "pi": [...]},
          ...
        ]
      }
    每一步还附带着当时的合法走法列表 moves 和对应 pi（MCTS访问分布），用于之后展开样本。
    """
    board = cc.init_board()
    steps = []
    move_count = 0
    while True:
        w = cc.is_game_over(board)
        if w is not None:
            break
        player = 1 if (move_count % 2 == 0) else 2
        v_est, (moves, pi_probs), (best_s, best_path) = mcts_think(
            params, board, player, n_simulations=n_simulations)
        move_tuples = [(m[0], m[1], tuple(m[2])) for m in moves]
        if move_count < 15:
            p = np.array(pi_probs, dtype=np.float64)
            s = p.sum()
            if s < 1e-12:
                i = np.random.choice(len(moves))
            else:
                i = int(np.random.choice(len(moves), p=p / s))
        else:
            i = int(np.argmax(pi_probs))
        _, _, played_path = moves[i]
        steps.append({
            "player": player,
            "move_path": list(played_path),
            "moves": move_tuples,
            "pi": [float(x) for x in pi_probs],
        })
        board = cc.apply_move(board, played_path)
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


def expand_game_records(game_records):
    """把 replay 中的多条 game_record（完整棋谱）展开成训练样本 list[dict]。
    每个样本：{s: (4,17,9), moves: list[(src,dst,path_tuple)], pi: (len(moves),), z: float}
    """
    samples = []
    for rec in game_records:
        winner = rec["winner"]
        board = cc.init_board()
        for step in rec["steps"]:
            player = step["player"]
            s = encode_board(board, player)
            pi = np.array(step["pi"], dtype=np.float32)
            z = 1.0 if player == winner else -1.0
            samples.append({
                "s": s,
                "player": player,
                "moves": step["moves"],
                "pi": pi,
                "z": z,
            })
            # 推进棋盘，保证下一步的 s 正确（这个 replay 存的每步都带着当时的 moves 和 pi，
            # 推进只为了校验一致性，若与 step.move_path 不一致说明文件损坏）
            board = cc.apply_move(board, step["move_path"])
    return samples


# ---------- Replay buffer：持久化到 replay.pkl ----------
REPLAY_FILENAME = "replay.pkl"
DEFAULT_REPLAY_CAP = 200  # 最多保留多少局完整棋谱（滑动窗口）


def _replay_path(save_dir):
    return Path(save_dir) / REPLAY_FILENAME


def load_replay(save_dir, cap=DEFAULT_REPLAY_CAP):
    p = _replay_path(save_dir)
    if not p.exists():
        return []
    with open(p, "rb") as f:
        data = pickle.load(f)
    # 保留最近 cap 局（避免历史文件过大）
    if len(data) > cap:
        data = data[-cap:]
    return data


def save_replay(save_dir, records, cap=DEFAULT_REPLAY_CAP):
    """把 records（完整棋谱列表）滑到最近 cap 局，写入 replay.pkl。"""
    if len(records) > cap:
        records = records[-cap:]
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(_replay_path(save_dir), "wb") as f:
        pickle.dump(records, f)


def append_replay(save_dir, new_records, cap=DEFAULT_REPLAY_CAP):
    old = load_replay(save_dir, cap=cap)
    merged = old + list(new_records)
    if len(merged) > cap:
        merged = merged[-cap:]
    save_replay(save_dir, merged, cap=cap)
    return merged


# 兼容旧名字（外部代码可能仍在调用）：也保留 self_play_one_game，行为 = 自对弈+立刻展开
def self_play_one_game(params, n_simulations, max_moves=200):
    rec = self_play_one_game_raw(params, n_simulations=n_simulations, max_moves=max_moves)
    return expand_game_records([rec])


# ---------- 训练 ----------
def sample_minibatch(replay, batch_size, rng):
    """从 replay[list(dict)] 采样 batch，转换为适合训练的数组。
    因为每步合法走法个数不一，我们把 CE 损失做成：对每个样本，网络输出 162 logits，把它们按该样本的 moves 投影成 per-move logits，再算 CE。
    """
    idxs = rng.integers(0, len(replay), size=batch_size)
    batch = [replay[i] for i in idxs]
    xs = np.stack([b["s"] for b in batch], axis=0)  # (B,4,17,9)
    return batch, xs


def compute_loss_and_grads(params, batch, weight_decay=1e-4, label_sharpen=1.0):
    """batch: list of dict。返回 (loss, grads_list)
    对每个样本：
      pl, v = forward(params, s, cache)
      per-move logits = pl[:81][src] + pl[81:][dst]  （len = len(moves)）
      CE = - Σ pi * log(softmax(per-move logits))
      MSE = (v - z)^2
    loss 末尾加 L2 正则项 = weight_decay * Σ||W||²（只对权重层）
    label_sharpen: >1 时对 pi 做 pi^k 归一化，让软标签更尖锐（选中走法权重更高）
    """
    xs = np.stack([b["s"] for b in batch], axis=0)
    B = len(batch)
    total_loss = 0.0
    # 梯度累加器
    grads = [np.zeros_like(p) for p in params.as_list()]
    caches = []
    pls = []
    vs = []
    for i in range(B):
        cache = {}
        pl, v = forward(params, xs[i], cache)
        caches.append(cache)
        pls.append(pl)
        vs.append(v)

    for i, rec in enumerate(batch):
        pl = pls[i]
        v = vs[i]
        moves = rec["moves"]
        pi = rec["pi"]  # (len(moves),)
        if label_sharpen != 1.0:
            pi = pi ** label_sharpen
            s_pi = pi.sum()
            if s_pi > 1e-12:
                pi = pi / s_pi
        z = rec["z"]

        # policy CE
        lgs = np.array([pl[:81][s] + pl[81:][d] for (s, d, _) in moves], dtype=np.float64)
        lgs = lgs - lgs.max()
        ex = np.exp(lgs)
        s_exp = ex.sum()
        if s_exp < 1e-12:
            probs_m = np.ones(len(moves)) / len(moves)
        else:
            probs_m = ex / s_exp
        # CE
        ce = -float(np.sum(pi * np.log(probs_m + 1e-12)))
        # value MSE
        mse = float((v - z) ** 2)
        total_loss += (ce + mse)

        # dL/dv
        dv = 2.0 * (v - z)
        # dL/d(pl)  per logit = Σ π[p] - probs[p]：对 per-move logit 梯度是 probs - π
        dper = (probs_m - pi)  # (len(moves),)
        dsrc = np.zeros(81, dtype=np.float64)
        ddst = np.zeros(81, dtype=np.float64)
        for j, (s, d, _) in enumerate(moves):
            g = dper[j]
            dsrc[s] += g
            ddst[d] += g
        dpl = np.concatenate([dsrc, ddst]).astype(np.float32)

        grd = backward(params, caches[i], dpl, dv)
        for k in range(len(grads)):
            grads[k] += grd[k]

    total_loss /= B
    for k in range(len(grads)):
        grads[k] = grads[k] / B

    # L2 正则项加到 loss（只算权重层，不算偏置）
    weight_indices = [0, 2, 4, 6, 8, 10]
    l2_reg = sum(float(np.sum(params.as_list()[i] ** 2)) for i in weight_indices)
    total_loss += weight_decay * l2_reg

    return total_loss, grads


def train_step_sgd(params, grads, lr=1e-2, max_norm=5.0, weight_decay=1e-4):
    """SGD 更新 + 梯度裁剪 + L2 权重衰减。
    max_norm: 全局梯度范数上限，超过就等比缩小（防爆炸）。
    weight_decay: L2 正则系数，直接加到梯度上（等价 W -= lr*(grad + wd*W)）。
    """
    # 1. 全局梯度裁剪
    total_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
    if total_norm > max_norm and total_norm > 0:
        scale = max_norm / total_norm
        grads = [g * scale for g in grads]

    # 2. L2 权重衰减（只对权重 W，不对偏置 b）
    ps = params.as_list()
    weight_indices = [0, 2, 4, 6, 8, 10]  # w1,w2,wp1,wp2,wv1,wv2 的索引
    for i in range(len(ps)):
        g = grads[i]
        if i in weight_indices:
            g = g + weight_decay * ps[i]
        ps[i] -= lr * g

    params.w1, params.b1, params.w2, params.b2 = ps[:4]
    params.wp1, params.bp1, params.wp2, params.bp2 = ps[4:8]
    params.wv1, params.bv1, params.wv2, params.bv2 = ps[8:]


class AdamOptimizer:
    """简单 Adam 优化器（带梯度裁剪 + L2 权重衰减）。

    比纯 SGD 多了：
      - 动量 m（一阶矩，看历史梯度方向）
      - 自适应 lr v（二阶矩，每个参数单独调步长）
      - 偏差修正（修正初始 m/v 偏向 0 的问题）

    对"卡在局部最优"特别有效，因为动量能帮跳出。
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 max_norm=5.0, weight_decay=1e-4):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_norm = max_norm
        self.weight_decay = weight_decay
        self.t = 0
        ps = params.as_list()
        self.m = [np.zeros_like(p) for p in ps]
        self.v = [np.zeros_like(p) for p in ps]

    def step(self, params, grads):
        # 1. 全局梯度裁剪
        total_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
        if total_norm > self.max_norm and total_norm > 0:
            scale = self.max_norm / total_norm
            grads = [g * scale for g in grads]

        self.t += 1
        ps = params.as_list()
        weight_indices = [0, 2, 4, 6, 8, 10]

        for i in range(len(ps)):
            g = grads[i]
            # L2 权重衰减（只对权重）
            if i in weight_indices:
                g = g + self.weight_decay * ps[i]
            # Adam 更新
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            ps[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        params.w1, params.b1, params.w2, params.b2 = ps[:4]
        params.wp1, params.bp1, params.wp2, params.bp2 = ps[4:8]
        params.wv1, params.bv1, params.wv2, params.bv2 = ps[8:]


# ---------- NN Engine：适配 GUI / play_vs_ai 的接口 ----------
class NNEasyEngine:
    """用 NN 的快速走法（1 次前向 + argmax policy 合法走法）。"""
    name = "NN-easy"

    def __init__(self, params, use_mcts=True, n_simulations=32,
                 temperature=0.0, seed=None):
        # temperature=0 → argmax（确定性）；>0 → 按概率采样，越大越随机
        self.params = params
        self.use_mcts = use_mcts
        self.n_simulations = n_simulations
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def think(self, board, player):
        if self.use_mcts:
            _, _, (best_s, best_path) = mcts_think(
                self.params, board, player, n_simulations=self.n_simulations)
            if best_path is None:
                return None, {}
            return best_path, {"engine": self.name, "mcts": True}
        else:
            moves = cc.get_all_moves(board, player)
            if not moves:
                return None, {}
            indexed = [(s, p[-1], p) for (s, p) in moves]
            enc = encode_board(board, player)
            pl, _ = forward(self.params, enc)
            probs = policy_logits_to_move_probs(pl, indexed)
            if self.temperature <= 0:
                i = int(np.argmax(probs))
            else:
                p = probs ** (1.0 / self.temperature)
                p = p / p.sum()
                i = int(self.rng.choice(len(indexed), p=p))
            return indexed[i][2], {"engine": self.name, "temperature": self.temperature}


# ---------- 一键入口 ----------
def train_iteration(
    save_dir="/Users/liuzirui/Desktop/py/chinese-checkers/nn_checkpoints",
    n_games_per_step=10, n_sim=24, n_steps=1, bs=8, lr=3e-4, seed_init=0,
    replay_cap=DEFAULT_REPLAY_CAP, train_steps_per_it=50,
    max_norm=5.0, weight_decay=1e-4,
    optimizer="adam",
    warm_start_path=None,
    save_ckpt_every=1,
):
    """
    AlphaZero 式完整训练迭代（支持 warm start + Adam + 迭代 ckpt）：

      每一步 iteration：
        1. 让当前网络自对弈 n_games_per_step 局（完整棋谱），append 进 replay.pkl
        2. 从 replay.pkl（滑动窗口，最多 replay_cap 局）中展开全部样本
        3. 随机抽 mini-batch 训练 train_steps_per_it 步
           optimizer="adam" → 用 AdamOptimizer（推荐，含梯度裁剪+L2）
           optimizer="sgd"  → 用 train_step_sgd
        4. 保存 params_latest.pkl；若 save_ckpt_every>0 且 it+1 满足条件，
           额外存 params_az_it_{it+1:03d}.pkl

    warm_start_path:
      若指定（如 BC 预训练的 params_bc_chosen_by_eval.pkl），**第一次** 找不到
      params_latest.pkl 时优先加载它（而不是重新初始化）。
      支持文件名（相对 save_dir）或绝对路径。
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "params_latest.pkt"  # fallback 占位（稍后覆盖）
    ckpt_path = save_dir / "params_latest.pkl"
    replay_path = _replay_path(save_dir)

    if ckpt_path.exists():
        print(f"[AZ] 加载已有参数 {ckpt_path}（续训）", flush=True)
        params = NetParams.load(ckpt_path)
    elif warm_start_path is not None:
        ws = Path(warm_start_path)
        if not ws.is_absolute():
            ws = save_dir / ws
        if ws.exists():
            print(f"[AZ] 加载 warm start 参数 {ws}（BC 模型）", flush=True)
            params = NetParams.load(ws)
        else:
            raise FileNotFoundError(f"warm_start_path 不存在: {ws}")
    else:
        params = NetParams(seed=seed_init)
        print("[AZ] 初始化新网络（零起点）", flush=True)

    # 优化器（每个 train_iteration 调用新建；若做长生命周期续训，后续可持久化 opt 状态）
    if optimizer == "adam":
        opt = AdamOptimizer(params, lr=lr, max_norm=max_norm, weight_decay=weight_decay)
        print(f"[AZ] 优化器 = Adam(lr={lr}, beta1=0.9, beta2=0.999)", flush=True)
    else:
        opt = None
        print(f"[AZ] 优化器 = SGD(lr={lr})", flush=True)

    rng = np.random.default_rng(42)

    for it in range(n_steps):
        t_it = time.time()
        # 1. 自对弈（产出完整棋谱），append 到 replay
        new_records = []
        for gi in range(n_games_per_step):
            t_g = time.time()
            rec = self_play_one_game_raw(params, n_simulations=n_sim)
            new_records.append(rec)
            elapsed = time.time() - t_g
            print(f"  [{it + 1}/{n_steps}] selfplay game {gi + 1}/{n_games_per_step}"
                  f"  winner=P{rec['winner']}  steps={len(rec['steps'])}"
                  f"  {elapsed:.1f}s/game", flush=True)

        merged = append_replay(save_dir, new_records, cap=replay_cap)
        print(f"  replay 总棋谱 {len(merged)} (保存在 {replay_path})", flush=True)

        # 2. 展开为训练样本
        samples = expand_game_records(merged)
        print(f"  展开训练样本数 {len(samples)}", flush=True)

        # 3. 训练若干步（Adam 或 SGD）
        n_train_steps = train_steps_per_it if len(samples) >= bs else max(1, len(samples) // bs)
        for step in range(n_train_steps):
            batch, xs = sample_minibatch(samples, bs, rng)
            loss, grads = compute_loss_and_grads(params, batch, weight_decay=weight_decay)
            if opt is not None:
                opt.step(params, grads)
            else:
                train_step_sgd(params, grads, lr=lr, max_norm=max_norm, weight_decay=weight_decay)
            if step % max(1, n_train_steps // 5) == 0 or step == n_train_steps - 1:
                print(f"  train step {step + 1}/{n_train_steps} loss={loss:.4f}", flush=True)

        # 4. 保存 latest + 条件保存迭代 ckpt
        params.save(ckpt_path)
        print(f"  [{it + 1}/{n_steps}] 保存 latest → {ckpt_path}", flush=True)
        if save_ckpt_every > 0 and ((it + 1) % save_ckpt_every == 0 or it == n_steps - 1):
            it_ck = save_dir / f"params_az_it_{it + 1:03d}.pkl"
            params.save(it_ck)
            print(f"  [{it + 1}/{n_steps}] 保存迭代 ckpt → {it_ck}", flush=True)
        print(f"  [{it + 1}/{n_steps}] 迭代总用时 {(time.time()-t_it)/60:.1f}min", flush=True)
    return params


def train_iteration_bc(
    save_dir="/Users/liuzirui/Desktop/py/chinese-checkers/nn_checkpoints",
    replay_filename="replay_v2_aug.pkl",
    n_steps=500, bs=32, lr=1e-3, seed_init=7,
    max_norm=5.0, weight_decay=1e-4,
    eval_every=50, holdout_ratio=0.1,
    ckpt_filename="params_bc_latest.pkl",
    optimizer="adam",
    label_sharpen=1.0,
    save_ckpt_every=0,
):
    """...
    save_ckpt_every>0 时每 save_ckpt_every 步额外存 ckpt，文件名 params_bc_ckpt_{step}.pkl
    """
    """
    监督预训练（Behavior Cloning）：
      1. 加载 replay_v2_aug.pkl（V2 自对弈棋谱 + 视角交换增强后的数据）
      2. 展开成训练样本 (s, pi, z)
      3. 切 holdout 验证集（holdout_ratio 比例），监控泛化
      4. 训练 n_steps 步（带梯度裁剪 + L2 正则）
         optimizer="adam" 用 AdamOptimizer（默认，推荐）
         optimizer="sgd"  用 train_step_sgd（旧版，对比用）
      5. 每 eval_every 步在验证集上算 loss，打印 train_loss / val_loss
      6. 保存 params_bc_latest.pkl

    跟 train_iteration 区别：不自对弈，直接用现成棋谱；lr 稍高、bs 大、训练步数多。
    监督训练 loss 应该明显下降（不像 RL 从零那样 loss 不动）。
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / ckpt_filename
    replay_path = save_dir / replay_filename

    if not replay_path.exists():
        raise FileNotFoundError(f"棋谱文件未找到: {replay_path}")

    # 1. 加载并展开棋谱
    with open(replay_path, "rb") as f:
        records = pickle.load(f)
    print(f"[BC] 加载 {len(records)} 局棋谱 from {replay_path}", flush=True)
    samples = expand_game_records(records)
    print(f"[BC] 展开训练样本数 {len(samples)}", flush=True)

    # 2. 切 holdout 验证集
    rng = np.random.default_rng(seed_init)
    # 用 rng.shuffle 打乱（samples 是 list，需要转 idx）
    idxs = np.arange(len(samples))
    rng.shuffle(idxs)
    n_val = max(1, int(len(samples) * holdout_ratio))
    val_samples = [samples[i] for i in idxs[:n_val]]
    train_samples = [samples[i] for i in idxs[n_val:]]
    print(f"[BC] 训练集 {len(train_samples)}, 验证集 {len(val_samples)}", flush=True)

    # 3. 加载或初始化网络
    if ckpt_path.exists():
        print(f"[BC] 加载已有参数 {ckpt_path}", flush=True)
        params = NetParams.load(ckpt_path)
    else:
        params = NetParams(seed=seed_init)
        print(f"[BC] 初始化新网络 seed={seed_init}", flush=True)

    # 3.5 初始化优化器
    if optimizer == "adam":
        opt = AdamOptimizer(params, lr=lr, max_norm=max_norm, weight_decay=weight_decay)
        print(f"[BC] 优化器 = Adam(lr={lr}, beta1=0.9, beta2=0.999)", flush=True)
    else:
        opt = None
        print(f"[BC] 优化器 = SGD(lr={lr})", flush=True)

    # 4. 训练循环
    print(f"[BC] 开始训练 {n_steps} 步, bs={bs}, lr={lr}, "
          f"max_norm={max_norm}, weight_decay={weight_decay}, "
          f"label_sharpen={label_sharpen}", flush=True)
    t0 = time.time()
    best_val_loss = float("inf")
    for step in range(n_steps):
        batch, _ = sample_minibatch(train_samples, bs, rng)
        loss, grads = compute_loss_and_grads(params, batch, weight_decay=weight_decay,
                                             label_sharpen=label_sharpen)
        if opt is not None:
            opt.step(params, grads)
        else:
            train_step_sgd(params, grads, lr=lr, max_norm=max_norm, weight_decay=weight_decay)

        if save_ckpt_every > 0 and ((step + 1) % save_ckpt_every == 0 or step == n_steps - 1):
            ck = Path(save_dir) / f"params_bc_ckpt_{step+1:05d}.pkl"
            params.save(ck)

        if step % eval_every == 0 or step == n_steps - 1:
            # 在验证集上算 loss（不开 L2，纯看预测准不准）
            val_bs = min(bs * 2, len(val_samples))
            val_batch, _ = sample_minibatch(val_samples, val_bs, rng)
            val_loss, _ = compute_loss_and_grads(params, val_batch, weight_decay=0.0,
                                                 label_sharpen=label_sharpen)
            elapsed = time.time() - t0
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                params.save(Path(save_dir) / "params_bc_best.pkl")
                mark = " ★ 保存best"
            else:
                mark = ""
            print(f"[BC] step {step+1}/{n_steps} train_loss={loss:.4f} "
                  f"val_loss={val_loss:.4f} elapsed={elapsed:.1f}s{mark}", flush=True)

    # 5. 保存最后参数到 ckpt_path；best 已循环内保存到 params_bc_best.pkl
    # 注：实战发现 last 参数（5:5）反而比 best 参数（0:10）强，故评估用 last
    params.save(ckpt_path)
    print(f"[BC] 完成! 最后参数 → {ckpt_path}", flush=True)
    print(f"[BC] best参数 → {Path(save_dir) / 'params_bc_best.pkl'}", flush=True)
    print(f"[BC] 总用时 {time.time()-t0:.1f}s, 最佳 val_loss={best_val_loss:.4f}", flush=True)
    return params


if __name__ == "__main__":
    # 小烟雾测试：初始化 + 自对弈 1 局 + 1 步训练
    print("烟雾测试：初始化网络")
    p = NetParams(seed=1)
    x = encode_board(cc.init_board(), 1)
    pl, v = forward(p, x)
    print(f"  forward shape: policy {pl.shape}, value={v:.4f}")

    moves = all_moves_indexed(cc.init_board(), 1)
    probs = policy_logits_to_move_probs(pl, moves)
    print(f"  root moves={len(moves)}, probs sum={probs.sum():.4f}")

    print("MCTS(6 sim) 一次")
    v, (ms, ps), (bs, bp) = mcts_think(p, cc.init_board(), 1, n_simulations=6)
    print(f"  MCTS OK, best={bs}→{bp}")

    print("自对弈 1 局（MCTS 6 模拟）")
    hist = self_play_one_game(p, n_simulations=6, max_moves=8)
    print(f"  samples={len(hist)}")

    print("训练 1 小步")
    loss, grads = compute_loss_and_grads(p, hist[:4])
    print(f"  loss={loss:.4f}, grad norms: w1={np.linalg.norm(grads[0]):.4f} wp2={np.linalg.norm(grads[6]):.4f}")
    train_step_sgd(p, grads, lr=1e-2)
    print("DONE")
