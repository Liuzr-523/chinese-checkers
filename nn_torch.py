"""PyTorch 版 AlphaZero CNN —— 与 alphazero_nn.py（手写 numpy 版）结构完全一致，
用于加速训练迭代。权重可双向转换，评估/自对弈管线无需改动。

网络: 输入(4,17,9) → conv3x3(4→F)+ReLU → conv3x3(F→F)+ReLU → flatten
      policy: FC(→64)+ReLU → FC(→162)  [81 src + 81 dst]
      value : FC(→64)+ReLU → FC(→1)+tanh
"""
import os
import sys
import time
import pickle
import argparse
import glob as globmod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
from alphazero_nn import (NetParams, encode_board, policy_logits_to_move_probs,
                          H, W, CIN, FILTERS)

cc = engine.cc
FC_IN = H * W * FILTERS  # 4896


# ---------- 网络 ----------
class NetTorch(nn.Module):
    def __init__(self, filters=FILTERS):
        super().__init__()
        self.filters = filters
        self.conv1 = nn.Conv2d(CIN, filters, 3, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        fc_in = H * W * filters
        self.p1 = nn.Linear(fc_in, 64)
        self.p2 = nn.Linear(64, 162)
        self.v1 = nn.Linear(fc_in, 64)
        self.v2 = nn.Linear(64, 1)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        flat = h.flatten(1)
        plogits = self.p2(F.relu(self.p1(flat)))
        value = torch.tanh(self.v2(F.relu(self.v1(flat))))
        return plogits, value.squeeze(-1)

    # ---- 与 numpy NetParams 双向转换 ----
    @staticmethod
    def from_numpy_params(np_params):
        m = NetTorch(filters=getattr(np_params, "filters", FILTERS))
        with torch.no_grad():
            m.conv1.weight.copy_(torch.from_numpy(np_params.w1))
            m.conv1.bias.copy_(torch.from_numpy(np_params.b1))
            m.conv2.weight.copy_(torch.from_numpy(np_params.w2))
            m.conv2.bias.copy_(torch.from_numpy(np_params.b2))
            m.p1.weight.copy_(torch.from_numpy(np_params.wp1.T))
            m.p1.bias.copy_(torch.from_numpy(np_params.bp1))
            m.p2.weight.copy_(torch.from_numpy(np_params.wp2.T))
            m.p2.bias.copy_(torch.from_numpy(np_params.bp2))
            m.v1.weight.copy_(torch.from_numpy(np_params.wv1.T))
            m.v1.bias.copy_(torch.from_numpy(np_params.bv1))
            m.v2.weight.copy_(torch.from_numpy(np_params.wv2.T))
            m.v2.bias.copy_(torch.from_numpy(np_params.bv2))
        return m

    def to_numpy_params(self):
        p = object.__new__(NetParams)
        p.filters = self.filters
        with torch.no_grad():
            p.w1 = self.conv1.weight.detach().numpy().copy()
            p.b1 = self.conv1.bias.detach().numpy().copy()
            p.w2 = self.conv2.weight.detach().numpy().copy()
            p.b2 = self.conv2.bias.detach().numpy().copy()
            p.wp1 = self.p1.weight.detach().numpy().T.copy()
            p.bp1 = self.p1.bias.detach().numpy().copy()
            p.wp2 = self.p2.weight.detach().numpy().T.copy()
            p.bp2 = self.p2.bias.detach().numpy().copy()
            p.wv1 = self.v1.weight.detach().numpy().T.copy()
            p.bv1 = self.v1.bias.detach().numpy().copy()
            p.wv2 = self.v2.weight.detach().numpy().T.copy()
            p.bv2 = self.v2.bias.detach().numpy().copy()
        return p

    def save(self, path):
        torch.save({"state_dict": self.state_dict(), "filters": self.filters}, path)

    @staticmethod
    def load(path):
        d = torch.load(path, map_location="cpu", weights_only=True)
        m = NetTorch(filters=d["filters"])
        m.load_state_dict(d["state_dict"])
        return m


def convert_numpy_to_torch(pkl_path):
    """旧 numpy 权重 pkl → NetTorch"""
    return NetTorch.from_numpy_params(NetParams.load(pkl_path))


# ---------- 推理引擎（与 NNEasyEngine 接口兼容） ----------
class NNEasyEngineTorch:
    name = "NN-torch"

    def __init__(self, model, temperature=0.0, seed=None, use_book=False):
        self.model = model.eval()
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.use_book = use_book

    @torch.no_grad()
    def _forward(self, board, player):
        x = torch.from_numpy(encode_board(board, player)).unsqueeze(0)
        pl, v = self.model(x)
        return pl[0].numpy(), float(v)

    def think(self, board, player):
        if self.use_book:
            import clysto_book
            bm = clysto_book.book_move(board, player, cc.get_all_moves)
            if bm is not None:
                path = next(p for s, p in cc.get_all_moves(board, player)
                            if s == bm[0] and p[-1] == bm[1])
                return path, {"engine": self.name, "book": True}
        moves = cc.get_all_moves(board, player)
        if not moves:
            return None, {}
        indexed = [(s, p[-1], p) for (s, p) in moves]
        pl, _ = self._forward(board, player)
        probs = policy_logits_to_move_probs(pl, indexed)
        if self.temperature <= 0:
            i = int(np.argmax(probs))
        else:
            p = probs ** (1.0 / self.temperature)
            p = p / p.sum()
            i = int(self.rng.choice(len(indexed), p=p))
        return indexed[i][2], {"engine": self.name, "temperature": self.temperature}


# ---------- 训练 ----------
def _make_batch(batch, device):
    """list[dict] → padded tensors。moves 变长，用 mask 处理。"""
    B = len(batch)
    M = max(len(rec["moves"]) for rec in batch)
    xs = np.stack([rec["s"] for rec in batch])
    src_idx = np.zeros((B, M), dtype=np.int64)
    dst_idx = np.zeros((B, M), dtype=np.int64)
    pi = np.zeros((B, M), dtype=np.float32)
    mask = np.zeros((B, M), dtype=np.float32)
    z = np.zeros((B,), dtype=np.float32)
    vw = np.ones((B,), dtype=np.float32)
    for i, rec in enumerate(batch):
        ms = rec["moves"]
        for j, (s, d, _) in enumerate(ms):
            src_idx[i, j] = s
            dst_idx[i, j] = d + 81  # logits 后 81 维是 dst
        pi[i, :len(ms)] = np.asarray(rec["pi"], dtype=np.float32)
        mask[i, :len(ms)] = 1.0
        z[i] = rec["z"]
        vw[i] = rec.get("vw", 1.0)
    t = lambda a: torch.from_numpy(a).to(device)
    return t(xs), t(src_idx), t(dst_idx), t(pi), t(mask), t(z), t(vw)


def train_bc_torch(samples, model, steps=800, lr=2e-4, bs=32, weight_decay=1e-4,
                   seed=0, log_every=100, ckpt_cb=None, device="cpu"):
    """BC 训练：masked CE(policy) + vw*MSE(value)，Adam+梯度裁剪+L2。
    ckpt_cb(step, model) 每步回调（保存用）。返回最终 train loss。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    last_loss = None
    t0 = time.time()
    for step in range(steps):
        idxs = rng.integers(0, len(samples), size=bs)
        batch = [samples[i] for i in idxs]
        xs, src_i, dst_i, pi, mask, z, vw = _make_batch(batch, device)
        plogits, value = model(xs)
        lgs = plogits.gather(1, src_i) + plogits.gather(1, dst_i)
        lgs = lgs.masked_fill(mask == 0, -1e30)
        logp = F.log_softmax(lgs, dim=1)
        ce = -(pi * logp).sum(1)
        vle = vw * (value - z) ** 2
        loss = (ce + vle).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        last_loss = float(loss.detach())
        if log_every and (step + 1) % log_every == 0:
            print(f"[TORCH-TRAIN] step {step+1}/{steps} loss={last_loss:.4f} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
        if ckpt_cb:
            ckpt_cb(step + 1, model)
    return last_loss


# ---------- CLI 入口（被 train_torch.py 复用） ----------
def load_samples(patterns, ckpt_dir):
    from alphazero_nn import expand_game_records
    samples = []
    for pat in patterns:
        for fn in sorted(globmod.glob(str(Path(ckpt_dir) / pat))):
            with open(fn, "rb") as f:
                data = pickle.load(f)
            if data and isinstance(data, list) and "steps" in data[0]:
                data = expand_game_records(data)  # 完整棋谱 → 训练样本
            for smp in data:
                if smp.get("z", 1) == 0.0:
                    smp["vw"] = 0.0  # z=0 样本（开局库）只训 policy，防 value head 腐蚀
            samples += data
    return samples


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="棋谱 pkl glob 模式")
    ap.add_argument("--warm", default=None, help="warm-start numpy pkl（NetParams）")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="tt")
    args = ap.parse_args()
    HERE = Path(__file__).parent / "nn_checkpoints"
    samples = load_samples(args.data, HERE)
    print(f"[TORCH-TRAIN] 样本 {len(samples)}", flush=True)
    if args.warm:
        model = convert_numpy_to_torch(HERE / args.warm if not os.path.isabs(args.warm) else args.warm)
    else:
        model = NetTorch()

    def cb(step, m):
        if step % 200 == 0 or step == args.steps:
            np_p = m.to_numpy_params()
            np_p.save(HERE / f"params_{args.tag}_ckpt_{step:04d}.pkl")
            m.save(HERE / f"params_{args.tag}_ckpt_{step:04d}.pt")

    train_bc_torch(samples, model, steps=args.steps, lr=args.lr, bs=args.bs,
                   seed=args.seed, ckpt_cb=cb)
    np_p = model.to_numpy_params()
    np_p.save(HERE / f"params_{args.tag}_last.pkl")
    model.save(HERE / f"params_{args.tag}_last.pt")
    print(f"[TORCH-TRAIN] 完成 → params_{args.tag}_last.pkl/.pt", flush=True)
