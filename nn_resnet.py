"""残差网络版 AlphaZero CNN：stem + N个残差块，policy/value 双头。

对比原 2 层 conv：深度 10 层（stem + 4×2 conv），filters=128，参数量约 2 倍。
残差连接缓解深层梯度消失，BN 加速收敛。
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
                          H, W, CIN)

cc = engine.cc


class ResBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class NetResNet(nn.Module):
    def __init__(self, filters=128, blocks=4):
        super().__init__()
        self.filters = filters
        self.blocks = blocks
        self.stem = nn.Conv2d(CIN, filters, 3, padding=1)
        self.resblocks = nn.ModuleList([ResBlock(filters) for _ in range(blocks)])
        # policy head
        self.p_conv = nn.Conv2d(filters, 32, 1)
        self.p_bn = nn.BatchNorm2d(32)
        self.p_fc1 = nn.Linear(H * W * 32, 64)
        self.p_fc2 = nn.Linear(64, 162)
        # value head
        self.v_conv = nn.Conv2d(filters, 32, 1)
        self.v_bn = nn.BatchNorm2d(32)
        self.v_fc1 = nn.Linear(H * W * 32, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = F.relu(self.stem(x))
        for block in self.resblocks:
            h = block(h)
        # policy
        p = F.relu(self.p_bn(self.p_conv(h)))
        p = p.flatten(1)
        p = F.relu(self.p_fc1(p))
        plogits = self.p_fc2(p)
        # value
        v = F.relu(self.v_bn(self.v_conv(h)))
        v = v.flatten(1)
        v = F.relu(self.v_fc1(v))
        value = torch.tanh(self.v_fc2(v)).squeeze(-1)
        return plogits, value

    def save(self, path):
        torch.save({"state_dict": self.state_dict(),
                    "filters": self.filters, "blocks": self.blocks}, path)

    @staticmethod
    def load(path):
        d = torch.load(path, map_location="cpu", weights_only=True)
        m = NetResNet(filters=d["filters"], blocks=d["blocks"])
        m.load_state_dict(d["state_dict"])
        return m


def load_samples(patterns, ckpt_dir):
    from alphazero_nn import expand_game_records
    samples = []
    for pat in patterns:
        for fn in sorted(globmod.glob(str(Path(ckpt_dir) / pat))):
            with open(fn, "rb") as f:
                data = pickle.load(f)
            if data and isinstance(data, list) and "steps" in data[0]:
                data = expand_game_records(data)
            for smp in data:
                if smp.get("z", 1) == 0.0:
                    smp["vw"] = 0.0
            samples += data
    return samples


def _make_batch(batch, device):
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
            dst_idx[i, j] = d + 81
        pi[i, :len(ms)] = np.asarray(rec["pi"], dtype=np.float32)
        mask[i, :len(ms)] = 1.0
        z[i] = rec["z"]
        vw[i] = rec.get("vw", 1.0)
    t = lambda a: torch.from_numpy(a).to(device)
    return t(xs), t(src_idx), t(dst_idx), t(pi), t(mask), t(z), t(vw)


def train(samples, model, steps=4000, lr=2e-4, bs=32, weight_decay=1e-4,
          seed=0, log_every=200, ckpt_cb=None, device="cpu"):
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
            print(f"[RESNET] step {step+1}/{steps} loss={last_loss:.4f} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
        if ckpt_cb:
            ckpt_cb(step + 1, model)
    return last_loss


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--filters", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, default="nn_checkpoints")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.data, ckpt_dir)
    print(f"[RESNET] 样本数={len(samples)}", flush=True)

    model = NetResNet(filters=args.filters, blocks=args.blocks)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[RESNET] filters={args.filters} blocks={args.blocks} params={n_params/1e6:.2f}M",
          flush=True)

    best_loss = [float("inf")]

    def ckpt_cb(step, m):
        if step % 500 == 0:
            m.save(ckpt_dir / f"params_resnet_{args.tag}_step{step}.pt")

    loss = train(samples, model, steps=args.steps, lr=args.lr, bs=args.bs,
                 seed=args.seed, ckpt_cb=ckpt_cb)
    out = ckpt_dir / f"params_resnet_{args.tag}_last.pt"
    model.save(out)
    print(f"[RESNET] done loss={loss:.4f} -> {out}", flush=True)
