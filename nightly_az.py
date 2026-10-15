"""通宵 AlphaZero 自对弈训练循环（8小时无人值守）。

每轮:
  1. 6 进程并行自对弈（当前冠军 NN+MCTS, n_sim=24, 每步按访问分布采样）
  2. torch 快速训练 3 种子: 专家15k + 开局库×3 + 最近3轮自对弈数据（滚动窗口）
  3. 串行评估 3 个种子 vs v0d4 (20局, n_opening=10)，实战胜率选新冠军
  4. 新冠军写 params_az_champ.pkl，删除最老自对弈轮

日志: /tmp/night/az_loop.log   结果汇总: /tmp/night/az_results.txt
"""
import os
import subprocess
import sys
import time
from pathlib import Path

PROJ = "/Users/liuzirui/Desktop/py/chinese-checkers"
CKPT = Path(PROJ) / "nn_checkpoints"
PY = sys.executable
LOG = open("/tmp/night/az_loop.log", "a", buffering=1)
RES = open("/tmp/night/az_results.txt", "a", buffering=1)
DEADLINE = time.time() + 7.8 * 3600  # 8小时留余量
N_SHARDS = 6
GAMES_PER_SHARD = 80
EVAL_SEED = 999  # 每轮更新
CHAMP = "params_az_champ.pkl"

EXP = "replay_clysto_merged.pkl"
BOOK = "replay_book.pkl"


def run(cmd):
    LOG.write(f"$ {cmd}\n")
    p = subprocess.run(cmd, shell=True, cwd=PROJ, stdout=LOG, stderr=subprocess.STDOUT)
    LOG.write(f"-- exit {p.returncode}\n")
    return p.returncode


def eval_game(ckpt, tag):
    """串行评估 20 局，返回 NN 胜局数。seed 每轮轮换，防止在同一固定开局集上反复选最大导致过拟合虚高。"""
    out = subprocess.run(
        f"{PY} evaluate_nn_vs_opponent.py --ckpt {ckpt} --opponent v0d4 "
        f"--n_games 20 --n_opening 10 --seed {EVAL_SEED}",
        shell=True, cwd=PROJ, capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if "NN:V0d4 = " in l]
    score = line[-1] if line else "ERR " + out.stderr[-200:]
    RES.write(f"{tag} {ckpt} -> {score}\n")
    LOG.write(f"[EVAL] {tag} {ckpt} -> {score}\n")
    try:
        return int(score.split("= ")[1].split(":")[0])
    except Exception:
        return -1


# 初始冠军 = torch 冒烟模型（8:12 实测）
if not (CKPT / CHAMP).exists():
    subprocess.run(["cp", str(CKPT / "params_tsmoke_last.pkl"), str(CKPT / CHAMP)])

rnd = 0
while time.time() < DEADLINE:
    rnd += 1
    t0 = time.time()
    globals()["EVAL_SEED"] = 4321 + rnd * 17  # 每轮换开局集
    LOG.write(f"\n===== ROUND {rnd} start {time.strftime('%F %T')} champ={CHAMP} =====\n")

    # 1. 并行自对弈
    procs = [subprocess.Popen(
        [PY, "az_selfplay.py", "--ckpt", CHAMP, "--n_games", str(GAMES_PER_SHARD),
         "--n_sim", "24",
         "--seed", str(rnd * 100 + i), "--out", f"az_r{rnd}_{i}.pkl"],
        cwd=PROJ, stdout=LOG, stderr=subprocess.STDOUT) for i in range(N_SHARDS)]
    for p in procs:
        p.wait()
    LOG.write(f"[R{rnd}] selfplay done {time.time()-t0:.0f}s\n")

    # 2. 训练数据 = 专家 + 开局×3 + 最近3轮自对弈
    rounds = [r for r in range(max(1, rnd - 2), rnd + 1)]
    az_globs = " ".join(f"az_r{r}_*.pkl" for r in rounds)
    for f in CKPT.glob("az_r*_*.pkl"):
        r_old = int(f.stem.split("_")[1][1:])
        if r_old < rnd - 2:
            f.unlink()  # 滚动窗口：只保留最近3轮自对弈数据

    wins = {}
    for seed in (7, 11, 23):
        tag = f"az{rnd}s{seed}"
        run(f"{PY} nn_torch.py --data {EXP} {BOOK} {BOOK} {BOOK} {az_globs} "
            f"--warm {CHAMP} --steps 2000 --lr 2e-4 --seed {seed} --tag {tag}")
        wins[seed] = eval_game(f"params_{tag}_last.pkl", f"R{rnd}")

    # 3. 选冠军（实战胜率；同分保留旧冠军）
    best_seed = max(wins, key=lambda s: wins[s])
    best_ckpt = f"params_az{rnd}s{best_seed}_last.pkl"
    if wins[best_seed] > eval_game(CHAMP, f"R{rnd}-base"):
        subprocess.run(["cp", str(CKPT / best_ckpt), str(CKPT / CHAMP)])
        RES.write(f"ROUND {rnd}: NEW CHAMP {best_ckpt} ({wins[best_seed]} wins)\n")
        LOG.write(f"[R{rnd}] NEW CHAMP {best_ckpt} {wins[best_seed]}\n")
    else:
        RES.write(f"ROUND {rnd}: keep champ (best new {best_ckpt}={wins[best_seed]})\n")
        LOG.write(f"[R{rnd}] keep champ, best new {wins[best_seed]}\n")
    LOG.write(f"===== ROUND {rnd} done {time.time()-t0:.0f}s =====\n")

RES.write(f"===== LOOP END after {rnd} rounds {time.strftime('%F %T')} =====\n")
