"""跨引擎对战：我们的 V2 (AlphaBeta+启发式) vs clysto/chinese-checkers (MTD(f)+开局库)。

clysto 项目: /Users/liuzirui/Desktop/py/clysto-cc (C++, 已编译 cc_cli)
棋盘编号不同，必须做坐标映射：
  - 他是 9x9 轴坐标菱形 (p%9, p//9)，行 r = x+y，共 17 行 (1..9..1)
  - 我们是 17 行星形，行内从左到右编号
  - 他的 GREEN('2')=顶部三角 ↔ 我们 P1(0-9)；他的 RED('1')=底部三角 ↔ 我们 P2(71-80)
映射在本脚本开头自动推导并验证（邻接同构 + 初始区一致），失败则报错退出。

CLI 协议: stdin 发 "SEARCH <81字符state><r|g>"，stdout 收 "<src> <dst>"
"""
from checkpoint_paths import artifact_path, artifact_glob
import os
import re
import sys
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
from engine import AlphaBetaIDSEngine

cc = engine.cc

# 依赖外部引擎 clysto/chinese-checkers（C++）：需自行克隆并编译出 cc_cli。
# 默认路径是本机；其他机器用环境变量覆盖：CLYSTO_DIR=/path/to/clysto-cc python3 arena_vs_clysto.py
CLYSTO_DIR = os.environ.get("CLYSTO_DIR", "/Users/liuzirui/Desktop/py/clysto-cc")
CLI_PATH = os.path.join(CLYSTO_DIR, "cc_cli")


# ---------- 坐标映射 ----------
def build_mapping(mirror):
    """his p -> our pos。行 r=x+y 对应我们的行 r，行内按序号对齐；mirror=左右镜像版。"""
    m = {}
    for p in range(81):
        x, y = p % 9, p // 9
        r = x + y
        xmin = max(0, r - 8)
        xmax = min(8, r)
        width = xmax - xmin + 1
        k = x - xmin
        if mirror:
            k = width - 1 - k
        if r <= 8:
            s = r * (r + 1) // 2
        else:
            s = 81 - (17 - r) * (18 - r) // 2
        m[p] = s + k
    return m


def load_clysto_adj():
    """从 constants.hpp 解码 ADJ_POSITIONS bitmask -> 81 个邻接集合。"""
    text = open(os.path.join(CLYSTO_DIR, "src", "constants.hpp")).read()
    m = re.search(r"ADJ_POSITIONS\[81\] = \{(.*?)\n\};", text, re.S)
    body = m.group(1)
    masks = re.findall(r"\(\(uint128_t\)0x([0-9a-fA-F]+) << 64\) \| 0x([0-9a-fA-F]+)", body)
    adj = []
    for hi, lo in masks:
        v = (int(hi, 16) << 64) | int(lo, 16)
        adj.append({i for i in range(81) if (v >> i) & 1})
    if len(adj) != 81:
        raise RuntimeError(f"ADJ_POSITIONS 解析出 {len(adj)} 项, 应为 81")
    return adj


def verify_mapping():
    his_adj = load_clysto_adj()
    for mirror in (False, True):
        M = build_mapping(mirror)
        ok = all({M[q] for q in his_adj[p]} == set(cc.NEIGHBORS_81[M[p]]) for p in range(81))
        print(f"[MAP] mirror={mirror}: 邻接同构 {'OK' if ok else 'FAIL'}")
        if not ok:
            continue
        green = {0, 1, 2, 3, 9, 10, 11, 18, 19, 27}
        red = {53, 61, 62, 69, 70, 71, 77, 78, 79, 80}
        if {M[p] for p in green} != set(range(10)):
            raise RuntimeError("his GREEN 没映射到我们的 P1 区 0-9")
        if {M[p] for p in red} != set(range(71, 81)):
            raise RuntimeError("his RED 没映射到我们的 P2 区 71-80")
        print("[MAP] 初始三角区映射 OK (GREEN->P1区, RED->P2区)")
        return M
    raise RuntimeError("两种镜像都未通过邻接同构验证, 映射失败")


MAP_H2O = verify_mapping()
MAP_O2H = [0] * 81
for p in range(81):
    MAP_O2H[MAP_H2O[p]] = p


# ---------- clysto 引擎适配器 ----------
class ClystoEngine:
    name = "clysto-cc"

    def __init__(self, timelimit=1):
        self.timelimit = timelimit
        self.proc = subprocess.Popen(
            [CLI_PATH], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, cwd=CLYSTO_DIR,
        )
        # clysto CLI 用 int 解析 TIMELIMIT，"0.5" 会被截断成 0（等于不搜索）。
        # 这里强制至少 1 秒，保证棋力。
        self._send(f"TIMELIMIT {max(1, int(round(timelimit)))}")
        resp = self.proc.stdout.readline().strip()
        if resp != "ok":
            raise RuntimeError(f"TIMELIMIT 响应异常: {resp}")
        print(f"[CLY] clysto 引擎已启动 (timelimit={timelimit}s)")

    def _send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def state_str(self, board, player):
        """我们的 board/player -> 他的 state 字符串。
        我们 P1 -> 他的 GREEN('2')；我们 P2 -> 他的 RED('1')；turn: P1='g', P2='r'
        """
        chars = ["0"] * 81
        for p in range(81):
            if board[p] == 1:
                chars[MAP_O2H[p]] = "2"
            elif board[p] == 2:
                chars[MAP_O2H[p]] = "1"
        turn = "g" if player == 1 else "r"
        return "".join(chars) + turn

    def think(self, board, player):
        state = self.state_str(board, player)
        self._send("SEARCH " + state)
        line = self.proc.stdout.readline().strip()
        if not line:
            raise RuntimeError("clysto CLI 无响应(进程可能已退出)")
        src, dst = map(int, line.split())
        osrc, odst = MAP_H2O[src], MAP_H2O[dst]
        for s, path in cc.get_all_moves(board, player):
            if s == osrc and path[-1] == odst:
                return path, {"engine": self.name, "his_raw": (src, dst)}
        raise RuntimeError(
            f"clysto 走法在我们规则下非法: his({src}->{dst}) = our({osrc}->{odst})")

    def quit(self):
        try:
            self._send("QUIT")
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()


# ---------- 对战 ----------
def play_one_game(our_eng, opp_eng, our_player, max_moves=250):
    board = cc.init_board()
    move_count = 0
    while True:
        w = cc.is_game_over(board)
        if w is not None:
            return w, move_count
        player = 1 if move_count % 2 == 0 else 2
        if player == our_player:
            path, _ = our_eng.think(board, player)
        else:
            path, _ = opp_eng.think(board, player)
        if path is None:
            return 3 - player, move_count
        board = cc.apply_move(board, path)
        move_count += 1
        if move_count >= max_moves:
            break
    s1 = engine.evaluate_v2(board, 1)
    s2 = engine.evaluate_v2(board, 2)
    return (1 if s1 >= s2 else 2), move_count


def main(n_games=10, our_time=1.0, cly_time=1, our_engine="v2",
         ckpt=None, n_simulations=32):
    if our_engine == "c":
        from engine_c import CESEngine
        our = CESEngine(time_limit_sec=our_time, max_depth=20, eval_version="v2")
    elif our_engine == "nn":
        from alphazero_nn import NetParams, NNEasyEngine
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "nn_checkpoints")
        ckpt_path = ckpt or artifact_path(os.path.join(save_dir, "params_bc_latest.pkl"))
        params = NetParams.load(ckpt_path)
        our = NNEasyEngine(params, use_mcts=(n_simulations > 0),
                           n_simulations=max(n_simulations, 1))
        our.name = f"NN({os.path.basename(ckpt_path)})"
        print(f"[MATCH] 加载 NN 参数 {ckpt_path}")
    else:
        our = AlphaBetaIDSEngine(time_limit_sec=our_time, max_depth=8,
                                 eval_version="v2")
        our.name = "V2"
    cly = ClystoEngine(timelimit=cly_time)

    print(f"[MATCH] {our.name} vs {cly.name}: {n_games} 局, 双方各 {our_time}s/{cly_time}s, 交替先后手")
    our_wins = cly_wins = draws = 0
    total_moves = 0
    t0 = time.time()
    for g in range(n_games):
        our_player = 1 if g % 2 == 0 else 2
        side = "P1(先)" if our_player == 1 else "P2(后)"
        t_g = time.time()
        try:
            winner, moves = play_one_game(our, cly, our_player)
        except RuntimeError as e:
            print(f"[MATCH] 局 {g+1}: 桥接错误! {e}")
            break
        total_moves += moves
        if winner == our_player:
            our_wins += 1
            res = f"{our.name}胜"
        elif winner == 3 - our_player:
            cly_wins += 1
            res = f"{cly.name}胜"
        else:
            draws += 1
            res = "平"
        print(f"[MATCH] 局 {g+1}/{n_games} 我方走{side} → {res}, 步数={moves}, "
              f"用时 {(time.time()-t_g)/60:.1f}min, 累计 {our.name}:{cly.name} = {our_wins}:{cly_wins} (平 {draws})",
              flush=True)

    print(f"[MATCH] === 最终结果 ===")
    print(f"[MATCH] {our.name}:{cly.name} = {our_wins}:{cly_wins} (平 {draws})")
    print(f"[MATCH] 平均步数 {total_moves / max(1, our_wins + cly_wins + draws):.0f}")
    print(f"[MATCH] 总用时 {(time.time() - t0) / 60:.1f} 分钟")
    cly.quit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games", type=int, default=10)
    parser.add_argument("--our_time", type=float, default=1.0)
    parser.add_argument("--cly_time", type=int, default=1)
    parser.add_argument("--our_engine", choices=["v2", "nn", "c"], default="v2",
                        help="我方引擎：v2=启发式搜索，nn=神经网络")
    parser.add_argument("--ckpt", default=None,
                        help="nn 模式下的参数文件路径，默认 nn_checkpoints/params_bc_latest.pkl")
    parser.add_argument("--n_simulations", type=int, default=32,
                        help="nn 模式 MCTS 模拟次数，0=纯 policy argmax")
    args = parser.parse_args()
    main(args.n_games, args.our_time, args.cly_time, args.our_engine,
         args.ckpt, args.n_simulations)
