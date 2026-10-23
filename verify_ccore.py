"""C 引擎四步等价性验证：ccore（C）vs Python 基准

用法：
    python3 verify_ccore.py           # 全部四步
    python3 verify_ccore.py 1 2       # 只跑指定步骤

步骤：
    1. 走法集合   200 局随机局面，C 的 (start,end) 集合 == Python 的 (pos, path[-1]) 集合
    2. 评估值     200 局 x 3 版本(v0/v2/v8)，C 的 float == Python 的 float（误差 < 1e-9）
    3. 最优走法   20 局固定 depth=4 / null_r=0，C 与 Python 选出的 (start,end) 必须一致
    4. 速度       初始局面 1s 限时，比较 nodes/s 与达到深度
"""
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import Chinese_Checker as cc   # noqa: E402
import engine                  # noqa: E402
import ccore                   # noqa: E402

random.seed(42)


def random_board(n_steps):
    """从初始局面随机走 n_steps 步，返回 (board, 下一步该谁走)。"""
    b = cc.init_board()
    player = 1
    for _ in range(n_steps):
        moves = cc.get_all_moves(b, player)
        if not moves:
            break
        _, path = random.choice(moves)
        b = cc.apply_move(b, path)
        player = 3 - player
    return b, player


def step1_move_sets(trials=200):
    print("=== 1. 走法集合 ===")
    fails = 0
    for t in range(trials):
        b, p = random_board(random.randint(0, 40))
        py_set = set((pos, path[-1]) for pos, path in cc.get_all_moves(b, p))
        c_set = set(tuple(x) for x in ccore.cc_gen_moves(b, p))
        if py_set != c_set:
            fails += 1
            if fails <= 3:
                print(f"  FAIL trial={t} player={p}")
                print(f"    only_py={sorted(py_set - c_set)}")
                print(f"    only_c ={sorted(c_set - py_set)}")
    print(f"  {trials} 局，不一致: {fails}")
    return fails == 0


def step2_eval(trials=200):
    print("=== 2. 评估值 ===")
    cases = [("v0", 0, engine.evaluate_v0),
             ("v2", 2, engine.evaluate_v2),
             ("v8", 8, engine.evaluate_v8)]
    fails = 0
    for t in range(trials):
        b, p = random_board(random.randint(0, 40))
        for name, vid, fn in cases:
            py_v = fn(b, p)
            c_v = ccore.cc_eval(b, p, vid)
            if abs(py_v - c_v) > 1e-9:
                fails += 1
                if fails <= 5:
                    print(f"  FAIL trial={t} {name} py={py_v} c={c_v} diff={abs(py_v - c_v)}")
    print(f"  {trials} 局 x 3 版本，差异>1e-9: {fails}")
    return fails == 0


def step3_best_move(trials=20, depth=4):
    print(f"=== 3. 固定深度最优走法 (depth={depth}, null_r=0) ===")
    from engine_pvs import PVSIDSEngine
    fails = 0
    for t in range(trials):
        b, p = random_board(random.randint(0, 20))
        eng = PVSIDSEngine(time_limit_sec=100, max_depth=depth,
                           eval_version="v2", null_move_r=0)
        path, info = eng.think(b, p)
        if not path:
            continue
        py_move = (path[0], path[-1])
        r = ccore.cc_search(b, p, 100.0, depth, 2, 0)
        c_move = (r[0], r[1])
        if py_move != c_move:
            fails += 1
            if fails <= 3:
                print(f"  FAIL trial={t} player={p}")
                print(f"    py={py_move} score={info.get('score')} nodes={info.get('nodes_searched')}")
                print(f"    c ={c_move} score={r[2]} nodes={r[4]}")
    print(f"  {trials} 局，最优走法不一致: {fails}")
    return fails == 0


def step4_speed(seconds=1.0):
    print(f"=== 4. 速度 ({seconds}s，初始局面) ===")
    from engine_pvs import PVSIDSEngine
    b = cc.init_board()

    eng = PVSIDSEngine(time_limit_sec=seconds, max_depth=20,
                       eval_version="v2", null_move_r=2)
    t0 = time.time()
    _, info = eng.think(b, 1)
    t_py = time.time() - t0
    n_py = info.get("nodes_searched", 0)
    print(f"  Python: {n_py} 节点 depth={info.get('depth_reached')} "
          f"{t_py*1000:.0f}ms {n_py/max(t_py,1e-9):.0f} nodes/s")

    t0 = time.time()
    r = ccore.cc_search(b, 1, seconds, 20, 2, 2)
    t_c = time.time() - t0
    print(f"  C     : {r[4]} 节点 depth={r[3]} "
          f"{t_c*1000:.0f}ms {r[4]/max(t_c,1e-9):.0f} nodes/s")
    print(f"  提速: {(r[4]/max(t_c,1e-9))/max(n_py/max(t_py,1e-9),1e-9):.1f}x")


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["1", "2", "3", "4"]
    ok = True
    if "1" in wanted:
        ok &= step1_move_sets()
    if "2" in wanted:
        ok &= step2_eval()
    if "3" in wanted:
        ok &= step3_best_move()
    if "4" in wanted:
        step4_speed()
    print("\nRESULT:", "ALL PASS" if ok else "HAS FAILURES")
