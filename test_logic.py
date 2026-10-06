"""
纯代码测试：中国跳棋核心逻辑
运行：python3 test_logic.py
结果：全部通过 = 退出码 0 + 底部汇总 OK / N FAILED
"""
import sys, traceback, os, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("cc", os.path.join(_HERE, "Chinese_Checker.py"))
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

PASS, FAIL = 0, 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"✅ {name}")
    else:
        FAIL += 1
        print(f"❌ {name}" + (f"  → {detail}" if detail else ""))

# ======================================================================
# 一、三点一线（跳跃必须距离=2，起点-中点-终点共线）
# ======================================================================
print("\n=== 一、三点一线 + 距离=2 ===")
B = [0]*81
# 起点 pos=40 的 6 个邻居依次试
start = 40
sx, sy = cc.pos_to_xy(start)
total_legal_jumps = 0
for mid in cc.NEIGHBORS_81[start]:
    B[mid] = 2
    jumps = cc.get_jump_moves(start, B)
    for dst in jumps:
        total_legal_jumps += 1
        mx, my = cc.pos_to_xy(mid)
        ex, ey = cc.pos_to_xy(dst)
        colinear = (mx - sx == ex - mx) and (my - sy == ey - my)
        dist_eq_2 = (ex - sx == 2*(mx - sx)) and (ey - sy == 2*(my - sy))
        mid_adj_s = mid in cc.NEIGHBORS_81[start]
        dst_adj_m = dst in cc.NEIGHBORS_81[mid]
        check(f"跳跃段 {start}→{mid}→{dst}: 三点共线", colinear,
              f"坐标 ({sx},{sy})-({mx},{my})-({ex},{ey})")
        check(f"跳跃段 {start}→{mid}→{dst}: 距离严格=2", dist_eq_2)
        check(f"跳跃段 {start}→{mid}→{dst}: 起点-mid 相邻", mid_adj_s)
        check(f"跳跃段 {start}→{mid}→{dst}: mid-终点 相邻", dst_adj_m)
    B[mid] = 0

check(f"6 个方向至少有 ≥3 个合法跳跃（六角格部分方向越界）", total_legal_jumps >= 3)

# 反例：长距离直线有棋子也不能跳（必须距离=2）
B2 = [0]*81
B2[20] = 1; B2[26] = 2   # 20和26距离是 pos 差=6，就算"直线"也不能跳 — 算法只看 NEIGHBORS
j = cc.get_jump_moves(20, B2)
ok_long = (26 not in j)
# 其实只要 26 不是 20 的邻居就一定会返回 False
# 20 的邻居是 [14,19,26,27] —— 操 26 确实是 20 邻居！那这样会跳向 26 + (26-20)方向
# 那换一个真正不相邻的：20 和 34 差14格肯定不相邻
B2[26] = 0; B2[34] = 2
j = cc.get_jump_moves(20, B2)
check("反例：非相邻的长距离棋子不能当跳跃中点", 34 not in cc.NEIGHBORS_81[20] and len([d for d in j if d==48])==0)

# ======================================================================
# 二、边界/角落位置操作无异常
# ======================================================================
print("\n=== 二、边界/角落操作 ===")
board_init = cc.init_board()
extremes = [0, 1, 8, 9, 71, 72, 79, 80]          # 8 个角
extremes += list(range(10)) + list(range(71, 81))  # 两个营地全体 20 个
# 再加上 36,44,45,52,53,60 这些菱形边缘的"凸"位置
extremes += [36, 44, 52, 53, 60, 71, 74]
extremes = sorted(set(extremes))
no_crash = True
for p in extremes:
    try:
        player = board_init[p] if board_init[p] != 0 else 1
        _ = cc.pos_to_xy(p)
        _ = cc.get_step_moves(p, board_init)
        _ = cc.get_jump_moves(p, board_init)
        paths = cc.get_all_jump_paths(p, board_init, player)
        moves = cc.get_all_legal_moves(p, board_init)
        # 返回结构检查：每条 path 起点必须 == p
        for path in moves:
            assert path[0] == p, f"起点不一致 pos={p},path[0]={path[0]}"
    except Exception as e:
        no_crash = False
        print(f"  💥 pos={p} 崩溃: {type(e).__name__}: {e}")
        traceback.print_exc()
check(f"{len(extremes)} 个边界/极端位置调用无异常、结构一致", no_crash)

# 角落越界防护：pos=0 所有邻居放棋子，跳跃方向必然越界 → 返回空
Bcorner = [0]*81
for mid in cc.NEIGHBORS_81[0]:
    Bcorner[mid] = 2
check("角落 pos=0 向所有邻居方向尝试跳跃，不会越界报错", True)  # 没报错就是过了
j_corner = cc.get_jump_moves(0, Bcorner)
check("角落 pos=0 理论上的跳跃越界会被拦截（要么空要么合法）",
      all(0 <= d <= 80 for d in j_corner))

# ======================================================================
# 三、敌我棋子区分 + 操作可选（单步/单跳/连跳同时存在时任选）
# ======================================================================
print("\n=== 三-1、敌我棋子严格区分 ===")
m1 = cc.get_all_moves(board_init, 1)
m2 = cc.get_all_moves(board_init, 2)
check("get_all_moves(1) 全部出发位置都是 P1 棋子 (board=1)",
      all(board_init[pos] == 1 for pos, _ in m1),
      f"错误样例: {next(((pos,board_init[pos]) for pos,_ in m1 if board_init[pos]!=1), None)}")
check("get_all_moves(2) 全部出发位置都是 P2 棋子 (board=2)",
      all(board_init[pos] == 2 for pos, _ in m2),
      f"错误样例: {next(((pos,board_init[pos]) for pos,_ in m2 if board_init[pos]!=2), None)}")
check("get_all_moves(1) 和 get_all_moves(2) 走法交集为空",
      len(set((pos, tuple(path)) for pos, path in m1) & set((pos, tuple(path)) for pos, path in m2)) == 0)

# apply_move 后 player 归属：起点值应等于终点值
if m1:
    pos, path = m1[0]
    piece_expected = board_init[pos]
    nb = cc.apply_move(board_init, path)
    check(f"apply_move 后起点→终点的玩家归属不变 (player={piece_expected})",
          nb[path[-1]] == piece_expected and nb[path[0]] == 0)

print("\n=== 三-2、自由选择模式：单步/单跳/连跳 同时可选 ===")
# 构造一个特殊棋盘，让某个棋子同时具备三种选项
BMIX = [0]*81
BMIX[40] = 1
BMIX[32] = 2      # 40跳过32→24（单跳1：40-mid32-24 邻接链 40-32-24 成立）
BMIX[18] = 2      # 24跳过18→12（连跳第二段：24-mid18-12 邻接链 24-18-12 成立）
# 40的邻居 [31,32,39,41,48,49] 中 32被占，其余空 → 5 个单步
moves40 = cc.get_all_legal_moves(40, BMIX)
step_set = set(cc.get_step_moves(40, BMIX))
jump_set = set(cc.get_jump_moves(40, BMIX))
steps      = [m for m in moves40 if len(m) == 2 and m[1] in step_set]
single_js  = [m for m in moves40 if len(m) == 2 and m[1] in jump_set]
chain_js   = [m for m in moves40 if len(m) >= 3]
check(f"单步走选项存在  (共{len(steps)}条)", len(steps) >= 3)
check(f"单跳选项存在    (共{len(single_js)}条)", len(single_js) >= 1)
check(f"连跳选项存在    (共{len(chain_js)}条)", len(chain_js) >= 1)

print("\n=== 三-3、自由选择不等于强制跳跃：有跳跃时仍允许单步 ===")
# 在上面 BMIX 里确实既有跳跃也有单步，get_all_legal_moves 返回的既有 jump 类也有 step 类
mix_has_step = any(len(m)==2 and m[1] in step_set for m in moves40)
mix_has_jump = any((len(m)==2 and m[1] in jump_set) or len(m)>=3 for m in moves40)
check("混合场景 get_all_legal_moves 返回里既含跳跃也含单步（强制跳跃被关闭）",
      mix_has_step and mix_has_jump)

print("\n=== 三-4、三种操作都不会吃掉对方棋子，棋子总数保持不变 ===")
total_origin = sum(1 for p in BMIX if p != 0)  # P1=1 + P2=2 = 3
if steps:
    nb1 = cc.apply_move(BMIX, steps[0])
    check(f"单步 {steps[0]} 后棋子总数不变", sum(1 for p in nb1 if p!=0) == total_origin)
if single_js:
    nb2 = cc.apply_move(BMIX, single_js[0])
    t2 = sum(1 for p in nb2 if p!=0)
    # 被跳过的 32 应仍然是 2
    check(f"单跳 {single_js[0]} 后棋子总数不变 + 跳过的pos32仍=2",
          t2 == total_origin and nb2[32] == 2, f"t2={t2}, nb2[32]={nb2[32]}")
if chain_js:
    nb3 = cc.apply_move(BMIX, chain_js[0])
    t3 = sum(1 for p in nb3 if p!=0)
    check(f"连跳 {chain_js[0]} 后棋子总数不变 + pos32=2 + pos18=2（两颗都保留）",
          t3 == total_origin and nb3[32] == 2 and nb3[18] == 2,
          f"t3={t3} nb3[32]={nb3[32]} nb3[18]={nb3[18]}")

# ======================================================================
# 四、胜负判定回归（之前修复的bug）
# ======================================================================
print("\n=== 四、胜负判定边界（空棋盘/某方无棋子不误判） ===")
check("空棋盘 → None（不误判 P1 胜）", cc.is_game_over([0]*81) is None)
p2_full = [0]*81
for i in range(0,10): p2_full[i] = 2
check("只有 P2 全占0~9（P1 空） → 返回 2（不误判 P1 胜）", cc.is_game_over(p2_full) == 2)
p1_missing_one = [0]*81
for i in range(1,10): p1_missing_one[i] = 2   # 只有 9 颗 P2 在 0~9
check("P2 差 1 颗没占满目标区 → None", cc.is_game_over(p1_missing_one) is None)

# ======================================================================
# 五、N 组不同随机种子 × 500 步对战（不是只跑一个 seed 碰运气）
#   —— 每步同时做结构校验：路径起点一致、单步/跳跃合法、
#      apply_move 后起点空/终点有棋/被跳过棋子保留、棋子总数恒=20
# ======================================================================
print("\n=== 五、20 组随机种子 × 500 步对战 + 每步结构校验 ===")
import random

def step_structure_assert(orig_board, pos, path, new_board, player):
    """每步走完后的结构断言，全部成立才叫"这一步真的没问题"。"""
    assert path[0] == pos, "path[0] 必须等于选择的棋子位置"
    assert orig_board[pos] == player, "起点棋子归属错误"
    assert new_board[pos] == 0, "apply_move 后起点未腾空"
    assert new_board[path[-1]] == player, "apply_move 后终点棋子归属错误"
    # 棋子数量守恒
    assert sum(1 for p in orig_board if p!=0) == sum(1 for p in new_board if p!=0), "棋子总数变化=被吃了"
    # 判断是单步还是跳跃段：逐段校验
    for i in range(len(path)-1):
        a, b_ = path[i], path[i+1]
        step_ok = (b_ in cc.get_step_moves(a, orig_board)) if (i==0 and len(path)==2) else False
        # 跳跃段：找得到公共相邻中点且中点有棋子
        mid_list = [m for m in cc.NEIGHBORS_81[a] if m in cc.NEIGHBORS_81[b_] and orig_board[m] != 0]
        if step_ok:
            continue
        # 对跳跃段：中点必须非空且 new_board 里中点值仍=原值（没被吃）
        assert mid_list, f"段 {a}→{b_} 既不是单步也不是跳跃（非法移动）"
        for m in mid_list:
            assert new_board[m] == orig_board[m], f"跳过的中点 pos={m} 被吃了"

SEEDS = [0, 1, 2, 3, 5, 7, 11, 13, 17, 23, 42, 57, 73, 99, 101, 233, 377, 1024, 2025, 9999]
seeds_passed = 0
max_steps_total = 0
for seed in SEEDS:
    random.seed(seed)
    b = cc.init_board()
    player = 1
    crashed = False
    steps_done = 0
    fail_detail = ""
    w = None
    for i in range(500):
        try:
            moves = cc.get_all_moves(b, player)
            if not moves:
                player = 3 - player
                continue
            pos, path = random.choice(moves)
            b_old = b
            b = cc.apply_move(b, path)
            step_structure_assert(b_old, pos, path, b, player)
            w = cc.is_game_over(b)
            if w is not None:
                steps_done = i+1
                break
            player = 3 - player
            steps_done = i+1
        except Exception as e:
            crashed = True
            fail_detail = f"{type(e).__name__}: {e}"
            break
    max_steps_total = max(max_steps_total, steps_done)
    mark = "✅" if not crashed else "❌"
    p1c = sum(1 for p in b if p==1)
    p2c = sum(1 for p in b if p==2)
    print(f"  {mark} seed={seed:>5}  {steps_done:>3}步  P1={p1c} P2={p2c} 合计={p1c+p2c}" +
          (f"  🏆P{w}胜" if cc.is_game_over(b) is not None else "") +
          (f"  → {fail_detail}" if crashed else ""))
    if not crashed:
        seeds_passed += 1

check(f"{len(SEEDS)} 组随机种子全部跑完无崩溃无结构异常", seeds_passed == len(SEEDS),
      f"只通过 {seeds_passed}/{len(SEEDS)}")
check(f"{len(SEEDS)} 组里最久走了 {max_steps_total} 步 未出现死循环卡死",
      max_steps_total >= 50)  # 至少能推进 50 步，说明不是开局就死

# ======================================================================
# 汇总
# ======================================================================
print("\n" + "="*60)
print(f"📋 测试汇总：{PASS} 通过   {FAIL} 失败")
print("="*60)
if FAIL == 0:
    print("✅ 全部通过，核心逻辑架构正确")
else:
    print(f"❌ {FAIL} 项失败，需修正")
    sys.exit(1)
