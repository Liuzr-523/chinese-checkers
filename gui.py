"""
中国跳棋 GUI（pygame）
支持四种身份模式：人 vs 人、人 vs AI、AI vs 人、AI vs AI
AI 引擎可选：v2启发式(AlphaBeta) 或 神经网络(MCTS+PUCT)
点击棋子选中 → 绿色圆圈高亮合法目标 → 点击目标落子
"""
import os
import sys
import pygame
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
from engine import AlphaBetaIDSEngine
import alphazero_nn as nn_module
from alphazero_nn import NetParams, NNEasyEngine, train_iteration

cc = engine.cc

AI_AB = AlphaBetaIDSEngine(time_limit_sec=1.2, max_depth=8, eval_version="v2")
AI_AB.name = "v2(AlphaBeta)"

# ===== 引擎工厂：根据选择返回可 think() 的对象 =====
def make_engine(kind):
    """kind in {'v2', 'nn'}"""
    if kind == "v2":
        return AI_AB
    # 神经网络
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "nn_checkpoints", "params_latest.pkl")
    if os.path.exists(ckpt):
        params = NetParams.load(ckpt)
        print(f"[gui] 载入神经网络参数 {ckpt}")
    else:
        params = NetParams(seed=7)
        print(f"[gui] 神经网络参数未找到，使用随机初始化")
    eng = NNEasyEngine(params, use_mcts=True, n_simulations=24)
    eng.name = "NN(MCTS)"
    return eng

# ===== 窗口 =====
WIN_W, WIN_H = 880, 960
FPS = 60

# ===== 中文字体：用系统确定存在的字体文件，避免乱码 =====
_CANDIDATE_FONTS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]

def _load_font(size):
    for p in _CANDIDATE_FONTS:
        if os.path.exists(p):
            return pygame.font.Font(p, size)
    return pygame.font.SysFont(None, size)

# ===== 棋盘尺寸 =====
COL_SP = 52
ROW_SP = 48
R = 20
CENTER_X = WIN_W // 2
MARGIN_Y = 60

# ===== 颜色 =====
BG       = (245, 235, 210)
HOLE     = (205, 188, 158)
HOLE_RIM = (180, 160, 130)
RED      = (225, 70, 70)
RED_DK   = (150, 30, 30)
CYAN     = (70, 185, 205)
CYAN_DK  = (30, 120, 140)
GREEN    = (80, 200, 80)
GOLD     = (255, 200, 0)
TEXT     = (50, 40, 30)
PANEL    = (255, 250, 240)
PANEL_BD = (200, 180, 150)
BTN_BG   = (255, 250, 240)
BTN_HL   = (255, 230, 180)
BTN_SEL  = (255, 200, 100)


# ===== 坐标转换 =====
def pos_to_px(pos):
    x, y = cc.POS_TO_XY[pos]
    px = CENTER_X + (x - 4) * COL_SP + (y - 8) * COL_SP // 2
    py = MARGIN_Y + y * ROW_SP
    return (px, py)


def px_to_pos(mx, my):
    best = None
    best_d = R * R * 2
    for pos in range(81):
        px, py = pos_to_px(pos)
        d = (px - mx) ** 2 + (py - my) ** 2
        if d < best_d:
            best_d = d
            best = pos
    return best


def piece_colors(player):
    """P1=红, P2=青，固定不变"""
    if player == 1:
        return RED, RED_DK, (255, 180, 180)
    else:
        return CYAN, CYAN_DK, (180, 230, 240)


# ===== 绘制 =====
def draw_board(screen, board, selected, legal_dests):
    screen.fill(BG)

    # 棋盘孔
    for pos in range(81):
        px, py = pos_to_px(pos)
        pygame.draw.circle(screen, HOLE_RIM, (px, py), R)
        pygame.draw.circle(screen, HOLE, (px, py), R - 2)

    # 棋子
    for pos in range(81):
        px, py = pos_to_px(pos)
        if board[pos] != 0:
            c, dk, hl = piece_colors(board[pos])
            pygame.draw.circle(screen, dk, (px, py), R)
            pygame.draw.circle(screen, c, (px, py), R - 2)
            pygame.draw.circle(screen, hl, (px - 5, py - 5), 4)

    # 选中棋子高亮
    if selected is not None:
        px, py = pos_to_px(selected)
        pygame.draw.circle(screen, GOLD, (px, py), R + 4, 3)

    # 合法目标高亮
    for dst in legal_dests:
        px, py = pos_to_px(dst)
        pygame.draw.circle(screen, GREEN, (px, py), R, 3)
        pygame.draw.circle(screen, (200, 255, 200), (px, py), 7)


def draw_panel(screen, font, font_s, current_player, p1_is_ai, p2_is_ai,
               ai_thinking, winner, last_move):
    y = WIN_H - 95
    pygame.draw.rect(screen, PANEL, (0, y, WIN_W, 95))
    pygame.draw.line(screen, PANEL_BD, (0, y), (WIN_W, y), 2)

    # 当前回合
    is_ai = p1_is_ai if current_player == 1 else p2_is_ai
    pcolor_name = "红" if current_player == 1 else "青"
    c, dk, _ = piece_colors(current_player)

    if winner:
        wname = "红" if winner == 1 else "青"
        wtype = "AI" if (p1_is_ai if winner == 1 else p2_is_ai) else "人"
        text = f"{wname}方({wtype}) 获胜！"
        color = dk
    elif ai_thinking:
        text = f"{pcolor_name}方 AI 思考中..."
        color = dk
    elif not is_ai:
        text = f"{pcolor_name}方(人) 回合 — 点击棋子"
        color = dk
    else:
        text = f"{pcolor_name}方(AI) 回合"
        color = dk

    surf = font.render(text, True, color)
    screen.blit(surf, (20, y + 10))

    # 上一手
    if last_move:
        path_str = " → ".join(str(p) for p in last_move)
        n = len(last_move)
        kind = "单步" if n == 2 else ("单跳" if n == 3 else f"连跳{n - 2}")
        mtext = f"上一手: [{kind}] {path_str}"
        msurf = font_s.render(mtext, True, TEXT)
        screen.blit(msurf, (20, y + 42))

    # 操作提示
    hint = "点击棋子选中 → 点击绿圈目标落子 | 右键取消 | R 重新开始 | ESC 退出"
    hs = font_s.render(hint, True, TEXT)
    screen.blit(hs, (20, y + 68))


_ENGINE_LABELS = {"v2": "v2 启发式(AlphaBeta)", "nn": "NN(MCTS+PUCT)"}

def draw_start_screen(screen, font, font_s, p1_is_ai, p2_is_ai, ai_kind):
    screen.fill(BG)
    title = font.render("中国跳棋", True, TEXT)
    screen.blit(title, (CENTER_X - title.get_width() // 2, 120))

    sub = font_s.render("选择双方身份 + AI 引擎", True, TEXT)
    screen.blit(sub, (CENTER_X - sub.get_width() // 2, 170))

    # P1 行
    p1_label = font.render("红方 (P1, 0-9)", True, RED_DK)
    screen.blit(p1_label, (CENTER_X - 220, 240))

    # P2 行
    p2_label = font.render("青方 (P2, 71-80)", True, CYAN_DK)
    screen.blit(p2_label, (CENTER_X - 220, 310))

    # AI 引擎行
    ai_label = font.render("AI 引擎", True, TEXT)
    screen.blit(ai_label, (CENTER_X - 220, 380))

    # 按钮
    btn_w, btn_h = 110, 44
    bx = CENTER_X + 20

    btns = []
    # P1 人
    btns.append({"key": "p1", "val": False, "rect": pygame.Rect(bx, 234, btn_w, btn_h)})
    # P1 AI
    btns.append({"key": "p1", "val": True, "rect": pygame.Rect(bx + 130, 234, btn_w, btn_h)})
    # P2 人
    btns.append({"key": "p2", "val": False, "rect": pygame.Rect(bx, 304, btn_w, btn_h)})
    # P2 AI
    btns.append({"key": "p2", "val": True, "rect": pygame.Rect(bx + 130, 304, btn_w, btn_h)})
    # AI kind
    btns.append({"key": "kind", "val": "v2", "rect": pygame.Rect(bx, 374, btn_w, btn_h)})
    btns.append({"key": "kind", "val": "nn", "rect": pygame.Rect(bx + 130, 374, btn_w, btn_h)})
    # 开始按钮
    btns.append({"key": "start", "rect": pygame.Rect(CENTER_X - 90, 470, 180, 54)})

    for btn in btns:
        if btn["key"] == "start":
            pygame.draw.rect(screen, BTN_SEL, btn["rect"], border_radius=8)
            pygame.draw.rect(screen, PANEL_BD, btn["rect"], 2, border_radius=8)
            t = font.render("开始对局", True, TEXT)
            screen.blit(t, (btn["rect"].x + 38, btn["rect"].y + 14))
        elif btn["key"] == "kind":
            sel = (ai_kind == btn["val"])
            bg_c = BTN_SEL if sel else BTN_BG
            pygame.draw.rect(screen, bg_c, btn["rect"], border_radius=6)
            pygame.draw.rect(screen, PANEL_BD, btn["rect"], 2, border_radius=6)
            t = font_s.render(_ENGINE_LABELS[btn["val"]], True, TEXT)
            # 如果太长就缩小
            if t.get_width() > btn_w - 8:
                t = font_s.render("NN" if btn["val"]=="nn" else "v2", True, TEXT)
            screen.blit(t, (btn["rect"].x + 6, btn["rect"].y + 12))
        else:
            sel = (p1_is_ai if btn["key"] == "p1" else p2_is_ai) == btn["val"]
            bg_c = BTN_SEL if sel else BTN_BG
            pygame.draw.rect(screen, bg_c, btn["rect"], border_radius=6)
            pygame.draw.rect(screen, PANEL_BD, btn["rect"], 2, border_radius=6)
            label = "AI" if btn["val"] else "人"
            t = font_s.render(label, True, TEXT)
            screen.blit(t, (btn["rect"].x + 38, btn["rect"].y + 12))

    pygame.display.flip()
    return btns


def step_kind(path):
    n = len(path)
    if n == 2: return "单步"
    if n == 3: return "单跳"
    return f"连跳{n - 2}"


def get_legal_dests(board, player, piece_pos):
    """获取选中棋子的所有合法目标 → {dst_pos: best_path}"""
    moves = cc.get_all_moves(board, player)
    dest_map = {}
    for pos, path in moves:
        if pos == piece_pos:
            dst = path[-1]
            if dst not in dest_map or len(path) > len(dest_map[dst]):
                dest_map[dst] = path
    return dest_map


# ===== 主循环 =====
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("中国跳棋")
    clock = pygame.time.Clock()
    font = _load_font(22)
    font_s = _load_font(16)

    # 模式：'menu' | 'playing'
    mode = "menu"
    p1_is_ai = False
    p2_is_ai = True
    ai_kind = "v2"   # 'v2' | 'nn'
    ai_engine = None  # 运行时对弈的引擎对象，由 reset_game 时创建
    btns = draw_start_screen(screen, font, font_s, p1_is_ai, p2_is_ai, ai_kind)

    # 棋局运行时变量
    board = None
    is_ai = {1: False, 2: False}
    current_player = 1
    selected = None
    legal_dest_map = {}
    ai_pending = False
    last_move = None
    winner = None

    def reset_game():
        nonlocal board, current_player, selected, legal_dest_map, ai_pending, last_move, winner, ai_engine
        ai_engine = make_engine(ai_kind)
        print(f"[gui] 使用 AI 引擎: {getattr(ai_engine, 'name', ai_kind)}")
        board = cc.init_board()
        is_ai[1] = p1_is_ai
        is_ai[2] = p2_is_ai
        current_player = 1
        selected = None
        legal_dest_map = {}
        ai_pending = is_ai[1]
        last_move = None
        winner = None

    while True:
        # ===== 菜单界面 =====
        if mode == "menu":
            btns = draw_start_screen(screen, font, font_s, p1_is_ai, p2_is_ai, ai_kind)
            in_menu = True
            while in_menu:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        pygame.quit(); return
                    if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                        pygame.quit(); return
                    if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                        for btn in btns:
                            if btn["rect"].collidepoint(ev.pos):
                                if btn["key"] == "p1":
                                    p1_is_ai = btn["val"]
                                    btns = draw_start_screen(screen, font, font_s, p1_is_ai, p2_is_ai, ai_kind)
                                elif btn["key"] == "p2":
                                    p2_is_ai = btn["val"]
                                    btns = draw_start_screen(screen, font, font_s, p1_is_ai, p2_is_ai, ai_kind)
                                elif btn["key"] == "kind":
                                    ai_kind = btn["val"]
                                    btns = draw_start_screen(screen, font, font_s, p1_is_ai, p2_is_ai, ai_kind)
                                elif btn["key"] == "start":
                                    reset_game()
                                    mode = "playing"
                                    in_menu = False
                                break

        # ===== 对弈界面 =====
        if mode == "playing":
            # ---- 事件处理 ----
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); return
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        pygame.quit(); return
                    elif ev.key == pygame.K_r:
                        mode = "menu"
                        break
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                    selected = None
                    legal_dest_map = {}
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if winner is not None or ai_pending:
                        continue
                    if is_ai[current_player]:
                        continue
                    pos = px_to_pos(*ev.pos)
                    if pos is None:
                        continue

                    if selected is None:
                        if board[pos] == current_player:
                            dest_map = get_legal_dests(board, current_player, pos)
                            if dest_map:
                                selected = pos
                                legal_dest_map = dest_map
                            else:
                                selected = None
                                legal_dest_map = {}
                    else:
                        if pos in legal_dest_map:
                            path = legal_dest_map[pos]
                            board = cc.apply_move(board, path)
                            last_move = path
                            selected = None
                            legal_dest_map = {}
                            w = cc.is_game_over(board)
                            if w:
                                winner = w
                            else:
                                current_player = 3 - current_player
                                if is_ai[current_player]:
                                    ai_pending = True
                        elif board[pos] == current_player:
                            dest_map = get_legal_dests(board, current_player, pos)
                            if dest_map:
                                selected = pos
                                legal_dest_map = dest_map
                            else:
                                selected = None
                                legal_dest_map = {}
                        else:
                            selected = None
                            legal_dest_map = {}

            if mode != "playing":
                clock.tick(FPS)
                continue

            # ---- 绘制 ----
            draw_board(screen, board, selected, list(legal_dest_map.keys()))
            draw_panel(screen, font, font_s, current_player, p1_is_ai, p2_is_ai,
                       ai_pending, winner, last_move)
            # 右上角显示当前引擎
            eng_text = font_s.render(f"AI: {getattr(ai_engine,'name', ai_kind)}", True, TEXT)
            screen.blit(eng_text, (WIN_W - eng_text.get_width() - 16, 10))
            pygame.display.flip()

            # ---- AI 思考（由引擎对象 think 决定）----
            if ai_pending and winner is None:
                path, info = ai_engine.think(board, current_player)
                ai_pending = False
                if path is None:
                    winner = 3 - current_player
                else:
                    board = cc.apply_move(board, path)
                    last_move = path
                    w = cc.is_game_over(board)
                    if w:
                        winner = w
                    else:
                        current_player = 3 - current_player
                        if is_ai[current_player]:
                            ai_pending = True

        clock.tick(FPS)


if __name__ == "__main__":
    main()
