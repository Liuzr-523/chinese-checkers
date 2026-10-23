"""
中国跳棋 GUI（pygame）
支持四种身份模式：人 vs 人、人 vs AI、AI vs 人、AI vs AI
双方可独立选择：人 / C-PVS / PVS / 神经网络 / Clysto，并指定先手。
走法沿真实跳链播放动画，路径连线随动画淡入淡出。
点击棋子选中 → 绿色圆圈高亮合法目标 → 点击目标落子
"""
from checkpoint_paths import artifact_path, artifact_glob
import atexit
import os
import sys
import pygame
from functools import lru_cache
import math
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
from engine import AlphaBetaIDSEngine
from engine_pvs import PVSIDSEngine
import alphazero_nn as nn_module
from alphazero_nn import NetParams, NNEasyEngine, train_iteration

cc = engine.cc

# 默认引擎 = PVS（主变窗 + 空着裁剪）：实测 PVS@4s vs clysto@1s = 1:9，
# 显著强于原版 IDS@4s 的 0:10。评估函数仍用 v2（竞速权重版 v8 尚未验证胜率）。
AI_AB = PVSIDSEngine(time_limit_sec=4.0, max_depth=12, eval_version="v2")
AI_AB.name = "PVS(AlphaBeta)"

# ===== 引擎工厂：根据选择返回可 think() 的对象 =====
def _quit_engine(eng):
    try:
        if hasattr(eng, "quit"):
            eng.quit()
    except Exception:
        pass

def make_engine(kind):
    """为每位玩家创建独立引擎。"""
    if kind == "c":
        from engine_c import CESEngine
        return CESEngine(time_limit_sec=4.0, max_depth=20, eval_version="v2")
    if kind == "v2":
        eng = PVSIDSEngine(time_limit_sec=4.0, max_depth=12, eval_version="v2")
        eng.name = "PVS(AlphaBeta)"
        return eng
    if kind == "cly":
        # clysto C++ 引擎 @1s ≈ 业余高手（妈妈）水平，作为陪练/强敌
        try:
            from arena_vs_clysto import ClystoEngine
            eng = ClystoEngine(timelimit=1)
        except Exception as e:
            print(f"[gui] clysto 引擎启动失败({e})，回退 v2")
            return make_engine("v2")
        eng.name = "Clysto(1s)"
        atexit.register(_quit_engine, eng)
        return eng
    # 神经网络：ResNet 冠军（torch 直推，无 MCTS）
    from nn_resnet import NetResNet
    from nn_torch import NNEasyEngineTorch
    ckpt = artifact_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "nn_checkpoints", "params_resnet_champ.pt"))
    if os.path.exists(ckpt):
        model = NetResNet.load(ckpt)
        print(f"[gui] 载入 ResNet 冠军 {ckpt}")
    else:
        from nn_torch import NetTorch
        pkl = artifact_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "nn_checkpoints", "params_az_champ.pkl"))
        model = NetTorch.from_numpy_params(NetParams.load(pkl))
        print(f"[gui] ResNet 冠军缺失，回退旧模型 {pkl}")
    eng = NNEasyEngineTorch(model, temperature=0.0, seed=42)
    eng.name = "神经网络(ResNet)"
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
BG       = (238, 243, 248)
HOLE     = (231, 229, 220)
HOLE_RIM = (184, 190, 191)
RED      = (229, 83, 92)
RED_DK   = (158, 47, 65)
CYAN     = (43, 168, 190)
CYAN_DK  = (24, 106, 137)
GREEN    = (44, 170, 113)
GOLD     = (236, 170, 48)
TEXT     = (47, 59, 75)
PANEL    = (255, 255, 255)
PANEL_BD = (211, 220, 228)
BTN_BG   = (255, 255, 255)
BTN_HL   = (229, 237, 246)
BTN_SEL  = (217, 231, 248)


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
@lru_cache(maxsize=128)
def _circle_sprite(color, radius):
    # 先放大绘制再缩小，避开 SDL_gfx 的圆周混色黑点。
    scale = 4
    size = 2 * (radius + 2)
    sprite = pygame.Surface((size * scale, size * scale), pygame.SRCALPHA)
    # 透明区保留同色 RGB，缩放边缘时也不会混入黑色。
    sprite.fill((*color[:3], 0))
    pygame.draw.circle(sprite, color, ((radius+2)*scale, (radius+2)*scale), radius*scale)
    return pygame.transform.smoothscale(sprite, (size, size))


def smooth_circle(screen, color, center, radius):
    x, y = map(int, center)
    screen.blit(_circle_sprite(tuple(color), radius), (x-radius-2, y-radius-2))


def draw_piece(screen, player, center):
    x, y = map(int, center)
    color, dark, light = piece_colors(player)
    smooth_circle(screen, (184, 190, 193), (x+1, y+3), R)
    smooth_circle(screen, dark, (x,y), R)
    smooth_circle(screen, color, (x,y-1), R-2)
    glow = tuple(min(255, int(c*.75+255*.25)) for c in color)
    smooth_circle(screen, glow, (x-3,y-5), R-8)
    smooth_circle(screen, light, (x-6,y-8), 4)
    smooth_circle(screen, (255,255,255), (x-7,y-9), 2)


def draw_board(screen, board, selected, legal_dests):
    screen.fill(BG)
    # 棋盘底板与双方营地，中心区域保持中性色。
    outline = [(440,28),(684,444),(440,860),(196,444)]
    shadow = [(x,y+5) for x,y in outline]
    pygame.draw.polygon(screen, (215,223,231), shadow)
    pygame.draw.polygon(screen, (250,249,245), outline)
    pygame.draw.aalines(screen, (206,215,222), True, outline)
    pygame.draw.polygon(screen, (253,224,225), [(440,30),(326,225),(554,225)])
    pygame.draw.polygon(screen, (215,241,246), [(326,663),(554,663),(440,858)])
    # 邻接网格让可走方向更容易辨认。
    for pos, neighbors in cc.NEIGHBORS_81.items():
        for other in neighbors:
            if other > pos:
                pygame.draw.aaline(screen, (216,218,211), pos_to_px(pos), pos_to_px(other))
    for pos in range(81):
        center = pos_to_px(pos)
        rim = (217,162,166) if pos < 10 else (139,194,204) if pos >= 71 else HOLE_RIM
        fill = (248,211,213) if pos < 10 else (201,231,237) if pos >= 71 else HOLE
        smooth_circle(screen, rim, center, R)
        smooth_circle(screen, fill, center, R-2)
        if board[pos]:
            draw_piece(screen, board[pos], center)
    if selected is not None:
        pygame.draw.circle(screen, GOLD, pos_to_px(selected), R+5, 3)
    for dst in legal_dests:
        center = pos_to_px(dst)
        pygame.draw.circle(screen, GREEN, center, R+2, 3)
        smooth_circle(screen, (104,203,151), center, 6)


def draw_game_status(screen, font, font_s, counts, current_player, first_player,
                     is_ai, engines, winner):
    """步数按整手统计，连跳无论几段都只算一步。"""
    for player, x in ((1,24),(2,670)):
        rect = pygame.Rect(x,88,186,168)
        color, dark, _ = piece_colors(player)
        active = current_player == player and not winner
        pygame.draw.rect(screen, (219,227,235), rect.move(0,3), border_radius=14)
        pygame.draw.rect(screen, PANEL, rect, border_radius=14)
        pygame.draw.rect(screen, color if active else PANEL_BD, rect, 2, border_radius=14)
        pygame.draw.rect(screen, color, (x+14,104,5,24), border_radius=2)
        label = f"P{player} {'红方' if player == 1 else '青方'}"
        screen.blit(font.render(label, True, dark), (x+28,103))
        role = "先手" if player == first_player else "后手"
        kind = getattr(engines.get(player), 'name', '人') if is_ai[player] else '人'
        # 使用短名，避免引擎参数挤出侧栏。
        if is_ai[player]:
            kind = 'C-PVS' if kind.startswith('C-PVS') else 'PVS' if kind.startswith('PVS') else 'Clysto' if kind.startswith('Clysto') else 'NN'
        screen.blit(font_s.render(f"{role} · {kind}", True, (112,125,141)), (x+16,138))
        screen.blit(font.render(f"{counts[player]} 步", True, dark), (x+16,169))
        target = "目标：下方青营 ↓" if player == 1 else "目标：上方红营 ↑"
        screen.blit(font_s.render(target, True, TEXT), (x+16,207))
        if active:
            smooth_circle(screen, color, (x+161,115), 5)
    total = sum(counts.values())
    round_number = max(1, (total+1)//2) if winner else total//2+1
    for text, y in ((f"第 {round_number} 回合",302),(f"共走 {total} 手",332)):
        screen.blit(font_s.render(text, True, (103,119,137)), (40,y))
    screen.blit(font_s.render("一手连跳计一步", True, (125,139,154)), (40,365))


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
        kind = step_kind(last_move)
        mtext = f"上一手: [{kind}] {path_str}"
        msurf = font_s.render(mtext, True, TEXT)
        screen.blit(msurf, (20, y + 42))

    # 操作提示
    hint = "点击棋子选中 → 点击绿圈目标落子 | 右键取消 | U 悔棋 | R 返回设置 | ESC 返回/退出"
    hs = font_s.render(hint, True, TEXT)
    screen.blit(hs, (20, y + 68))


_ENGINE_LABELS = {"c": "C-PVS 加速", "v2": "PVS 竞速(AlphaBeta)", "nn": "神经网络(ResNet)", "cly": "Clysto(强敌)"}
_ENGINE_SHORT = {"c": "C-PVS", "v2": "PVS", "nn": "NN", "cly": "Clysto"}

def draw_start_screen(screen, font, font_s, is_ai, ai_kinds, first_player, error=""):
    screen.fill(BG)
    screen.blit(font.render("中国跳棋 · 对局设置", True, TEXT), (80, 110))
    screen.blit(font_s.render("双方独立选择身份和引擎，颜色与先后手互不绑定", True, TEXT), (80, 160))
    btns = []
    for player, y in ((1, 235), (2, 400)):
        label = "P1 红方" if player == 1 else "P2 青方"
        screen.blit(font.render(label, True, piece_colors(player)[1]), (80, y))
        for j, value in enumerate((False, True)):
            btns.append(dict(key="identity", player=player, val=value,
                             label="AI" if value else "人", selected=is_ai[player] == value,
                             rect=pygame.Rect(300+j*130, y-8, 115, 44)))
        for j, kind in enumerate(("c", "v2", "nn", "cly")):
            btns.append(dict(key="engine", player=player, val=kind,
                             label=_ENGINE_SHORT[kind], selected=ai_kinds[player] == kind,
                             disabled=not is_ai[player],
                             rect=pygame.Rect(300+j*125, y+50, 115, 44)))
    screen.blit(font.render("谁先走", True, TEXT), (80, 570))
    for j, player in enumerate((1, 2)):
        btns.append(dict(key="first", val=player, label=f"P{player} 先手",
                         selected=first_player == player,
                         rect=pygame.Rect(300+j*150, 560, 140, 44)))
    btns.append(dict(key="start", label="开始对局", selected=True,
                     rect=pygame.Rect(330, 675, 220, 54)))
    for btn in btns:
        disabled = btn.get("disabled", False)
        color = (225, 220, 210) if disabled else BTN_SEL if btn.get("selected") else BTN_BG
        pygame.draw.rect(screen, color, btn["rect"], border_radius=8)
        pygame.draw.rect(screen, PANEL_BD, btn["rect"], 1, border_radius=8)
        text = font_s.render(btn["label"], True, (145, 140, 130) if disabled else TEXT)
        screen.blit(text, text.get_rect(center=btn["rect"].center))
    if error:
        screen.blit(font_s.render(error[:75], True, RED_DK), (70, 770))
    pygame.display.flip()
    return btns


def step_kind(path):
    if len(path) == 2 and path[-1] in cc.NEIGHBORS_81[path[0]]:
        return "单步"
    return "单跳" if len(path) == 2 else f"连跳{len(path)-1}次"


def resolve_move_path(board, player, path):
    """C 引擎只返回端点；从规则层补全用于演示的合法跳链。"""
    if not path or len(path) < 2:
        raise ValueError("引擎未返回有效走法")
    paths = cc.get_all_legal_moves(path[0], board) if board[path[0]] == player else []
    for candidate in paths:
        if candidate == list(path):
            return candidate
    for candidate in paths:
        if candidate[-1] == path[-1]:
            return candidate
    raise ValueError(f"非法走法: {path[0]} → {path[-1]}")


class MoveAnimation:
    """按真实路径逐段跳跃，停留展示落点后再交给下一方。"""
    segment_seconds = 0.48
    landing_seconds = 0.75
    fade_in_seconds = 0.25
    fade_out_seconds = 0.55

    def __init__(self, board, player, path, now):
        self.path = resolve_move_path(board, player, path)
        self.player = player
        self.started = now
        self.board = board.copy()
        self.board[self.path[0]] = 0
        self.duration = (len(self.path)-1) * self.segment_seconds + self.landing_seconds

    def done(self, now):
        return now - self.started >= self.duration

    def draw(self, screen, now):
        points = [pos_to_px(pos) for pos in self.path]
        layer = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        pygame.draw.lines(layer, (255, 180, 35, 150), False, points, 6)
        for i, point in enumerate(points):
            pygame.draw.circle(layer, (255, 210, 60, 200), point, R+5, 2)
            label = _load_font(14).render(str(i+1), True, TEXT)
            layer.blit(label, (point[0]+18, point[1]-28))
        elapsed = max(0.0, now - self.started)
        opacity = min(1.0, elapsed / self.fade_in_seconds,
                      max(0.0, (self.duration - elapsed) / self.fade_out_seconds))
        # smoothstep：连线、落点圈及编号一起柔和淡入淡出。
        opacity = opacity * opacity * (3 - 2 * opacity)
        layer.set_alpha(round(255 * opacity))
        screen.blit(layer, (0, 0))
        progress = max(0, (now-self.started) / self.segment_seconds)
        segment = min(int(progress), len(points)-2)
        t = min(1.0, progress-segment)
        t = t*t*(3-2*t)
        a, b = points[segment], points[segment+1]
        lift = 16 * math.sin(math.pi*t)
        x = round(a[0]+(b[0]-a[0])*t)
        ground_y = round(a[1]+(b[1]-a[1])*t)
        y = round(ground_y-lift)
        pygame.draw.ellipse(screen, (165, 147, 116), (x-17, ground_y+12, 34, 10))
        pygame.draw.circle(screen, GOLD, (x, y), R+5, 3)
        draw_piece(screen, self.player, (x,y))


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
# Side-panel controls stay clear of the board and bottom status text.
GAME_BUTTONS = {
    "undo": pygame.Rect(24, 410, 186, 40),
    "back": pygame.Rect(24, 462, 186, 40),
    "resume": pygame.Rect(24, 514, 186, 40),
}


def draw_game_controls(screen, font, can_undo, paused):
    for key, label in (("undo", "悔棋  U"), ("back", "返回对战设置  R"),
                       ("resume", "继续 AI  空格")):
        if key == "resume" and not paused:
            continue
        rect = GAME_BUTTONS[key]
        enabled = key != "undo" or can_undo
        pygame.draw.rect(screen, PANEL if enabled else (226,231,237), rect, border_radius=9)
        pygame.draw.rect(screen, PANEL_BD, rect, 1, border_radius=9)
        text = font.render(label, True, TEXT if enabled else (150,159,168))
        screen.blit(text, text.get_rect(center=rect.center))


def undo_index(history, is_ai, pending_player=None):
    """Human vs AI rewinds to the human's last turn; other modes undo one move.
    An uncommitted human/AI-vs-AI animation is simply cancelled first.
    """
    human_vs_ai = sum(is_ai.values()) == 1
    if pending_player is not None and (not human_vs_ai or not is_ai[pending_player]):
        return len(history)
    if human_vs_ai:
        return next((i for i in range(len(history)-1, -1, -1)
                     if not is_ai[history[i]['player']]), None)
    return len(history)-1 if history else None


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("中国跳棋")
    clock = pygame.time.Clock()
    font, font_s = _load_font(22), _load_font(16)
    mode = "menu"
    is_ai = {1: False, 2: True}
    ai_kinds = {1: "c", 2: "c"}
    first_player = 1
    engines = {}
    executor = ThreadPoolExecutor(max_workers=1)
    future = None
    history = []
    ai_paused = False
    undo_at = None
    board = cc.init_board()
    current_player = 1
    move_counts = {1: 0, 2: 0}
    selected, legal_dest_map = None, {}
    last_move, winner, animation = None, None, None
    error = ""
    running = True

    def close_engines():
        for eng in engines.values():
            _quit_engine(eng)
        engines.clear()

    def start_game():
        nonlocal board, current_player, selected, legal_dest_map, last_move, winner, animation, error, ai_paused
        close_engines()
        try:
            for player in (1, 2):
                if is_ai[player]:
                    engines[player] = make_engine(ai_kinds[player])
        except Exception as exc:
            close_engines()
            error = f"引擎启动失败：{exc}"
            return False
        history.clear()
        ai_paused = False
        board = cc.init_board()
        current_player = first_player
        move_counts.update({1: 0, 2: 0})
        selected, legal_dest_map = None, {}
        last_move, winner, animation = None, None, None
        error = ""
        return True

    def restore_undo():
        nonlocal board, current_player, last_move, winner, selected, legal_dest_map, error, ai_paused
        if undo_at is not None and undo_at < len(history):
            snapshot = history[undo_at]
            board = snapshot['board'].copy()
            current_player = snapshot['player']
            move_counts.update(snapshot['counts'])
            last_move = snapshot['last_move']
            winner = snapshot['winner']
            del history[undo_at:]
        selected, legal_dest_map = None, {}
        error = ""
        ai_paused = all(is_ai.values())
        # Rebuild history-sensitive NN state from the retained game, after search has stopped.
        for eng in engines.values():
            if hasattr(eng, 'reset_history'):
                eng.reset_history()
                if hasattr(eng, 'observe_position'):
                    for snapshot in history:
                        eng.observe_position(snapshot['board'], snapshot['player'])
                    eng.observe_position(board, current_player)

    try:
        while running:
            now = time.monotonic()
            if mode == "menu":
                btns = draw_start_screen(screen, font, font_s, is_ai, ai_kinds, first_player, error)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (mode == "menu" and ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    running = False
                    break
                if mode == "menu":
                    if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                        for btn in btns:
                            if not btn["rect"].collidepoint(ev.pos) or btn.get("disabled"):
                                continue
                            if btn["key"] == "identity":
                                is_ai[btn["player"]] = btn["val"]
                            elif btn["key"] == "engine":
                                ai_kinds[btn["player"]] = btn["val"]
                            elif btn["key"] == "first":
                                first_player = btn["val"]
                            elif btn["key"] == "start" and start_game():
                                mode = "playing"
                            break
                    continue
                back = (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_r, pygame.K_ESCAPE)) or (
                    ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and GAME_BUTTONS['back'].collidepoint(ev.pos))
                undo = (ev.type == pygame.KEYDOWN and ev.key == pygame.K_u) or (
                    ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and GAME_BUTTONS['undo'].collidepoint(ev.pos))
                resume = (ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE) or (
                    ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and GAME_BUTTONS['resume'].collidepoint(ev.pos))
                if back:
                    mode = "waiting_menu" if future is not None else "menu"
                    animation = None
                    error = ""
                    if future is None:
                        close_engines()
                    else:
                        future.cancel()
                    break
                if mode.startswith("waiting_"):
                    continue
                if undo:
                    undo_at = undo_index(history, is_ai, current_player if animation or future is not None else None)
                    if undo_at is not None:
                        animation = None
                        if future is not None:
                            future.cancel()
                            mode = "waiting_undo"
                        else:
                            restore_undo()
                    break
                if resume and ai_paused:
                    ai_paused = False
                    continue
                if animation or future is not None or winner or is_ai[current_player]:
                    continue
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                    selected, legal_dest_map = None, {}
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    pos = px_to_pos(*ev.pos)
                    if pos in legal_dest_map:
                        animation = MoveAnimation(board, current_player, legal_dest_map[pos], now)
                        selected, legal_dest_map = None, {}
                    elif pos is not None and board[pos] == current_player:
                        selected = pos
                        legal_dest_map = get_legal_dests(board, current_player, pos)
                    else:
                        selected, legal_dest_map = None, {}
            if not running:
                break
            if mode in ("waiting_menu", "waiting_undo"):
                if future.done():
                    future = None  # Discard stale search result, including any exception.
                    if mode == "waiting_menu":
                        close_engines()
                        mode = "menu"
                    else:
                        restore_undo()
                        mode = "playing"
                else:
                    draw_board(screen, board, None, [])
                    text = "正在结束思考，随后返回设置…" if mode == "waiting_menu" else "正在结束思考，随后悔棋…"
                    screen.blit(font_s.render(text, True, TEXT), (240, 30))
                    draw_game_controls(screen, font_s, False, False)
                    pygame.display.flip()
                clock.tick(FPS)
                continue
            if mode == "menu":
                clock.tick(FPS)
                continue
            if animation and animation.done(now):
                history.append(dict(board=board.copy(), player=current_player,
                                    counts=move_counts.copy(), last_move=last_move, winner=winner))
                last_move = animation.path
                board = cc.apply_move(board, last_move)
                move_counts[current_player] += 1
                animation = None
                winner = cc.is_game_over(board)
                if not winner:
                    current_player = 3-current_player
            if future is not None and future.done():
                try:
                    path, info = future.result()
                    if path is None:
                        winner = 3-current_player
                    else:
                        animation = MoveAnimation(board, current_player, path, now)
                except Exception as exc:
                    error = f"AI 出错：{exc}（按 R 返回设置）"
                    print(f"[gui] {error}")
                future = None
            if not animation and future is None and not winner and not error and not ai_paused and is_ai[current_player]:
                future = executor.submit(engines[current_player].think, board.copy(), current_player)
            draw_board(screen, animation.board if animation else board,
                       selected, list(legal_dest_map))
            if animation:
                animation.draw(screen, now)
            draw_panel(screen, font, font_s, current_player, is_ai[1], is_ai[2],
                       future is not None, winner, animation.path if animation else last_move)
            draw_game_status(screen, font, font_s, move_counts, current_player,
                             first_player, is_ai, engines, winner)
            can_undo = undo_index(history, is_ai, current_player if animation or future is not None else None) is not None
            draw_game_controls(screen, font_s, can_undo, ai_paused)
            if ai_paused:
                screen.blit(font_s.render("已暂停 AI，可继续悔棋或恢复对弈", True, TEXT), (20, 35))
            if animation:
                screen.blit(font_s.render("走棋演示中…", True, TEXT), (20, 35))
            if error:
                screen.blit(font_s.render(error[:85], True, RED_DK), (20, 35))
            pygame.display.flip()
            clock.tick(FPS)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        close_engines()
        pygame.quit()


if __name__ == "__main__":
    main()
