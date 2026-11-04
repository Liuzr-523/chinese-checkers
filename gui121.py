"""121-cell GUI used by the standard gui.py entry point."""
import math
import time
from concurrent.futures import ThreadPoolExecutor
import pygame
import rules121 as rules
from engine121 import Search121
import gui_style as skin

W, H = 880, 960

def pos_to_px(pos):
    q, r = rules.COORDS[pos]
    return round(440+(q+r/2)*52), round(465+r*43)

def px_to_pos(point):
    return next((i for i in range(121) if math.dist(point, pos_to_px(i)) <= 21), None)

def draw_board(screen, board, selected=None, paths=()):
    screen.fill(skin.BG)
    for s, neighbors in rules.NEIGHBORS.items():
        for dst in neighbors:
            if dst > s:
                pygame.draw.aaline(screen, (211, 217, 219), pos_to_px(s), pos_to_px(dst))
    for s, piece in enumerate(board):
        q, r = rules.COORDS[s]
        color = (248, 211, 213) if s in rules.HOME[1] else (201, 231, 237) if s in rules.HOME[2] else (229, 222, 242) if max(abs(q), abs(r), abs(q+r)) > 4 else skin.HOLE
        skin.smooth_circle(screen, skin.HOLE_RIM, pos_to_px(s), 20)
        skin.smooth_circle(screen, color, pos_to_px(s), 18)
        if piece:
            skin.draw_piece(screen, piece, pos_to_px(s))
    if selected is not None:
        pygame.draw.circle(screen, skin.GOLD, pos_to_px(selected), 24, 3)
    for path in paths:
        pygame.draw.circle(screen, skin.GREEN, pos_to_px(path[-1]), 22, 3)

class Animation:
    def __init__(self, board, path):
        self.board = board.copy(); self.path = path
        self.player = board[path[0]]; self.board[path[0]] = 0
        self.start = time.monotonic()
        self.duration = .55*(len(path)-1)+.5
    def draw(self, screen, elapsed):
        alpha = int(210*min(1, elapsed/.25, max(0, (self.duration-elapsed)/.5)))
        layer = pygame.Surface((W, H), pygame.SRCALPHA)
        points = [pos_to_px(i) for i in self.path]
        pygame.draw.lines(layer, (*skin.GOLD, alpha), False, points, 6)
        screen.blit(layer, (0, 0))
        progress = min(len(points)-1, elapsed/.55)
        i = min(len(points)-2, int(progress)); fraction = progress-i
        fraction = fraction*fraction*(3-2*fraction)
        a, b = points[i], points[i+1]
        skin.draw_piece(screen, self.player, (a[0]+(b[0]-a[0])*fraction, a[1]+(b[1]-a[1])*fraction))

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H)); pygame.display.set_caption('中国跳棋 · 121格')
    font, small = skin._load_font(22), skin._load_font(16)
    clock = pygame.time.Clock(); game = rules.Game()
    ai = {1: 0, 2: 0}; engines = {}
    labels = ('人类', 'Python参考搜索', 'C-PVS 121格', '神经网络（已验收）')
    menu = True; running = True; future = None; animation = None
    selected = None; paths = []; error = ''; generation = 0; submitted = 0
    buttons = {'back': pygame.Rect(20, 885, 150, 42), 'undo': pygame.Rect(190, 885, 140, 42)}
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    if menu: running = False
                    else:
                        menu = True; generation += 1; animation = None
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if menu:
                        for p, y in ((1, 260), (2, 360)):
                            if pygame.Rect(320, y, 300, 50).collidepoint(event.pos): ai[p] = (ai[p]+1)%len(labels)
                        if pygame.Rect(320, 500, 240, 55).collidepoint(event.pos):
                            try:
                                chosen_engines = {}
                                for p in (1, 2):
                                    if ai[p] == 1: chosen_engines[p] = Search121()
                                    elif ai[p] == 2:
                                        from engine_c121 import CSearch121
                                        chosen_engines[p] = CSearch121()
                                    elif ai[p] == 3:
                                        from neural121_registry import load_registered
                                        chosen_engines[p] = load_registered()
                                engines = chosen_engines
                            except Exception as exc:
                                error = str(exc)
                                continue
                            game = rules.Game(); menu = False; generation += 1
                            animation = None; selected = None; paths = []; error = ''
                        if buttons['back'].collidepoint(event.pos): running = False
                    elif buttons['back'].collidepoint(event.pos):
                        menu = True; generation += 1; animation = None
                    elif buttons['undo'].collidepoint(event.pos):
                        generation += 1; animation = None; game.undo()
                        if any(ai.values()) and not all(ai.values()):
                            while game.history and ai[game.player]: game.undo()
                        selected = None; paths = []; error = ''
                    elif future is None and animation is None and not ai[game.player] and game.result()[0] is None:
                        pos = px_to_pos(event.pos)
                        chosen = next((p for p in paths if p[-1] == pos), None)
                        if chosen:
                            animation = Animation(game.board, chosen); selected = None; paths = []
                        elif pos is not None and game.board[pos] == game.player:
                            selected = pos; paths = rules.legal_paths(game.board, pos)
                        else: selected = None; paths = []
            if future is not None and future.done():
                try:
                    path, _ = future.result()
                    if submitted == generation and not menu and path:
                        if not rules.valid_path(game.board, game.player, path): raise ValueError('搜索返回非法走法')
                        animation = Animation(game.board, path)
                except Exception as exc:
                    if submitted == generation: error = str(exc)
                future = None
            if menu:
                screen.fill(skin.BG)
                screen.blit(font.render('121格 · 标准双人规则', True, skin.TEXT), (230, 150))
                for p, y in ((1, 260), (2, 360)):
                    label = f'P{p}：' + labels[ai[p]]
                    pygame.draw.rect(screen, skin.BTN_SEL, (320, y, 300, 50), border_radius=8)
                    screen.blit(font.render(label, True, skin.TEXT), (335, y+10))
                screen.blit(small.render('红方先手、青方后手；点击上方分别切换身份', True, skin.TEXT), (230, 440))
                pygame.draw.rect(screen, skin.BTN_SEL, (320, 500, 240, 55), border_radius=8)
                screen.blit(font.render('开始对局', True, skin.TEXT), (390, 514))
                screen.blit(small.render('进营后不可退出 · 堵营判胜 · 同轮完成判和 · 三次重复 / 240手判和', True, skin.TEXT), (130, 610))
                hint = '神经网络已通过旧版对照，但目前仍弱于C-PVS' if 3 in ai.values() else '点击双方身份可选择：人类 / Python参考搜索 / C-PVS / 神经网络'
                screen.blit(small.render(hint, True, skin.TEXT), (135, 650))
                if error: screen.blit(small.render(error[:75], True, skin.RED_DK), (100, 700))
            else:
                if animation and time.monotonic()-animation.start >= animation.duration:
                    game.play(animation.path); animation = None
                result, reason = game.result()
                if not animation and future is None and result is None and ai[game.player] and not error:
                    submitted = generation
                    future = pool.submit(engines[game.player].think, game.board.copy(), game.player, game.ply, game.repetitions.copy())
                draw_board(screen, animation.board if animation else game.board, selected, paths)
                if animation: animation.draw(screen, time.monotonic()-animation.start)
                title = f'P{game.player} 行棋' if result is None else ('和棋' if result == 0 else f'P{result} 获胜')
                screen.blit(font.render(title, True, skin.TEXT), (25, 25))
                screen.blit(small.render(f'P1 {(game.ply+1)//2} 步 · P2 {game.ply//2} 步 · 共 {game.ply} 手', True, skin.TEXT), (25, 60))
                screen.blit(small.render(error or reason or ('AI 思考中…' if future else rules.PROFILE), True, skin.TEXT), (25, 845))
            for key, rect in buttons.items():
                if menu and key == 'undo': continue
                pygame.draw.rect(screen, skin.BTN_BG, rect, border_radius=8)
                label = '退出游戏' if menu else '返回设置'
                screen.blit(small.render(label if key == 'back' else '悔棋', True, skin.TEXT), (rect.x+12, rect.y+10))
            pygame.display.flip(); clock.tick(60)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        pygame.quit()

if __name__ == '__main__': main()
