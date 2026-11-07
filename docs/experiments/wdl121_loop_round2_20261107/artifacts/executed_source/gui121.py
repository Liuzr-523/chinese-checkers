"""121-cell GUI used by the standard gui.py entry point."""
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
import pygame
import rules121 as rules
from engine121 import Search121
import gui_style as skin
from opening121 import BookEngine
from wdl121_display import WDLAnalysis

W, H = 1240, 960

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

def candidate_badges(candidates,selected=None):
    """以连线对应末点，贪心避让百分比标牌，保留每条候选身份。"""
    badges=[]
    offsets=[(18,-29),(18,3),(-102,-29),(-102,3),(18,35),(-102,35),(18,-61),(-102,-61),(18,67),(-102,67)]
    for c in candidates:
        if selected is not None and c['path'][0]!=selected:continue
        point=pos_to_px(c['path'][-1]);rect=None
        for dx,dy in offsets:
            proposed=pygame.Rect(point[0]+dx,point[1]+dy,84,25)
            proposed.clamp_ip(pygame.Rect(8,112,854,718))
            if not any(proposed.inflate(6,6).colliderect(other['rect']) for other in badges):
                rect=proposed;break
        if rect is None:
            rect=pygame.Rect(18,116+len(badges)*32,84,25)
        value=c['wdl'];label=f"{c['rank']} · "+('未估计' if value is None else f'{value[0]:.1%}')
        badges.append(dict(candidate=c,rect=rect,point=point,label=label))
    return badges

def draw_wdl_candidates(screen,candidates,font,player,selected=None,mouse=None):
    """候选编号与同一条完整路径对应；同末点用偏移标记避免覆盖。"""
    mouse=pygame.mouse.get_pos() if mouse is None else mouse
    panel=pygame.Rect(880,16,344,918)
    pygame.draw.rect(screen,(229,237,242),panel,border_radius=12)
    def text(value,x,y,color=skin.TEXT):screen.blit(font.render(value,True,color),(x,y))
    text('实验候选 · 未校准',896,32)
    text(f'P{player} 当前方胜率 · 128次搜索',896,59)
    text('编号 / 路径 / 胜·和·负 / 访问次数',896,86)
    hovered=None
    for i,c in enumerate(candidates):
        rect=pygame.Rect(890,120+i*119,324,108)
        active=rect.collidepoint(mouse)
        pygame.draw.rect(screen,(248,250,251) if not active else (246,229,184),rect,border_radius=8)
        path=c['path'];text(f"{c['rank']}   {path[0]} → {path[-1]}   访问 {c['visits']}"+(' ·部分限步' if c.get('partial') else ''),902,rect.y+10)
        value=c['wdl']
        text('胜率尚无有效估值' if value is None else f'胜 {value[0]:.1%}  和 {value[1]:.1%}  负 {value[2]:.1%}',902,rect.y+38)
        route='→'.join(map(str,path))
        if len(route)>60:route=route[:59]+'…'
        for line in range(2):text(route[line*30:(line+1)*30],902,rect.y+64+line*18)
        if active:hovered=c
    if not candidates:text('候选分析更新中／暂无模型',896,136)
    text('整体：网络预测；候选：搜索估计',896,731)
    text('搜索量小，仅作模型估计',896,760)
    text('棋盘标牌：编号 · 当前方胜率',896,789)
    text('选子筛选；悬停候选查看完整路径',896,818)
    if hovered is not None:
        points=[pos_to_px(i) for i in hovered['path']]
        pygame.draw.lines(screen,skin.GREEN,False,points,4)
        pygame.draw.circle(screen,skin.GREEN,points[0],24,3)
    counts={};offsets=[(0,-18),(18,0),(0,18),(-18,0),(14,14)]
    for c in candidates:
        path=c['path']
        if selected is not None and path[0]!=selected:continue
        x,y=pos_to_px(path[-1]);j=counts.get(path[-1],0);counts[path[-1]]=j+1
        dx,dy=offsets[j%len(offsets)];center=(x+dx,y+dy)
        skin.smooth_circle(screen,(239,183,63),center,12)
        label=font.render(str(c['rank']),True,skin.TEXT);screen.blit(label,label.get_rect(center=center))
    badges=candidate_badges(candidates,selected)
    for badge in badges:pygame.draw.line(screen,(171,130,44),badge['point'],badge['rect'].center,1)
    for badge in badges:
        rect=badge['rect']
        pygame.draw.rect(screen,(255,240,190),rect,border_radius=6)
        pygame.draw.rect(screen,(185,139,49),rect,1,border_radius=6)
        label=font.render(badge['label'],True,skin.TEXT);screen.blit(label,label.get_rect(center=rect.center))
    return hovered

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
    use_book = os.environ.get('CC121_OPENING_BOOK','1') != '0'; last_source = ''
    book_button = pygame.Rect(300, 570, 340, 34)
    buttons = {'back': pygame.Rect(20, 885, 150, 42), 'undo': pygame.Rect(190, 885, 140, 42)}
    pool = ThreadPoolExecutor(max_workers=1)
    analysis = WDLAnalysis()
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
                        if book_button.collidepoint(event.pos): use_book = not use_book
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
                                if use_book: chosen_engines = {p: BookEngine(e) for p,e in chosen_engines.items()}
                                engines = chosen_engines
                            except Exception as exc:
                                error = str(exc)
                                continue
                            game = rules.Game(); menu = False; generation += 1
                            animation = None; selected = None; paths = []; error = ''; last_source = ''
                        if buttons['back'].collidepoint(event.pos): running = False
                    elif buttons['back'].collidepoint(event.pos):
                        menu = True; generation += 1; animation = None
                    elif buttons['undo'].collidepoint(event.pos):
                        generation += 1; animation = None; game.undo(); last_source = ''
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
                    path, info = future.result()
                    if submitted == generation and not menu and path:
                        if not rules.valid_path(game.board, game.player, path): raise ValueError('搜索返回非法走法')
                        animation = Animation(game.board, path)
                        last_source = '上一AI步：开局库' if info.get('book_hit') else '上一AI步：引擎搜索'
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
                pygame.draw.rect(screen, skin.BTN_BG, book_button, border_radius=8)
                screen.blit(small.render('Clysto开局库：'+('开' if use_book else '关')+'（点击切换）', True, skin.TEXT), (310,578))
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
                estimate = analysis.update(game, generation, animation is not None)
                screen.blit(small.render(estimate, True, skin.TEXT), (25, 82))
                draw_wdl_candidates(screen, analysis.candidates, small, game.player, selected)
                screen.blit(small.render(error or reason or ('AI 思考中…' if future else rules.PROFILE+'  '+last_source), True, skin.TEXT), (25, 845))
            for key, rect in buttons.items():
                if menu and key == 'undo': continue
                pygame.draw.rect(screen, skin.BTN_BG, rect, border_radius=8)
                label = '退出游戏' if menu else '返回设置'
                screen.blit(small.render(label if key == 'back' else '悔棋', True, skin.TEXT), (rect.x+12, rect.y+10))
            pygame.display.flip(); clock.tick(60)
    finally:
        analysis.close()
        pool.shutdown(wait=True, cancel_futures=True)
        pygame.quit()

if __name__ == '__main__': main()
