"""Shared pygame styling for the 121-cell GUI."""
import os
from functools import lru_cache
import pygame

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

R = 20

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

BTN_BG   = (255, 255, 255)

BTN_SEL  = (217, 231, 248)

def piece_colors(player):
    """P1=红, P2=青，固定不变"""
    if player == 1:
        return RED, RED_DK, (255, 180, 180)
    else:
        return CYAN, CYAN_DK, (180, 230, 240)

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
