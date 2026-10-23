"""Run with the checkers Python; SDL dummy keeps tests off the desktop."""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import patch
import random
import pygame
import gui


class GuiTests(unittest.TestCase):
    def test_endpoint_reconstruction_and_animation(self):
        b = gui.cc.init_board()
        rng = random.Random(9)
        jumps = 0
        pygame.init()
        screen = pygame.display.set_mode((gui.WIN_W, gui.WIN_H))
        for turn in range(60):
            player = 1 + turn % 2
            moves = gui.cc.get_all_moves(b, player)
            _, path = rng.choice(moves)
            original = b.copy()
            a = gui.MoveAnimation(b, player, [path[0], path[-1]], 10)
            self.assertEqual(a.path[0], path[0])
            self.assertEqual(a.path[-1], path[-1])
            self.assertIn(a.path, [p for _, p in moves])
            self.assertEqual(b, original)
            for t in (10, 10.15, 10+a.duration-.01):
                gui.draw_board(screen, a.board, None, [])
                a.draw(screen, t)
                self.assertFalse(a.done(t))
            self.assertTrue(a.done(10+a.duration+.01))
            if len(a.path)>2:
                jumps += 1
                pygame.image.save(screen, '/tmp/cc-gui-animation.png')
            b = gui.cc.apply_move(b, path)
        self.assertGreater(jumps, 0)
        pygame.quit()

    def test_independent_engines_second_player_starts_and_reset(self):
        frames = 0
        calls, engines = [], []
        class FakeEngine:
            def __init__(self, kind):
                self.name = kind
                self.closed = False
            def think(self, board, player):
                calls.append((self.name, player))
                _, path = gui.cc.get_all_moves(board, player)[0]
                return path, {}
            def quit(self):
                self.closed = True
        def factory(kind):
            eng = FakeEngine(kind)
            engines.append(eng)
            return eng
        def click(x,y):
            return pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(x,y))
        def events():
            nonlocal frames
            frames += 1
            if frames == 1:
                # P1 AI/C-PVS; P2 AI/Clysto; P2 first.
                return [click(480,250), click(350,305), click(720,470), click(500,580)]
            if frames == 3:
                pygame.image.save(pygame.display.get_surface(), '/tmp/cc-gui-menu.png')
                return [click(440,700)]
            if frames == 80:
                return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)]
            if frames == 86:
                return [pygame.event.Event(pygame.QUIT)]
            return []
        with patch.object(gui, 'make_engine', factory), patch.object(pygame.event, 'get', events):
            gui.main()
        self.assertEqual([eng.name for eng in engines], ['c','cly'])
        self.assertEqual(calls[0], ('cly',2))
        self.assertIn(('c',1), calls)
        self.assertTrue(all(eng.closed for eng in engines))

    def test_undo_modes_and_second_player_human(self):
        h=[{'player':2},{'player':1},{'player':2},{'player':1}]
        self.assertEqual(gui.undo_index(h,{1:True,2:False}),2)
        self.assertEqual(gui.undo_index(h,{1:False,2:False}),3)
        self.assertEqual(gui.undo_index(h,{1:True,2:True}),3)
        self.assertEqual(gui.undo_index(h,{1:False,2:True},pending_player=1),4)
        self.assertIsNone(gui.undo_index([],{1:False,2:True},pending_player=2))

    def test_undo_discards_running_search_and_back_button(self):
        frames=0; rendered=[]; states=[]; menu_frames=[]
        board0=gui.cc.init_board()
        path=gui.cc.get_all_moves(board0,1)[0][1]
        class Pending:
            def done(self):return frames>=5
            def cancel(self):return False
            def result(self):raise AssertionError('Stale AI result must never be read')
        class Executor:
            def __init__(self,**kw):pass
            def submit(self,*args):return Pending()
            def shutdown(self,**kw):pass
        class Engine:
            name='fake'
            def think(self,*args):pass
            def quit(self):pass
            def reset_history(self):pass
            def observe_position(self,*args):pass
        def click(pos):return pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,pos=pos)
        def events():
            nonlocal frames
            frames+=1
            if frames==1:return [click((440,700))]
            if frames==2:return [click(gui.pos_to_px(path[0])),click(gui.pos_to_px(path[-1]))]
            if frames==3:return [click(gui.GAME_BUTTONS['undo'].center)]
            if frames==7:
                self.assertEqual(rendered[-1],board0)
                self.assertEqual(states[-1],({1:0,2:0},1,None))
                pygame.image.save(pygame.display.get_surface(),'/tmp/cc-gui-undo.png')
                return [click(gui.GAME_BUTTONS['back'].center)]
            if frames==9:return [pygame.event.Event(pygame.QUIT)]
            return []
        original_board=gui.draw_board;original_status=gui.draw_game_status;original_menu=gui.draw_start_screen
        def draw_board(screen,board,*args):
            rendered.append(board.copy());return original_board(screen,board,*args)
        def draw_status(screen,font,small,counts,current,first,identities,engines,winner):
            states.append((counts.copy(),current,winner))
            return original_status(screen,font,small,counts,current,first,identities,engines,winner)
        def draw_menu(*args):
            menu_frames.append(frames);return original_menu(*args)
        with patch.object(gui,'ThreadPoolExecutor',Executor), patch.object(gui,'make_engine',lambda k:Engine()), \
             patch.object(gui.MoveAnimation,'done',lambda self,now:True), \
             patch.object(pygame.event,'get',events),patch.object(gui,'draw_board',draw_board), \
             patch.object(gui,'draw_game_status',draw_status),patch.object(gui,'draw_start_screen',draw_menu):
            gui.main()
        self.assertIn(7,menu_frames)

    def test_step_labels(self):
        self.assertEqual(gui.step_kind([6,10]), '单步')
        self.assertEqual(gui.step_kind([0,3]), '单跳')
        self.assertEqual(gui.step_kind([0,3,10]), '连跳2次')

if __name__ == '__main__':
    unittest.main()
