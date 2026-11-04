import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
import unittest
from unittest.mock import patch
import pygame
import gui121
import rules121 as r

class Gui121Tests(unittest.TestCase):
    def test_geometry_render_and_animation(self):
        pygame.init()
        screen = pygame.display.set_mode((880, 960))
        for s in range(121):
            self.assertEqual(gui121.px_to_pos(gui121.pos_to_px(s)), s)
        b = r.init_board()
        gui121.draw_board(screen, b)
        pygame.image.save(screen, '/tmp/checkers121-board.png')
        path = r.get_all_moves(b, 1)[0][1]
        animation = gui121.Animation(b, path)
        for t in (0, .25, animation.duration-.01): animation.draw(screen, t)
        self.assertEqual(b, r.init_board())
        pygame.quit()

    def test_native_engine_selected_and_receives_game_history(self):
        import engine_c121
        frame = 0
        calls = []
        class FakeNative:
            def think(self, board, player, ply, repetitions):
                calls.append((player, ply, repetitions.copy()))
                return r.get_all_moves(board, player)[0][1], {}
        def click(pos): return pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)
        def events():
            nonlocal frame
            frame += 1
            if frame == 1:
                return [click((400, 280)), click((400, 280)), click((400, 520))]
            if frame == 5: return [pygame.event.Event(pygame.QUIT)]
            return []
        with patch.object(engine_c121, 'CSearch121', return_value=FakeNative()) as factory, \
             patch.object(pygame.event, 'get', events):
            gui121.main()
        factory.assert_called_once()
        self.assertEqual(calls[0][:2], (1, 0))
        self.assertEqual(calls[0][2][(tuple(r.init_board()), 1)], 1)

    def test_registered_neural_engine_can_be_selected(self):
        import neural121_registry
        frames=0;calls=[]
        class FakeNeural:
            def think(self,board,player,ply,repetitions):
                calls.append((player,ply));return r.get_all_moves(board,player)[0][1],{}
        def click(pos):return pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,pos=pos)
        def events():
            nonlocal frames
            frames+=1
            if frames==1:return [click((400,280)) for _ in range(3)]+[click((400,520))]
            if frames==6:return [pygame.event.Event(pygame.QUIT)]
            return []
        with patch.object(neural121_registry,'load_registered',return_value=FakeNeural()) as factory,patch.object(pygame.event,'get',events):
            gui121.main()
        factory.assert_called_once();self.assertEqual(calls[0],(1,0))

    def test_move_undo_and_return(self):
        frame = 0; observed = []
        path = r.get_all_moves(r.init_board(), 1)[0][1]
        def click(pos): return pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)
        def events():
            nonlocal frame
            frame += 1
            if frame == 1: return [click((400, 520))]
            if frame == 2: return [click(gui121.pos_to_px(path[0])), click(gui121.pos_to_px(path[-1]))]
            if frame == 3: return [click((220, 905))]
            if frame == 4: return [click((80, 905))]
            if frame == 5: return [pygame.event.Event(pygame.QUIT)]
            return []
        real = gui121.draw_board
        def draw(screen, board, *args):
            observed.append(board.copy()); return real(screen, board, *args)
        with patch.object(pygame.event, 'get', events), patch.object(gui121, 'draw_board', draw):
            gui121.main()
        self.assertEqual(observed[-1], r.init_board())
        self.assertGreaterEqual(frame, 5)

if __name__ == '__main__': unittest.main()
