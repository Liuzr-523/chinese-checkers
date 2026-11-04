"""Public entry point must launch the 121-cell GUI."""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
import unittest
from unittest.mock import patch
import pygame
import gui
import gui121

class GuiEntryTests(unittest.TestCase):
    def test_public_entry_starts_121_cell_game(self):
        frame = 0
        boards = []
        draw = gui121.draw_board
        def events():
            nonlocal frame
            frame += 1
            if frame == 1:
                return [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(400, 520))]
            return [pygame.event.Event(pygame.QUIT)]
        def observe(screen, board, *args):
            boards.append(board.copy())
            self.assertIn('121格', pygame.display.get_caption()[0])
            return draw(screen, board, *args)
        with patch.object(pygame.event, 'get', events), patch.object(gui121, 'draw_board', observe):
            gui.main()
        self.assertTrue(boards)
        self.assertTrue(all(len(board) == 121 for board in boards))

if __name__ == '__main__': unittest.main()
