import unittest
from collections import Counter
import rules121 as r
from engine121 import Search121

class Rules121Tests(unittest.TestCase):
    def test_geometry_and_side_arms(self):
        self.assertEqual(len(r.COORDS), 121)
        self.assertEqual(len(set(r.COORDS)), 121)
        self.assertEqual(sum(len(v) for v in r.NEIGHBORS.values()) % 2, 0)
        for s, neighbors in r.NEIGHBORS.items():
            for dst in neighbors: self.assertIn(s, r.NEIGHBORS[dst])
        b = [0]*121; s = r.IDS[(4, 0)]; b[s] = 1
        self.assertTrue(any(r.COORDS[path[-1]][0] > 4 for path in r.legal_paths(b, s)))

    def test_goal_lock_and_path_validation(self):
        b = [0]*121; b[9] = 2
        self.assertTrue(r.legal_paths(b, 9))
        self.assertTrue(all(set(path) <= r.GOAL[2] for path in r.legal_paths(b, 9)))
        dst = next(i for i in r.NEIGHBORS[9] if i not in r.GOAL[2])
        self.assertFalse(r.valid_path(b, 2, [9, dst]))
        self.assertFalse(r.valid_path(b, 2, [9, 8, 9]))
        self.assertFalse(r.valid_path(b, 1, [9, 8]))

    def test_repetition_and_undo(self):
        g = r.Game(); original = g.board.copy()
        path = r.get_all_moves(g.board, 1)[0][1]
        g.play(path); g.undo()
        self.assertEqual(g.board, original); self.assertEqual(g.ply, 0)
        self.assertEqual(g.repetitions[(tuple(original), 1)], 1)
        g.repetitions[(tuple(original), 1)] = 3
        self.assertEqual(g.result(), (0, 'repetition'))
        with self.assertRaises(ValueError): g.play(path)
        self.assertEqual(r.terminal(original, 1, 240), (0, 'max_ply'))

    def test_anti_block_even_with_own_home_occupied(self):
        b = r.init_board(); b[0], b[111] = b[111], b[0]
        self.assertEqual(r.terminal(b, 2), (1, 'anti_block_goal_full'))

    def test_search_returns_legal_move_without_mutating(self):
        b = r.init_board(); copy = b.copy()
        path, info = Search121(seconds=.05).think(b, 1)
        self.assertTrue(r.valid_path(b, 1, path))
        self.assertEqual(b, copy)
        self.assertGreater(info['nodes'], 0)

if __name__ == '__main__': unittest.main()
