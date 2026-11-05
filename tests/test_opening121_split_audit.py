import unittest
from opening121 import OpeningBook
from train_opening121 import book_samples
from opening121_split_audit import orbit, mirror, rotate_swap, decision_key, audit_split


class SplitAuditTests(unittest.TestCase):
    def test_symmetry_invariance(self):
        board = [0] * 121
        board[0], board[14], board[90] = 1, 1, 2
        identity = decision_key(board, 1)
        self.assertEqual(orbit(identity), orbit(mirror(identity)))
        self.assertEqual(orbit(identity), orbit(rotate_swap(identity)))
        self.assertEqual(orbit(identity), orbit(mirror(rotate_swap(identity))))
        self.assertEqual(identity, mirror(mirror(identity)))
        self.assertEqual(identity, rotate_swap(rotate_swap(identity)))

    def test_cross_orbit_counts(self):
        a = decision_key([1] + [0] * 120, 1)
        b = decision_key([1, 1] + [0] * 119, 1)
        result = audit_split([a, mirror(a), b], [rotate_swap(a)])
        self.assertEqual(result['overlap_orbits'], 1)
        self.assertEqual(result['overlap_train_cases'], 2)
        self.assertEqual(result['overlap_validation_cases'], 1)
        self.assertEqual(result['overlapping_orbits'][0]['validation'], [rotate_swap(a)])
        self.assertEqual(audit_split([a], [b])['overlap_orbits'], 0)

    def test_current_book_regression(self):
        sets, _ = book_samples(OpeningBook())
        result = audit_split([s['decision_id'] for s in sets['train']],
                             [s['decision_id'] for s in sets['validation']])
        self.assertEqual((result['train_cases'], result['validation_cases']), (565, 128))
        self.assertEqual((result['overlap_orbits'], result['overlap_train_cases'],
                          result['overlap_validation_cases']), (19, 38, 38))

    def test_invalid_identity_rejected(self):
        for identity in ('0:1', '0' * 121 + ':3', 'x' * 121 + ':1'):
            with self.assertRaises(ValueError):
                orbit(identity)


if __name__ == '__main__':
    unittest.main()
