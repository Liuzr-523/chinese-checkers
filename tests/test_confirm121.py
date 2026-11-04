import unittest
from confirm121_unfrozen import fresh_opening,start_key
from neural121_registry import audit_game_counts

class Confirmation121Tests(unittest.TestCase):
    def test_declared_budgets_are_enforced_without_weakening_old_audit(self):
        self.assertEqual(audit_game_counts({}),[('confirmation',60),('fixed_c_control',20),('fixed_c_candidate',20)])
        self.assertEqual(audit_game_counts(dict(confirmation_games=120,fixed_c_games_per_model=40)),[('confirmation',120),('fixed_c_control',40),('fixed_c_candidate',40)])
        for config in [dict(confirmation_games=59),dict(confirmation_games=61),dict(fixed_c_games_per_model=18),dict(fixed_c_games_per_model='40')]:
            with self.assertRaises(ValueError):audit_game_counts(config)

    def test_freshness_excludes_transposed_start_positions_and_is_reproducible(self):
        used=set();a=fresh_opening(301048,4,used);b=fresh_opening(301048,4,used)
        self.assertNotEqual(start_key(a),start_key(b));self.assertEqual(len(used),2)
        replayed=set();self.assertEqual(a,fresh_opening(301048,4,replayed));self.assertEqual(b,fresh_opening(301048,4,replayed))

if __name__=='__main__':unittest.main()
