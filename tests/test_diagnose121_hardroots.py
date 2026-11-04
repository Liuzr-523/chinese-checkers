import unittest
from diagnostics.diagnose121_hardroots import learning_history
class HistoryTests(unittest.TestCase):
 def test_forgetting_is_observed_transition_not_inferred_from_final(self):
  h=learning_history([dict(step=0,regret=8),dict(step=100,regret=0),dict(step=200,regret=5)])
  self.assertTrue(h['recorded_forgetting']);self.assertEqual(h['first_recorded_good'],100);self.assertTrue(h['final_bad'])
 def test_never_observed_good_does_not_count_as_forgetting(self):
  h=learning_history([dict(step=0,regret=8),dict(step=100,regret=1),dict(step=200,regret=5)])
  self.assertFalse(h['recorded_forgetting']);self.assertFalse(h['ever_recorded_good'])
if __name__=='__main__':unittest.main()
