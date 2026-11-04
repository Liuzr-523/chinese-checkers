import unittest
from iterate121 import paired_gate
class IterationTests(unittest.TestCase):
    def test_gate_counts_opening_pairs(self):
        rows=[dict(pair=p,side=s,winner=s) for p in range(20) for s in (1,2)]
        self.assertTrue(paired_gate(rows)['passed'])
        rows=[dict(pair=p,side=s,winner=0) for p in range(20) for s in (1,2)]
        self.assertFalse(paired_gate(rows)['passed'])
        with self.assertRaises(ValueError):paired_gate([dict(pair=0,side=1,winner=1)])
if __name__=='__main__':unittest.main()
