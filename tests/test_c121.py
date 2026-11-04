import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import math
import ccore121
import rules121 as r
from engine_c121 import CSearch121
from verify_c121 import exact

class Native121Tests(unittest.TestCase):
    def test_rejects_malformed_inputs(self):
        for board,p in [([0]*81,1),([0]*121,1),(r.init_board(),0),(r.init_board(),3)]:
            with self.assertRaises(ValueError): ccore121.moves(board,p)
        for seconds,depth in [(float('nan'),2),(float('inf'),2),(-1,2),(1,0),(1,63)]:
            with self.assertRaises(ValueError): ccore121.search(r.init_board(),1,seconds,depth,0,[])
        with self.assertRaises(ValueError):
            ccore121.search(r.init_board(),1,1.,2,0,[(r.init_board(),1,-1)])

    def test_terminal_cases(self):
        b=[0]*121
        for i in r.GOAL[1]: b[i]=1
        for i in r.GOAL[2]: b[i]=2
        cases=[b.copy()]
        b[9]=0; b[next(i for i in r.NEIGHBORS[9] if i not in r.GOAL[2])]=2
        cases.append(b.copy())
        b=r.init_board(); b[0],b[111]=b[111],b[0]; cases.append(b)
        for b in cases:
            for p in (1,2):
                self.assertEqual(ccore121.terminal(b,p), r.terminal(b,p)[0])
                path,info=CSearch121().think(b,p)
                self.assertIsNone(path)
                winner=r.terminal(b,p)[0]
                self.assertEqual(info['score'],0 if winner==0 else 100000 if winner==p else -100000)

    def test_history_and_limit_affect_search(self):
        b=r.init_board(); history=Counter({(tuple(b),1):1})
        for _,path in r.get_all_moves(b,1): history[(tuple(r.apply_move(b,path)),2)]=2
        _,info=CSearch121(max_depth=3).think(b,1,0,history)
        self.assertEqual(info['score'],0)
        _,info=CSearch121(max_depth=3).think(b,1,239)
        self.assertEqual(info['score'],0)
        path,_=CSearch121().think(b,1,240)
        self.assertIsNone(path)

    def test_concurrent_and_timeout(self):
        b=r.init_board()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(CSearch121(seconds=.03).think,b,p) for p in (1,2)]
            for p,f in zip((1,2),futures):
                path,info=f.result()
                self.assertTrue(r.valid_path(b,p,path)); self.assertTrue(info['aborted'])
        path,info=CSearch121(seconds=1e-9).think(b,1)
        self.assertTrue(r.valid_path(b,1,path)); self.assertEqual(info['depth_reached'],0)
        self.assertEqual(b,r.init_board())

if __name__=='__main__': unittest.main()
