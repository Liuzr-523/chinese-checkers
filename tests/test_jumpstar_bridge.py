import unittest
import Chinese_Checker as cc
import engine_jumpstar as j

class BridgeTests(unittest.TestCase):
    @unittest.skipUnless(j.DEFAULT_ENGINE.is_file(), 'optional JumpStar referee binary absent')
    def test_initial_geometry_and_legal_moves(self):
        b=cc.init_board();info=j.reference(b,1,0)
        self.assertEqual(j.encode(b),'0'*10+'.'*101+'1'*10)
        local={(j.LOCAL_TO_STAR[s],j.LOCAL_TO_STAR[p[-1]]) for s,p in cc.get_all_moves(b,1)}
        self.assertEqual(local,{(m['from'],m['to']) for m in info['legal']})
        self.assertEqual(j.checked_path(b,1,[6,15],info),[6,11])

    @unittest.skipUnless(j.DEFAULT_ENGINE.is_file(), 'optional JumpStar referee binary absent')
    def test_search_receives_updated_ply(self):
        from unittest.mock import patch
        eng=j.JumpStarEngine();eng.set_ply(17)
        board=cc.init_board();info=j.reference(board,1,0)
        with patch.object(j,'reference',return_value=info) as ref, \
             patch.object(j,'remote_move',return_value={'move':{'from':6,'to':15,'path':[6,15]}}) as remote:
            path,_=eng.think(board,1)
        self.assertEqual(path,[6,11])
        self.assertEqual(ref.call_args.args[2],17)
        self.assertEqual(remote.call_args.args[2],17)

    def test_side_triangle_is_rejected(self):
        off=next(x for x in range(121) if x not in j.STAR_TO_LOCAL)
        with self.assertRaises(j.RuleMismatch):
            j.checked_path(cc.init_board(),1,[6,off],{'legal':[{'from':6,'to':off}]})

    def test_illegal_return_is_rejected(self):
        with self.assertRaises(j.RuleMismatch):
            j.checked_path(cc.init_board(),1,[6,15],{'legal':[]})

if __name__=='__main__':unittest.main()
