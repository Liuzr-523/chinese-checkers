import unittest
from collections import Counter
import rules121 as r
from opening121 import OpeningBook,BookEngine,MIRROR,key
class OpeningTests(unittest.TestCase):
 def test_three_moves_both_sides_and_all_entries_legal(self):
  book=OpeningBook();g=r.Game()
  expected=[[6,15],[114,105],[1,6,28],[119,114,92],[5,16,39],[115,104,81]]
  for path in expected:
   got=book.choose(g.board,g.player,g.ply,g.repetitions);self.assertEqual((got[0],got[-1]),(path[0],path[-1]));self.assertTrue(r.valid_path(g.board,g.player,got));g.play(got)
  for state,entry in book.data['nodes'].items():
   board,player=state.split(':');board=list(map(int,board));self.assertTrue(all(r.valid_path(board,int(player),p) for p in entry['moves']))
 def test_mirror_and_repetition_and_departure(self):
  book=OpeningBook();g=r.Game();self.assertEqual(book.moves(g.board,1,20),[]);self.assertEqual(book.moves(g.board,1,0,Counter({(tuple(g.board),1):3})),[])
  g.play([6,15]);b=[0]*121
  for i,v in enumerate(g.board):b[MIRROR[i]]=v
  self.assertTrue(book.moves(b,2,1));g.play([111,103]);g.play([15,14]);self.assertEqual(book.moves(g.board,g.player,g.ply),[])
 def test_miss_falls_back_and_hit_is_identified(self):
  class Engine:
   def think(self,b,p,ply,rep):return r.get_all_moves(b,p)[0][1],{'nodes':123}
  e=BookEngine(Engine());b=r.init_board();path,info=e.think(b,1);self.assertTrue(info['book_hit']);_,info=e.think(b,1,20);self.assertFalse(info['book_hit']);self.assertEqual(info['nodes'],123)
class BookTrainingTests(unittest.TestCase):
 def test_reflection_split_has_no_orbit_overlap(self):
  from train_opening121 import book_samples
  sets,groups=book_samples(OpeningBook());orbits={}
  for part,samples in sets.items():
   self.assertTrue(samples)
   for s in samples:
    board,p=s['decision_id'].split(':');b=list(map(int,board));ref=[0]*121
    for i,v in enumerate(b):ref[MIRROR[i]]=v
    orbit=min(s['decision_id'],key(ref,int(p)))
    if orbit in orbits:self.assertEqual(orbits[orbit],part)
    orbits[orbit]=part
 def test_set_loss_accepts_any_recommended_move(self):
  import torch
  from train_opening121 import set_ce
  a=torch.tensor([[1.,1.,-2.]],requires_grad=True);s=dict(actions=[0,1,2],positive=[0,1],target=0)
  loss=set_ce(a,[s]);g=torch.autograd.grad(loss,a)[0];self.assertLess(float(g[0,0]),0);self.assertLess(float(g[0,1]),0);self.assertGreater(float(g[0,2]),0)
if __name__=='__main__':unittest.main()
