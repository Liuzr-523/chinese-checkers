"""Exact strict121 moves/terminal and Python evaluator/search differential checks."""
import hashlib,json,random
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import clysto121_native as native
import rules121 as r
from engine121 import Search121
from engine_clysto121 import ClystoDerived121
from verify_c121 import exact

MANIFEST=Path('clysto121_generated/manifest.json')

def reference(board,player):
    # Direct Python port of upstream evaluate(), with documented wing mapping.
    table=json.loads(MANIFEST.read_text())['table81']
    scores={1:0,2:0};last={1:1000000,2:1000000}
    for i,p in enumerate(board):
        if not p:continue
        q,y=r.COORDS[i if p==1 else 120-i];x,z=4-q,y+4+q
        scores[p]+=table[min(8,max(0,z))*9+min(8,max(0,x))]
        last[p]=min(last[p],y+8)
    for p in (1,2):scores[p]-=1<<max(0,4-last[p])
    if last[2]==13:scores[2],scores[1]=10000,0
    if last[1]==13:scores[1],scores[2]=10000,0
    return scores[player]-scores[3-player]

def main():
    rng=random.Random(301042);samples=[];g=r.Game()
    for i in range(240):
        if i%2:
            ids=rng.sample(range(121),20);b=[0]*121
            for j,s in enumerate(ids):b[s]=1+j//10
        else:b=g.board.copy()
        for p in (1,2):
            assert set(native.moves(b,p))=={(s,path[-1]) for s,path in r.get_all_moves(b,p)}
            assert native.evaluate(b,p)==reference(b,p),(i,p)
            win=r.terminal(b,p)[0]
            assert native.terminal(b,p)==(-1 if win is None else win)
        if i<20:samples.append(b)
        if g.result()[0] is not None:g=r.Game()
        else:g.play(rng.choice(r.get_all_moves(g.board,g.player))[1])
    with patch.object(Search121,'evaluate',staticmethod(reference)):
        for i,b in enumerate(samples):
            p=1+i%2;depth=3 if i<4 else 2;counts=Counter({(tuple(b),p):1})
            expected=exact(b,p,depth,0,counts)
            path,info=ClystoDerived121(seconds=60,max_depth=depth).think(b,p,0,counts)
            assert not info['aborted'] and info['score']==expected,(i,info,expected)
            child=r.apply_move(b,path);counts[(tuple(child),3-p)]+=1
            assert -exact(child,3-p,depth-1,1,counts)==expected
    b=r.init_board();h=Counter({(tuple(b),1):1})
    for _,path in r.get_all_moves(b,1):h[(tuple(r.apply_move(b,path)),2)]=2
    _,info=ClystoDerived121(max_depth=3).think(b,1,0,h)
    assert info['score']==0
    for ply,h in ((240,Counter()),(0,Counter({(tuple(b),1):3}))):
        path,info=ClystoDerived121().think(b,1,ply,h)
        assert path is None and info['score']==0
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [MANIFEST,Path('clysto121_generated/core.c'),Path(native.__file__)]}
    result=dict(positions=240,sides_per_position=2,search_positions=20,history_checks=True,failures=0,sha256=hashes,scope='Exact Python121 legality and explicitly adapted evaluator; not original Clysto strength.')
    Path('clysto121_generated/verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
