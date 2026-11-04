"""Native move/eval/terminal/search parity, history boundaries and speed."""
import json
import random
import time
from collections import Counter
from pathlib import Path
import ccore121 as native
import rules121 as r
from engine121 import Search121
from engine_c121 import CSearch121

# Deliberately plain alpha-beta reference: no PVS, no TT, no null moves.
def exact(board, player, depth, ply, counts, alpha=-1000000, beta=1000000):
    win, _ = r.terminal(board, player, ply, counts)
    if win is not None:
        return 0 if win == 0 else 100000 if win == player else -100000
    moves = r.get_all_moves(board, player)
    if not moves: return 0
    if not depth: return Search121.evaluate(board, player)
    value = -1000000
    for _, path in moves:
        child = r.apply_move(board, path); key = (tuple(child), 3-player)
        counts[key] += 1
        try: score = -exact(child, 3-player, depth-1, ply+1, counts, -beta, -alpha)
        finally: counts[key] -= 1
        value = max(value, score); alpha = max(alpha, value)
        if alpha >= beta: break
    return value

def main():
    rng = random.Random(20261029); samples = []; g = r.Game()
    for i in range(240):
        if i % 2:
            ids = rng.sample(range(121), 20); board = [0]*121
            for j,s in enumerate(ids): board[s] = 1+j//10
            p = 1+i%3%2
        else:
            board = g.board; p = g.player
        assert set(native.moves(board, p)) == {(s,path[-1]) for s,path in r.get_all_moves(board,p)}
        for side in (1,2): assert native.evaluate(board,side) == Search121.evaluate(board,side)
        win = r.terminal(board,p)[0]
        assert native.terminal(board,p) == (-1 if win is None else win)
        if i < 20: samples.append((board.copy(), p))
        if i % 2 == 0:
            if g.result()[0] is not None: g = r.Game()
            else: g.play(rng.choice(r.get_all_moves(g.board,g.player))[1])
    for i,(board,p) in enumerate(samples):
        depth = 3 if i < 4 else 2
        counts = Counter({(tuple(board),p):1})
        expected = exact(board,p,depth,0,counts)
        path, info = CSearch121(seconds=60,max_depth=depth).think(board,p,0,counts)
        assert not info['aborted'] and info['depth_reached'] == depth, info
        assert info['score'] == expected, (i,info,expected)
        child=r.apply_move(board,path); key=(tuple(child),3-p); counts[key]+=1
        assert -exact(child,3-p,depth-1,1,counts) == expected, (i,path)
    b=r.init_board()
    for ply,counts in ((240,Counter()),(0,Counter({(tuple(b),1):3}))):
        path,info=CSearch121().think(b,1,ply,counts)
        assert path is None and info['score']==0
    # Every child is a third repetition: search must prefer the correct draw.
    counts=Counter({(tuple(b),1):1})
    for _,path in r.get_all_moves(b,1): counts[(tuple(r.apply_move(b,path)),2)]=2
    _,info=CSearch121(max_depth=3).think(b,1,0,counts)
    assert info['score']==0 and info['depth_reached']==3
    times={}
    for name,engine in [('python',Search121(seconds=1,max_depth=30)),('c',CSearch121(seconds=1,max_depth=30))]:
        start=time.monotonic();path,info=engine.think(b,1);elapsed=time.monotonic()-start
        assert r.valid_path(b,1,path)
        times[name]={'seconds':elapsed,'nodes':info['nodes'],'depth':info['depth_reached'], 'nodes_per_second':info['nodes']/elapsed}
    result={'positions':240,'search_positions':20,'failures':0,'speed':times,
            'throughput_ratio':times['c']['nodes_per_second']/times['python']['nodes_per_second'],
            'note':'Throughput is not playing strength; PVS vs Python alpha-beta.'}
    folder=Path('nn_checkpoints/experiments/native121_20261029'); folder.mkdir(parents=True,exist_ok=True)
    (folder/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
