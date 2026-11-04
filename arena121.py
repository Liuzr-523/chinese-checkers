"""Same-rule native-vs-Python baseline matches; no online JumpStar claims."""
import argparse
import json
import random
import time
from pathlib import Path
import rules121 as r
from engine121 import Search121
from engine_c121 import CSearch121

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--games',type=int,default=4)
    parser.add_argument('--seconds',type=float,default=.1)
    parser.add_argument('--output',type=Path,default=Path('nn_checkpoints/experiments/native121_20261029/arena.json'))
    args=parser.parse_args()
    if args.games<2 or args.games%2: parser.error('games must be positive even paired games')
    rows=[]; started=time.monotonic()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for index in range(args.games):
        rng=random.Random(12129+index//2); g=r.Game()
        opening=[]
        for _ in range(6):
            path=rng.choice(r.get_all_moves(g.board,g.player))[1]; g.play(path); opening.append(path)
        cplayer=1+index%2
        engines={cplayer:CSearch121(seconds=args.seconds),3-cplayer:Search121(seconds=args.seconds,max_depth=12)}
        moves=[]; start=time.monotonic()
        while g.result()[0] is None:
            player=g.player
            path,info=engines[player].think(g.board,player,g.ply,g.repetitions)
            if not path or not r.valid_path(g.board,player,path): raise RuntimeError('invalid move')
            g.play(path)
            moves.append({'player':player,'path':path,'depth':info['depth_reached'],'nodes':info['nodes']})
        winner,reason=g.result()
        rows.append({'game':index+1,'c_player':cplayer,'winner':winner,'reason':reason,
                     'plies':g.ply,'seconds':time.monotonic()-start,'opening':opening,'moves':moves})
        summary={'profile':r.PROFILE,'c_wins':sum(x['winner']==x['c_player'] for x in rows),
                 'python_wins':sum(x['winner']==3-x['c_player'] for x in rows),
                 'draws':sum(x['winner']==0 for x in rows),'games':len(rows),'seconds_per_move':args.seconds}
        args.output.write_text(json.dumps({'summary':summary,'games':rows,'wall_seconds':time.monotonic()-started},indent=2))
        print(json.dumps({**summary,'last_reason':reason,'last_plies':g.ply}),flush=True)

if __name__=='__main__': main()
