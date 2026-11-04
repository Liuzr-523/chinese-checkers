"""Experimental 81/121 interoperability probe against official online JumpStar.

A completed game is only a restricted 81-cell diagnostic, not an official CCERL
match. Stop on unsupported moves or differing termination; never count as a loss.
"""
import argparse,hashlib,json,subprocess,time,traceback
from pathlib import Path
import Chinese_Checker as cc
from engine_c import CESEngine
from engine_jumpstar import (JumpStarEngine,RuleMismatch,reference,checked_path,LOCAL_TO_STAR,STAR_TO_LOCAL,encode)


def local_allowed(board,player):
    return {(LOCAL_TO_STAR[s],LOCAL_TO_STAR[p[-1]]) for s,p in cc.get_all_moves(board,player)}


def game(our_player,out,our_time,simulations,book,max_moves):
    board=cc.init_board();steps=[];seen=set();our=CESEngine(time_limit_sec=our_time,max_depth=20,eval_version='v2',use_book=book,book_seed=42)
    opponent=JumpStarEngine(simulations=simulations)
    result=dict(our_player=our_player,winner=None,reason='move_limit',steps=steps,offboard_moves_available=0,rule_disagreement_positions=0)
    def save():
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2));tmp.replace(out)
    for ply in range(max_moves):
        player=1+ply%2
        try:
            official=reference(board,player,ply)
            terminal=official['terminal'];local_winner=cc.is_game_over(board)
            if terminal['terminal'] or local_winner is not None:
                if terminal['terminal'] and not terminal['draw'] and terminal['winner'] is not None and local_winner==terminal['winner']+1:
                    result.update(winner=local_winner,reason='agreed_win')
                else:
                    result.update(reason='terminal_rule_mismatch',official_terminal=terminal,local_winner=local_winner)
                break
            key=(tuple(board),player)
            if key in seen:
                # Stop on repetition without calling it a draw under local rules.
                result.update(reason='repeated_position');break
            seen.add(key)
            local=local_allowed(board,player);official_moves={(x['from'],x['to']) for x in official['legal']}
            if local!=official_moves:result['rule_disagreement_positions']+=1
            result['offboard_moves_available']+=sum(a not in STAR_TO_LOCAL or b not in STAR_TO_LOCAL for a,b in official_moves)
            start=time.monotonic()
            if player==our_player:
                path,info=our.think(board,player)
                if path is None:result.update(reason='our_no_move');break
                # C-PVS returns endpoints; reconstruct the legal multi-hop witness.
                path = next((p for s,p in cc.get_all_moves(board,player)
                             if s==path[0] and p[-1]==path[-1]),None)
                if path is None:raise RuleMismatch('C-PVS 返回的起终点不合法')
                mpath=[LOCAL_TO_STAR[i] for i in path]
                checked_path(board,player,mpath,official)
                details=info
            else:
                opponent.ply=ply
                path,info=opponent.think(board,player)
                details={'model':info['remote'].get('champion'), 'stats':info['remote'].get('stats'),
                         'root_moves':info['remote'].get('root_moves')}
            steps.append(dict(ply=ply,player=player,by='ours' if player==our_player else 'jumpstar',
                              path=path,board=board.copy(),time_seconds=time.monotonic()-start,details=details))
            board=cc.apply_move(board,path);save()
            print(f"P{our_player} ply={ply+1} {steps[-1]['by']} {path[0]}->{path[-1]}",flush=True)
        except RuleMismatch as exc:
            result.update(reason='incompatible_rules',error=str(exc),stopped_ply=ply);break
        except Exception as exc:
            result.update(reason='error',error=f'{type(exc).__name__}: {exc}',stopped_ply=ply);break
    else:
        result.update(reason='move_limit')
    save();return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--games',type=int,default=2);ap.add_argument('--our-time',type=float,default=1)
    ap.add_argument('--simulations',type=int,default=512);ap.add_argument('--max-moves',type=int,default=240)
    ap.add_argument('--no-book',action='store_true');args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd='nn_checkpoints/third_party/jumpstar',text=True).strip()
    meta=dict(our_engine='C-PVS v2',our_time=args.our_time,our_book=not args.no_book,jumpstar='official hosted service',simulations=args.simulations,
              referee_revision=revision,caveat='Opponent uses full 121 board; our engine is restricted 81. Results are diagnostic only, not official equal-rules strength comparison.')
    (args.out/'config.json').write_text(json.dumps(meta,indent=2))
    results=[]
    for i in range(args.games):
        res=game(1+i%2,args.out/f'game_{i+1}.json',args.our_time,args.simulations,not args.no_book,args.max_moves)
        results.append({k:v for k,v in res.items() if k!='steps'}|{'plies':len(res['steps'])})
        (args.out/'summary.json').write_text(json.dumps(dict(config=meta,games=results),ensure_ascii=False,indent=2))
        print('RESULT',json.dumps(results[-1],ensure_ascii=False),flush=True)

if __name__=='__main__':main()
