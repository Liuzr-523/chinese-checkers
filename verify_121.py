"""Differential checks against the separately installed JumpStar referee."""
import argparse
import json
import random
import subprocess
from pathlib import Path
import rules121 as r

DEFAULT = Path(__file__).resolve().parent/'nn_checkpoints/third_party/jumpstar/build/cczero'

def reference(board, player, ply, binary=DEFAULT):
    cells = ''.join('.' if p == 0 else str(p-1) for p in board)
    proc = subprocess.run([str(binary), 'position-info', '--cells', cells,
                           '--player', str(player-1), '--ply', str(ply), '--rules', 'strict'],
                          check=True, capture_output=True, text=True, timeout=10)
    return json.loads(proc.stdout)

def check(board, player, ply, binary=DEFAULT):
    official = reference(board, player, ply, binary)
    moves = r.get_all_moves(board, player)
    ours = {(s, path[-1]) for s, path in moves}
    theirs = {(m['from'], m['to']) for m in official['legal']}
    assert ours == theirs, (player, ply, ours-theirs, theirs-ours)
    assert all(r.valid_path(board, player, path) for _, path in moves)
    for move in official['legal']:
        assert r.valid_path(board, player, move['path']), move
    # position-info has no match max-ply/history; those are unit-tested separately.
    winner, reason = r.terminal(board, player, ply, max_ply=0)
    actual = official['terminal']
    assert (winner is not None) == actual['terminal'], (winner, reason, actual)
    if winner is not None:
        assert reason == actual['reason'], (reason, actual)
        assert winner == (0 if actual['draw'] else actual['winner']+1), (winner, actual)
    return moves

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--positions', type=int, default=400)
    parser.add_argument('--engine', type=Path, default=DEFAULT)
    args = parser.parse_args()
    rng = random.Random(12120261028)
    board = r.init_board(); player = 1; ply = 0
    for n in range(args.positions):
        # Half reachable playouts, half arbitrary 10v10 positions including arms.
        if n % 2:
            sample = rng.sample(range(121), 20)
            b = [0]*121
            for i, s in enumerate(sample): b[s] = 1 if i < 10 else 2
            check(b, 1+n%3%2, n%241, args.engine)
        else:
            moves = check(board, player, ply, args.engine)
            if not moves or r.terminal(board, player, ply, max_ply=0)[0] is not None:
                board = r.init_board(); player = 1; ply = 0
            else:
                board = r.apply_move(board, rng.choice(moves)[1]); player = 3-player; ply += 1
    # Goal completion, both full, equal-turn immediate reply, and anti-block.
    cases = []
    b = [0]*121
    for i in r.GOAL[1]: b[i] = 1
    for i in r.GOAL[2]: b[i] = 2
    cases.append((b.copy(), 1, 180))
    b[9] = 0; b[next(i for i in r.NEIGHBORS[9] if i not in r.GOAL[2])] = 2
    cases.append((b.copy(), 1, 179)); cases.append((b.copy(), 2, 179))
    b = r.init_board(); b[0], b[111] = b[111], b[0]
    cases.append((b, 2, 1))
    for b, p, ply in cases: check(b, p, ply, args.engine)
    print(json.dumps({'positions': args.positions+len(cases), 'failures': 0,
                      'checks': ['move_endpoints', 'witness_paths', 'terminal'],
                      'profile': r.PROFILE}))

if __name__ == '__main__': main()
