"""Independent 2-player CCERL-2P10-v2 rules (players numbered 1 and 2).

Geometry and move logic are implemented here; the optional external JumpStar
binary is used only by verify_121.py as a differential oracle.
"""
from collections import Counter
from dataclasses import dataclass, field

PROFILE = 'CCZ-121-Strict-LG-v2'
DIRECTIONS = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))

def _coordinates():
    cells = {(q, r) for q in range(-4, 5) for r in range(-4, 5)
             if abs(q+r) <= 4}
    for depth in range(1, 5):
        for q in range(depth, 5):
            a, b = q, -4-depth
            for _ in range(6):
                cells.add((a, b))
                a, b = -b, a+b
    return tuple(sorted(cells, key=lambda c: (c[1], c[0])))

COORDS = _coordinates()
IDS = {xy: i for i, xy in enumerate(COORDS)}
HOME = {1: frozenset(range(10)), 2: frozenset(range(111, 121))}
GOAL = {1: HOME[2], 2: HOME[1]}
NEIGHBORS = {s: tuple(IDS[(q+dq, r+dr)] for dq, dr in DIRECTIONS
                       if (q+dq, r+dr) in IDS) for s, (q, r) in enumerate(COORDS)}
HOPS = {s: tuple((IDS[(q+dq, r+dr)], IDS[(q+2*dq, r+2*dr)])
                 for dq, dr in DIRECTIONS
                 if (q+dq, r+dr) in IDS and (q+2*dq, r+2*dr) in IDS)
        for s, (q, r) in enumerate(COORDS)}

def init_board():
    return [1 if i in HOME[1] else 2 if i in HOME[2] else 0 for i in range(121)]

def validate_board(board):
    if len(board) != 121 or any(p not in (0, 1, 2) for p in board):
        raise ValueError('121格棋盘需要121个有效格子')
    if board.count(1) != 10 or board.count(2) != 10:
        raise ValueError('双方必须各有10枚棋子')

def legal_paths(board, start):
    player = board[start]
    if not player:
        return []
    goal = GOAL[player]
    occupied = {i for i, p in enumerate(board) if p and i != start}
    paths = {}
    # Reachability is sufficient: goal membership determines locking, and
    # revisiting a landing cannot reveal any new reachable endpoint.
    visited = {start}
    stack = [(start, [start])]
    while stack:
        here, path = stack.pop()
        for mid, dst in HOPS[here]:
            if mid not in occupied or dst in occupied or dst in visited:
                continue
            if here in goal and dst not in goal:
                continue
            visited.add(dst)
            next_path = path + [dst]
            paths[dst] = next_path
            stack.append((dst, next_path))
    for dst in NEIGHBORS[start]:
        if not board[dst] and (start not in goal or dst in goal):
            paths.setdefault(dst, [start, dst])
    return list(paths.values())

def get_all_moves(board, player):
    return [(s, path) for s, p in enumerate(board) if p == player
            for path in legal_paths(board, s)]

def valid_path(board, player, path):
    if len(path) < 2 or any(type(p) is not int or not 0 <= p < 121 for p in path):
        return False
    if board[path[0]] != player or len(set(path)) != len(path):
        return False
    occupied = {i for i, p in enumerate(board) if p and i != path[0]}
    for s, dst in zip(path, path[1:]):
        if dst in occupied or (s in GOAL[player] and dst not in GOAL[player]):
            return False
        if len(path) == 2 and dst in NEIGHBORS[s]:
            continue
        if not any(d == dst and mid in occupied for mid, d in HOPS[s]):
            return False
    return True

def apply_move(board, path):
    result = board.copy()
    result[path[0]], result[path[-1]] = 0, board[path[0]]
    return result

def terminal(board, player, ply=0, repetitions=None, max_ply=240):
    """Return (winner, reason); winner 0 is draw, None means ongoing."""
    counts = {p: sum(board[i] == p for i in GOAL[p]) for p in (1, 2)}
    if counts[1] == 10:
        if counts[2] == 10 or (player == 2 and counts[2] == 9 and any(
                s not in GOAL[2] and path[-1] in GOAL[2]
                for s, path in get_all_moves(board, 2))):
            return 0, 'equal_turn_goal_draw'
        return 1, 'all_pieces_in_goal'
    if counts[2] == 10:
        return 2, 'all_pieces_in_goal'
    for p in (1, 2):
        if counts[p] and all(board[i] for i in GOAL[p]):
            return p, 'anti_block_goal_full'
    if repetitions is not None and repetitions.get((tuple(board), player), 0) >= 3:
        return 0, 'repetition'
    if max_ply > 0 and ply >= max_ply:
        return 0, 'max_ply'
    return None, ''

@dataclass
class Game:
    # Strict profile fixes top/red as first player. UI seat selection can swap
    # engine identities without changing the profile's equal-turn semantics.
    board: list = field(default_factory=init_board)
    player: int = 1
    ply: int = 0
    max_ply: int = 240
    history: list = field(default_factory=list)
    repetitions: Counter = field(default_factory=Counter)

    def __post_init__(self):
        validate_board(self.board)
        if self.player not in (1, 2):
            raise ValueError('invalid player')
        self.repetitions[(tuple(self.board), self.player)] += 1

    def result(self):
        result = terminal(self.board, self.player, self.ply, self.repetitions, self.max_ply)
        if result[0] is None and not get_all_moves(self.board, self.player):
            return 0, 'no_legal_moves'
        return result

    def play(self, path):
        if self.result()[0] is not None:
            raise ValueError('对局已经结束')
        if not valid_path(self.board, self.player, path):
            raise ValueError('非法走法')
        self.history.append((self.board.copy(), self.player, self.ply))
        self.board = apply_move(self.board, path)
        self.player = 3-self.player
        self.ply += 1
        self.repetitions[(tuple(self.board), self.player)] += 1

    def undo(self):
        if self.history:
            self.repetitions[(tuple(self.board), self.player)] -= 1
            self.board, self.player, self.ply = self.history.pop()
