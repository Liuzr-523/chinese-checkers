"""Small reference alpha-beta engine for the 121-cell rules, not C-PVS."""
import time
import rules121 as rules

class Search121:
    name = '121格参考搜索'
    def __init__(self, seconds=0.5, max_depth=4):
        self.seconds, self.max_depth = seconds, max_depth

    @staticmethod
    def evaluate(board, player):
        def cost(p):
            # A fixed goal tip encourages forward progress; this is a baseline
            # heuristic, not a trained evaluation or a claim of strong play.
            tip = rules.COORDS[120 if p == 1 else 0]
            distances = []
            for i, occupant in enumerate(board):
                if occupant == p:
                    q, r = rules.COORDS[i]
                    dq, dr = q-tip[0], r-tip[1]
                    distances.append(max(abs(dq), abs(dr), abs(dq+dr)))
            return sum(distances) + max(distances, default=0)*2
        return cost(3-player)-cost(player)

    def think(self, board, player, ply=0, repetitions=None):
        from collections import Counter
        counts = Counter(repetitions or {(tuple(board), player): 1})
        deadline = time.monotonic()+self.seconds
        nodes = 0
        def search(b, p, n, depth, alpha, beta):
            nonlocal nodes
            nodes += 1
            if time.monotonic() >= deadline:
                raise TimeoutError
            winner, _ = rules.terminal(b, p, n, counts)
            if winner is not None:
                return 0 if winner == 0 else (100000 if winner == player else -100000)
            if depth == 0:
                return self.evaluate(b, player)
            moves = rules.get_all_moves(b, p)
            if not moves:
                return 0
            moves.sort(key=lambda m: self.evaluate(rules.apply_move(b, m[1]), p), reverse=True)
            best = -float('inf') if p == player else float('inf')
            for _, path in moves:
                child = rules.apply_move(b, path)
                key = (tuple(child), 3-p)
                counts[key] += 1
                try:
                    value = search(child, 3-p, n+1, depth-1, alpha, beta)
                finally:
                    counts[key] -= 1
                if p == player:
                    best = max(best, value); alpha = max(alpha, best)
                else:
                    best = min(best, value); beta = min(beta, best)
                if alpha >= beta:
                    break
            return best
        moves = rules.get_all_moves(board, player)
        if not moves:
            return None, {'nodes': 0}
        moves.sort(key=lambda m: self.evaluate(rules.apply_move(board, m[1]), player), reverse=True)
        best = moves[0][1]; reached = 0
        for depth in range(1, self.max_depth+1):
            score = -float('inf'); candidate = best
            try:
                for _, path in moves:
                    child = rules.apply_move(board, path)
                    key = (tuple(child), 3-player)
                    counts[key] += 1
                    try:
                        value = search(child, 3-player, ply+1, depth-1, score, float('inf'))
                    finally:
                        counts[key] -= 1
                    if value > score:
                        score, candidate = value, path
                best, reached = candidate, depth
                moves.sort(key=lambda m: m[1] == best, reverse=True)
            except TimeoutError:
                break
        return best, {'depth_reached': reached, 'nodes': nodes, 'engine': self.name}
