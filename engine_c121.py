"""Strict 121-cell native PVS; independent of the 81-cell extension and book."""
import math
import rules121 as rules

class CSearch121:
    name = 'C-PVS 121格'
    def __init__(self, seconds=0.5, max_depth=12, max_ply=240):
        if not math.isfinite(seconds) or seconds <= 0 or not 1 <= max_depth <= 62 or max_ply < 0:
            raise ValueError('invalid search limits')
        try:
            import ccore121
        except ImportError as exc:
            raise ImportError('请用启动GUI的Python执行 build_ccore121.py 编译121格引擎') from exc
        self.native = ccore121
        self.seconds, self.max_depth, self.max_ply = seconds, max_depth, max_ply

    def think(self, board, player, ply=0, repetitions=None):
        rules.validate_board(board)
        records = [(list(b), p, count) for (b, p), count in (repetitions or {}).items() if count > 0]
        s, e, score, depth, nodes, prunes, aborted = self.native.search(
            board, player, self.seconds, self.max_depth, ply, records, self.max_ply)
        info = dict(engine=self.name, score=score, depth_reached=depth,
                    nodes_searched=nodes, nodes=nodes, nodes_pruned=prunes,
                    aborted=bool(aborted), eval_version='tip-distance-121-v1',
                    rule_profile=rules.PROFILE)
        if s < 0:
            return None, info
        # Return a checked witness path for the existing slow hop animation.
        path = next((path for path in rules.legal_paths(board, s) if path[-1] == e), None)
        if path is None:
            raise RuntimeError('121格C内核返回非法走法')
        return path, info
