"""纯 C PVS 搜索，沿用 Python 引擎 think 接口。"""
import random
import warnings
import ccore
from engine import Engine, cc


class CESEngine(Engine):
    def __init__(self, time_limit_sec=1.5, max_depth=20,
                 eval_version="v2", null_move_r=2, use_book=True, book_seed=None):
        versions = {"v0": 0, "v2": 2, "v8": 8, "current": 2}
        if eval_version not in versions:
            raise ValueError("C engine supports v0, v2, v8, current")
        if not 1 <= max_depth < 64:
            raise ValueError("max_depth must be between 1 and 63")
        self.use_book = use_book
        self._book_rng = random.Random(book_seed)
        self._book_available = True
        self.time_limit = time_limit_sec
        self.max_depth = max_depth
        self.eval_version = eval_version
        self.null_move_r = null_move_r
        self._ver_id = versions[eval_version]
        self.name = f"C-PVS-t{time_limit_sec}s-{eval_version}"

    def think(self, board, player):
        if self.use_book and self._book_available:
            try:
                from clysto_book import book_move
                move = book_move(board, player, cc.get_all_moves, self._book_rng)
            except (OSError, ValueError, AssertionError, AttributeError) as exc:
                self._book_available = False
                warnings.warn(f"开局库不可用，继续 C 搜索: {exc}", RuntimeWarning)
                move = None
            if move is not None:
                return list(move), {
                    "score": ccore.cc_eval(board, player, self._ver_id),
                    "depth_reached": 0, "nodes_searched": 0, "nodes_pruned": 0,
                    "prune_rate": "0%", "aborted": False,
                    "eval_version": self.eval_version, "book_hit": True,
                    "score_source": "static_eval", "engine": "ccore+book"}
        s, e, score, depth, nodes, prunes, aborted = ccore.cc_search(
            board, player, self.time_limit, self.max_depth,
            self._ver_id, self.null_move_r)
        info = {"book_hit": False, "score": score, "depth_reached": depth,
                "nodes_searched": nodes, "nodes_pruned": prunes,
                "prune_rate": f"{prunes / max(nodes, 1) * 100:.0f}%",
                "aborted": bool(aborted), "eval_version": self.eval_version}
        if s is None:
            info["reason"] = "no_moves"
            return None, info
        return [s, e], info
