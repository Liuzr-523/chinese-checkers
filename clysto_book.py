"""clysto 开局库（book.hpp 内嵌 3526 条）的 Python 读取器。

原理：clysto 的 GameState::hash() = XOR ZOBRIST_TABLE[his_pos][color]，
      turn==GREEN 时再异或 0xc503204d9e521ac5。
      颜色索引：EMPTY=0, RED=1, GREEN=2（我们 P1->GREEN, P2->RED）。
开局库条目：每条 16 字节 = uint64 hash + int32 src + int32 dst（他的坐标）。

用法：
    from clysto_book import book_move
    m = book_move(board, player)   # 命中返回 (our_src, our_dst)，否则 None
"""
import os
import re
from pathlib import Path

CLYSTO_DIR = os.environ.get("CLYSTO_DIR", "/Users/liuzirui/Desktop/py/clysto-cc")

_TURN_GREEN_XOR = 0xC503204D9E521AC5

_cache = {}


def _load():
    if _cache:
        return _cache["zob"], _cache["book"], _cache["map_h2o"]
    # 1) Zobrist 表（81×3）
    text = open(os.path.join(CLYSTO_DIR, "src", "constants.hpp")).read()
    m = re.search(r"ZOBRIST_TABLE\[81\]\[3\] = \{(.*?)\n\};", text, re.S)
    rows = re.findall(r"\{0x([0-9a-f]+)ULL, 0x([0-9a-f]+)ULL, 0x([0-9a-f]+)ULL\}", m.group(1))
    assert len(rows) == 81, f"ZOBRIST_TABLE 解析出 {len(rows)} 行"
    zob = [[int(a, 16), int(b, 16), int(c, 16)] for a, b, c in rows]
    # 2) 开局库（hex bytes 数组，每 16 字节一条：hash(8B LE) + src(int32 LE) + dst(int32 LE)）
    btext = open(os.path.join(CLYSTO_DIR, "src", "book.hpp")).read()
    raw = bytes(int(x, 16) for x in re.findall(r"0x([0-9a-f]{2})", btext))
    assert len(raw) % 16 == 0, f"book 字节数 {len(raw)} 不是 16 的倍数"
    book = {}
    for i in range(0, len(raw), 16):
        h = int.from_bytes(raw[i:i + 8], "little")
        src = int.from_bytes(raw[i + 8:i + 12], "little", signed=True)
        dst = int.from_bytes(raw[i + 12:i + 16], "little", signed=True)
        book[h] = (src, dst)
    # 3) 坐标映射（他的 pos -> 我们的 pos）
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from arena_vs_clysto import MAP_H2O
    _cache.update(zob=zob, book=book, map_h2o=MAP_H2O)
    return zob, book, MAP_H2O


def _hash(board, player, map_h2o, zob):
    """我们的 board/player -> clysto 的 zobrist hash。
    我们 P1 -> 他 GREEN(idx2)；我们 P2 -> 他 RED(idx1)。
    轮次：player==1 即他 GREEN 走棋。"""
    # map_h2o[his_p] = our_p，需要 his_pos：遍历我们的 pos 反查
    h = 0
    # 建 his->our 的逆映射做一次
    o2h = [0] * 81
    for hp in range(81):
        o2h[map_h2o[hp]] = hp
    for op in range(81):
        v = board[op]
        if v == 0:
            continue
        hp = o2h[op]
        h ^= zob[hp][2 if v == 1 else 1]
    if player == 1:  # 他 GREEN 走
        h ^= _TURN_GREEN_XOR
    return h


def book_move(board, player, get_all_moves=None):
    """命中开局库返回 (our_src, our_dst)，否则 None。

    book 条目的 hash 轮次位约定不统一（实测部分条目总是带 GREEN 轮次位），
    因此两种 hash 都查，返回走法属于当前玩家的那个命中。
    传入 get_all_moves 时可顺便校验走法合法性（连跳路径由调用方匹配）。"""
    zob, book, map_h2o = _load()
    h = _hash(board, player, map_h2o, zob)
    for cand in (h, h ^ _TURN_GREEN_XOR):
        hit = book.get(cand)
        if hit is None:
            continue
        src, dst = map_h2o[hit[0]], map_h2o[hit[1]]
        if board[src] != player:
            continue
        if get_all_moves is not None:
            ok = any(s == src and p[-1] == dst for s, p in get_all_moves(board, player))
            if not ok:
                continue
        return src, dst
    return None


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import engine
    cc = engine.cc
    board = cc.init_board()
    # 双方沿开局库走，统计连续命中
    hits = 0
    for step in range(30):
        player = 1 if step % 2 == 0 else 2
        m = book_move(board, player, cc.get_all_moves)
        if m is None:
            print(f"step {step} player{player}: 未命中，停止")
            break
        hits += 1
        path = next(p for s, p in cc.get_all_moves(board, player)
                    if s == m[0] and p[-1] == m[1])
        board = cc.apply_move(board, path)
    print(f"连续命中 {hits} 步（初始局面起）")
