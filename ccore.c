/* ccore.c —— 中国跳棋 PVS + 空着裁剪 + TT 搜索引擎（CPython 扩展）
 *
 * 行为严格对标 engine_pvs.py + Chinese_Checker.py + engine.py：
 *   - 走法生成：81 格棋盘，按 pos 升序收集己方子；每子先跳后路
 *     跳链 DFS 用 pathbb 位图禁回跳，seenend 位图边生成边去重
 *   - 评估函数：v0 = min*100+sum；v2 = v0 - back*3 - opp*0.05；v8 = v0 - back*3 - opp*1.0
 *     全部 double 保证与 Python float64 一致
 *   - 搜索：PVS 主变窗 + 空着裁剪(R=2) + TT(数组hash比对) + killer[ply][2]
 *     + history[3][81*81] (depth^2) + 迭代加深 + CLOCK_MONOTONIC 超时
 *   - 排序：复现 order_moves (prio desc, s desc) 且稳定 → 插入排序严格 >
 *   - Zobrist：C 端独立生成(xorshift fixed seed)，与 Python 版互不依赖
 *
 * 编译：clang -O3 -fPIC -shared -I<py3.11 headers> ccore.c -o ccore.cpython-311-darwin.so
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#include "ccore_tables.h"

/* =========================================================
 * 位图（81 位，pos<64 在 lo，pos>=64 在 hi）
 * ========================================================= */
typedef struct { uint64_t lo, hi; } BB;

static inline int bb_test(BB b, int p) {
    if (p < 64) return (int)((b.lo >> p) & 1);
    return (int)((b.hi >> (p - 64)) & 1);
}
static inline BB bb_set(BB b, int p) {
    if (p < 64) b.lo |= (1ULL << p);
    else         b.hi |= (1ULL << (p - 64));
    return b;
}
static inline BB bb_clr(BB b, int p) {
    if (p < 64) b.lo &= ~(1ULL << p);
    else         b.hi &= ~(1ULL << (p - 64));
    return b;
}
static inline int bb_empty(BB b) { return b.lo == 0 && b.hi == 0; }
static inline BB bb_or(BB a, BB b) { return (BB){a.lo | b.lo, a.hi | b.hi}; }
static inline BB bb_and(BB a, BB b) { return (BB){a.lo & b.lo, a.hi & b.hi}; }
static inline BB bb_not(BB a) { return (BB){~a.lo, ~a.hi}; }

/* 遍历所有置位：for (int p = bb_next(b, -1); p >= 0; p = bb_next(b, p)) */
static inline int bb_next(BB b, int prev) {
    /* 找 prev 之后的下一个置位 */
    for (int p = prev + 1; p < 81; p++) {
        if (bb_test(b, p)) return p;
    }
    return -1;
}

/* =========================================================
 * Board / Move
 * ========================================================= */
typedef struct {
    BB p[3];  /* p[1], p[2] —— 双方棋子位图 */
} Board;

static inline BB board_occ(Board b) { return bb_or(b.p[1], b.p[2]); }

static Board board_from_list(const int *arr) {
    Board b = {{0}};
    for (int p = 0; p < 81; p++) {
        if (arr[p] == 1) b.p[1] = bb_set(b.p[1], p);
        else if (arr[p] == 2) b.p[2] = bb_set(b.p[2], p);
    }
    return b;
}

/* 胜负判定：P1 在 pos>=71 满 10 子胜；P2 在 pos<10 满 10 子胜（先判 P1） */
static int is_game_over(Board b) {
    int p1_in = 0, p2_in = 0;
    BB p1_tgt = b.p[1];
    BB p2_tgt = b.p[2];
    for (int p = 71; p < 81; p++) if (bb_test(p1_tgt, p)) p1_in++;
    for (int p = 0; p < 10; p++)  if (bb_test(p2_tgt, p)) p2_in++;
    if (p1_in == 10) return 1;
    if (p2_in == 10) return 2;
    return 0;
}

/* =========================================================
 * 走法生成
 *   - 按 pos 升序收集己方子
 *   - 每子先跳后路（get_all_legal_moves = jump_paths + step_paths）
 *   - 跳链 DFS：用 pathbb 禁回跳（等价 Python 的 dst in path）
 *     用 seenend 位图边生成边去重（保留 DFS 前序首次出现 → 等价 Python 末尾 seen_end）
 *   - 被跳过的棋子不被吃掉（occ_fixed 不含中间跳点）
 * ========================================================= */
#define MAX_MOVES 2048
typedef struct {
    int start;   /* 起点 */
    int end;     /* 终点 */
    int plen;    /* 路径节点数（>=3 表示连跳） */
} Move;

static inline int bb_popcount_safe(BB b) {
    return (int)__builtin_popcountll(b.lo) + (int)__builtin_popcountll(b.hi);
}

/* 跳链 DFS —— 生成单个己方子的所有跳路 */
static void gen_jump_paths(
    int cur, int start, BB occ_fixed, BB pathbb, BB *seenend,
    Move *out, int *out_cnt
) {
    /* 当前 cur 上的棋子已被"拿走"（occ_fixed 不含 cur），pathbb 含 cur */
    int cnt = NEIGHBOR_CNT[cur];
    for (int k = 0; k < cnt; k++) {
        int mid = NEIGHBOR[cur][k];
        int dst = JUMP_DST[cur][k];
        if (dst < 0) continue;
        /* mid 必须有子（敌我皆可）—— 用全棋盘占用判定 */
        /* 注意：occ_fixed = (p1|p2) & ~bit(start)，mid 若等于 start 也可能（但 start 已被从 occ_fixed 中移除，
           实际 mid=start 时 board[mid]==start 的棋子已被拿走，不应该跳自己）—— 此处用原始棋盘占用 mid */
        /* 修正：mid 必须是"有子"—— 用完整棋盘占用（从调用方传） */
        /* 我们在 occ_fixed 中不含 start，但含 mid（mid 是 start 以外的己/敌子） */
        if (!bb_test(occ_fixed, mid)) continue;  /* mid 必须有子 */
        /* dst 必须空（不在 occ_fixed 也不在 pathbb） */
        if (bb_test(occ_fixed, dst)) continue;
        if (bb_test(pathbb, dst)) continue;
        /* 禁回跳 —— dst 不能等于 start（pathbb 不含 start，但 seenend 也不允许 dst==start） */
        if (dst == start) continue;

        /* 去重：seenend 没见过这个终点才记录 */
        if (!bb_test(*seenend, dst)) {
            *seenend = bb_set(*seenend, dst);
            if (*out_cnt < MAX_MOVES) {
                out[*out_cnt].start = start;
                out[*out_cnt].end = dst;
                /* path = [start, ...visited..., cur, dst]
                 * pathbb 含 start 与已访问点(含 cur) → 节点数 = popcount(pathbb) + 1
                 * （Python 侧 plen 对应 len(path)，>=3 才算法连跳奖励） */
                out[*out_cnt].plen = (int)bb_popcount_safe(pathbb) + 1;
                (*out_cnt)++;
            }
        }

        /* 递归连跳：cur 挪到 dst */
        BB new_pathbb = bb_set(pathbb, dst);
        /* occ_fixed 需要临时清掉 cur（cur 上的棋子跳到 dst 去了）—— 但原 occ_fixed 不含 start
           我们传入的 occ_fixed 是初始的 (p1|p2) & ~bit(start)，cur 变化时需要临时清 cur 的占用 */
        /* 优化：occ_fixed 不动，在递归时用局部 mask 把 cur 清掉（因为 cur 在移动） */
        BB cur_mask = bb_clr(occ_fixed, cur);
        gen_jump_paths(dst, start, cur_mask, new_pathbb, seenend, out, out_cnt);
    }
}

/* 生成所有合法走法 */
static int gen_all_moves(Board board, int player, Move *out) {
    int cnt = 0;
    BB occ_all = board_occ(board);
    BB mine = board.p[player];

    /* 按 pos 升序收集己方子 */
    for (int start = 0; start < 81; start++) {
        if (!bb_test(mine, start)) continue;

        BB occ_fixed = bb_clr(occ_all, start);  /* 起点的棋即将被移走 */
        BB pathbb = bb_set((BB){0}, start);       /* 禁回跳起点 */
        BB seenend = (BB){0};

        /* 先跳 */
        gen_jump_paths(start, start, occ_fixed, pathbb, &seenend, out, &cnt);

        /* 后路 */
        int ncnt = NEIGHBOR_CNT[start];
        for (int k = 0; k < ncnt; k++) {
            int dst = NEIGHBOR[start][k];
            if (bb_test(occ_all, dst)) continue;  /* dst 必须空（完整棋盘） */
            if (cnt < MAX_MOVES) {
                out[cnt].start = start;
                out[cnt].end = dst;
                out[cnt].plen = 2;
                cnt++;
            }
        }
    }
    return cnt;
}

/* =========================================================
 * 评估函数 —— v0 / v2 / v8
 *   _PROG[player][pos]：P1=y, P2=16-y
 *   _BACK[player][pos]：P1=pos<10, P2=pos>=71
 * ========================================================= */
static double prog1[81], prog2[81];
static int back1[81], back2[81];
static int prog_tables_init = 0;

static void init_prog_tables(void) {
    if (prog_tables_init) return;
    for (int p = 0; p < 81; p++) {
        prog1[p] = (double)POS_Y[p];
        prog2[p] = 16.0 - (double)POS_Y[p];
        back1[p] = (p < 10);
        back2[p] = (p >= 71);
    }
    prog_tables_init = 1;
}

static double evaluate(Board board, int player, int version) {
    init_prog_tables();
    int opp = 3 - player;
    double *prog = (player == 1) ? prog1 : prog2;
    double *opp_prog = (opp == 1) ? prog1 : prog2;
    int *back_z = (player == 1) ? back1 : back2;

    double my_min = 17.0, my_sum = 0.0;
    int back = 0;
    double op_min = 17.0, op_sum = 0.0;

    BB my = board.p[player];
    BB op = board.p[opp];

    for (int p = 0; p < 81; p++) {
        if (bb_test(my, p)) {
            double v = prog[p];
            if (v < my_min) my_min = v;
            my_sum += v;
            if (back_z[p]) back++;
        } else if (bb_test(op, p)) {
            double v = opp_prog[p];
            if (v < op_min) op_min = v;
            op_sum += v;
        }
    }

    double score;
    if (version == 0) {
        /* v0 = min*100 + sum（无 back，无对手项） */
        score = my_min >= 17.0 ? 0.0 : my_min * 100.0 + my_sum;
    } else {
        /* v2 / v8 都有 back 惩罚 + 对手项 */
        score = my_min >= 17.0 ? 0.0 : my_min * 100.0 + my_sum - (double)back * 3.0;
        if (op_min < 17.0) {
            double opp_part = op_min * 100.0 + op_sum;
            double w = (version == 8) ? 1.0 : 0.05;
            score -= opp_part * w;
        }
    }
    return score;
}

/* =========================================================
 * Zobrist 哈希
 * ========================================================= */
static uint64_t _ZOB[81][3];   /* ZOB[pos][pc] —— pc=1/2 */
static uint64_t _ZOB_TURN;
static int zob_init = 0;

static uint64_t xorshift64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *s = x;
    return x;
}

static void init_zob(void) {
    if (zob_init) return;
    uint64_t seed = 0x123456789ABCDEFULL;
    for (int p = 0; p < 81; p++) {
        _ZOB[p][0] = 0;
        _ZOB[p][1] = xorshift64(&seed);
        _ZOB[p][2] = xorshift64(&seed);
    }
    _ZOB_TURN = xorshift64(&seed);
    zob_init = 1;
}

static uint64_t zob_init_hash(Board b, int player) {
    init_zob();
    uint64_t h = 0;
    BB p1 = b.p[1], p2 = b.p[2];
    for (int p = 0; p < 81; p++) {
        if (bb_test(p1, p)) h ^= _ZOB[p][1];
        else if (bb_test(p2, p)) h ^= _ZOB[p][2];
    }
    if (player == 2) h ^= _ZOB_TURN;
    return h;
}

static uint64_t zob_move_hash(uint64_t h, int start, int end, int player) {
    h ^= _ZOB[start][player];
    h ^= _ZOB[end][player];
    h ^= _ZOB_TURN;
    return h;
}

/* =========================================================
 * 置换表（数组 hash，比对完整 key 防误命中）
 * ========================================================= */
#define TT_SIZE (1 << 20)  /* 1M 条目 */
typedef struct {
    uint64_t key;
    double score;
    int32_t  depth;
    int32_t  flag;    /* 0=EXACT, 1=LOWER, 2=UPPER */
    int32_t  best;    /* best move code = start*81 + end */
    int32_t  used;
} TTEntry;

#define _EXACT 0
#define _LOWER 1
#define _UPPER 2

/* =========================================================
 * 时间
 * ========================================================= */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* =========================================================
 * 搜索上下文（每次 think 新建）
 * ========================================================= */
#define MAX_PLY 64
#define _INF  10000000000.0
#define _NEG_INF -10000000000.0
#define WIN_SCORE 10000000.0

typedef struct {
    TTEntry *tt;
    int32_t killer[MAX_PLY][2];
    int64_t history[3][81 * 81];  /* [player][code] */
    double deadline;
    int32_t null_r;
    int64_t nodes;
    int64_t prunes;
    int aborted;
    int eval_version;
} SearchCtx;

/* =========================================================
 * apply_move —— 原地修改棋盘（起点清掉，终点放入）
 * ========================================================= */
static void board_apply(Board *b, int start, int end, int player) {
    b->p[player] = bb_clr(b->p[player], start);
    b->p[player] = bb_set(b->p[player], end);
}

/* =========================================================
 * 走法排序 —— 复现 order_moves 的 (prio desc, s desc) 且稳定
 *   Python sort(reverse=True) 稳定 → C 插入排序严格 > 比较保序
 * ========================================================= */
typedef struct {
    int start, end, plen;
    int prio;
    int64_t s;
} ScoredMove;

static void order_moves_c(Move *moves, int n, int tt_best,
                          int32_t *killer, int player, SearchCtx *ctx) {
    ScoredMove scored[MAX_MOVES];
    for (int i = 0; i < n; i++) {
        int code = moves[i].start * 81 + moves[i].end;
        int prio;
        if (code == tt_best) prio = 30;
        else if (killer != NULL && (code == killer[0] || code == killer[1])) prio = 20;
        else prio = 10;
        int64_t s = ctx->history[player][code];
        if (moves[i].plen >= 3) s += 1000;
        scored[i].start = moves[i].start;
        scored[i].end = moves[i].end;
        scored[i].plen = moves[i].plen;
        scored[i].prio = prio;
        scored[i].s = s;
    }
    /* 插入排序 —— 严格 > 保证稳定（同键保持原序） */
    for (int i = 1; i < n; i++) {
        ScoredMove key = scored[i];
        int j = i - 1;
        while (j >= 0) {
            int cmp = 0;
            if (scored[j].prio > key.prio) cmp = 1;
            else if (scored[j].prio < key.prio) cmp = -1;
            else if (scored[j].s > key.s) cmp = 1;
            else if (scored[j].s < key.s) cmp = -1;
            if (cmp < 0) {
                scored[j + 1] = scored[j];
                j--;
            } else break;
        }
        scored[j + 1] = key;
    }
    for (int i = 0; i < n; i++) {
        moves[i].start = scored[i].start;
        moves[i].end = scored[i].end;
        moves[i].plen = scored[i].plen;
    }
}

/* =========================================================
 * PVS + 空着裁剪 搜索主体
 * ========================================================= */
static double pvs_count(Board *b, int player, int depth, int root_player,
                        double alpha, double beta, int ply,
                        uint64_t h, SearchCtx *ctx, int *e_best_out) {
    ctx->nodes++;

    if (ctx->aborted) return evaluate(*b, root_player, ctx->eval_version);

    /* 置换表命中 */
    uint64_t idx = h & (TT_SIZE - 1);
    TTEntry *entry = &ctx->tt[idx];
    int e_best = -1;
    if (entry->used && entry->key == h) {
        if (entry->depth >= depth) {
            if (entry->flag == _EXACT) return entry->score;
            if (entry->flag == _LOWER && entry->score >= beta) return entry->score;
            if (entry->flag == _UPPER && entry->score <= alpha) return entry->score;
            if (alpha >= beta) return entry->score;
        }
        e_best = entry->best;
    }

    /* 超时检查 */
    if (now_sec() >= ctx->deadline) {
        ctx->aborted = 1;
        return evaluate(*b, root_player, ctx->eval_version);
    }

    *e_best_out = e_best;

    /* 胜负 / 深度终止 */
    int winner = is_game_over(*b);
    if (winner) {
        return (winner == root_player) ? WIN_SCORE : -WIN_SCORE;
    }
    if (depth <= 0) return evaluate(*b, root_player, ctx->eval_version);

    /* ===== 空着裁剪 ===== */
    if (ctx->null_r > 0 && depth - 1 - ctx->null_r > 0) {
        uint64_t h_null = h ^ _ZOB_TURN;
        double v;
        if (player == root_player) {
            v = pvs_count(b, 3 - player, depth - 1 - ctx->null_r, root_player,
                          beta - 1, beta, ply + 1, h_null, ctx, &(int){0});
        } else {
            v = pvs_count(b, 3 - player, depth - 1 - ctx->null_r, root_player,
                          alpha, alpha + 1, ply + 1, h_null, ctx, &(int){0});
        }
        if (ctx->aborted) return evaluate(*b, root_player, ctx->eval_version);
        if (player == root_player && v >= beta) { ctx->prunes++; return v; }
        if (player != root_player && v <= alpha) { ctx->prunes++; return v; }
    }

    /* 走法生成 */
    Move moves[MAX_MOVES];
    int nmoves = gen_all_moves(*b, player, moves);
    if (nmoves == 0) return evaluate(*b, root_player, ctx->eval_version);

    /* 排序 */
    if (ply < MAX_PLY) {
        order_moves_c(moves, nmoves, e_best, ctx->killer[ply], player, ctx);
    } else {
        order_moves_c(moves, nmoves, e_best, NULL, player, ctx);
    }

    double alpha_orig = alpha;
    double beta_orig = beta;
    int best_move = -1;

    if (player == root_player) {
        double value = _NEG_INF;
        for (int i = 0; i < nmoves; i++) {
            if (ctx->aborted) break;
            int s = moves[i].start, e = moves[i].end;
            board_apply(b, s, e, player);
            uint64_t nh = zob_move_hash(h, s, e, player);
            double v;
            if (i == 0) {
                v = pvs_count(b, 3 - player, depth - 1, root_player,
                              alpha, beta, ply + 1, nh, ctx, &(int){0});
            } else {
                v = pvs_count(b, 3 - player, depth - 1, root_player,
                              alpha, alpha + 1, ply + 1, nh, ctx, &(int){0});
                if (!ctx->aborted && alpha < v && v < beta) {
                    v = pvs_count(b, 3 - player, depth - 1, root_player,
                                  alpha, beta, ply + 1, nh, ctx, &(int){0});
                }
            }
            board_apply(b, e, s, player);  /* 恢复 */

            if (v > value) {
                value = v;
                best_move = s * 81 + e;
                if (value > alpha) alpha = value;
            }
            if (alpha >= beta) {
                ctx->prunes++;
                /* record_cut */
                int code = s * 81 + e;
                if (ply < MAX_PLY) {
                    int32_t *k = ctx->killer[ply];
                    if (k[0] != code) { k[1] = k[0]; k[0] = code; }
                }
                ctx->history[player][code] += (int64_t)depth * (int64_t)depth;
                break;
            }
        }
        if (ctx->aborted) return evaluate(*b, root_player, ctx->eval_version);
        if (value <= alpha_orig) entry->flag = _UPPER;
        else if (value >= beta_orig) entry->flag = _LOWER;
        else entry->flag = _EXACT;
        entry->score = value;
        entry->depth = depth;
        entry->best = best_move;
        entry->key = h;
        entry->used = 1;
        return value;
    } else {
        /* 最小节点（镜像） */
        double value = _INF;
        for (int i = 0; i < nmoves; i++) {
            if (ctx->aborted) break;
            int s = moves[i].start, e = moves[i].end;
            board_apply(b, s, e, player);
            uint64_t nh = zob_move_hash(h, s, e, player);
            double v;
            if (i == 0) {
                v = pvs_count(b, 3 - player, depth - 1, root_player,
                              alpha, beta, ply + 1, nh, ctx, &(int){0});
            } else {
                v = pvs_count(b, 3 - player, depth - 1, root_player,
                              beta - 1, beta, ply + 1, nh, ctx, &(int){0});
                if (!ctx->aborted && alpha < v && v < beta) {
                    v = pvs_count(b, 3 - player, depth - 1, root_player,
                                  alpha, beta, ply + 1, nh, ctx, &(int){0});
                }
            }
            board_apply(b, e, s, player);  /* 恢复 */

            if (v < value) {
                value = v;
                best_move = s * 81 + e;
                if (value < beta) beta = value;
            }
            if (alpha >= beta) {
                ctx->prunes++;
                int code = s * 81 + e;
                if (ply < MAX_PLY) {
                    int32_t *k = ctx->killer[ply];
                    if (k[0] != code) { k[1] = k[0]; k[0] = code; }
                }
                ctx->history[player][code] += (int64_t)depth * (int64_t)depth;
                break;
            }
        }
        if (ctx->aborted) return evaluate(*b, root_player, ctx->eval_version);
        if (value <= alpha_orig) entry->flag = _UPPER;
        else if (value >= beta_orig) entry->flag = _LOWER;
        else entry->flag = _EXACT;
        entry->score = value;
        entry->depth = depth;
        entry->best = best_move;
        entry->key = h;
        entry->used = 1;
        return value;
    }
}

/* =========================================================
 * think() —— 迭代加深 + 限时
 *   返回 (start, end, score, depth, nodes, prunes, aborted)
 * ========================================================= */
typedef struct {
    int start, end;
    double score;
    int depth_reached;
    int64_t nodes, prunes;
    int aborted;
} ThinkResult;

static ThinkResult ccore_think(const int *board_arr, int player,
                                double time_limit, int max_depth,
                                int eval_version, int null_r) {
    ThinkResult res = {0};
    Board board = board_from_list(board_arr);

    /* 根节点走法 —— 生成一次，所有迭代共享 */
    Move root_moves[MAX_MOVES];
    int nroot = gen_all_moves(board, player, root_moves);
    if (nroot == 0) { return res; }

    /* 初始化搜索上下文 */
    TTEntry *tt = (TTEntry *)calloc(TT_SIZE, sizeof(TTEntry));
    if (!tt) { res.aborted = 1; return res; }

    SearchCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.tt = tt;
    memset(ctx.killer, -1, sizeof(ctx.killer));
    ctx.deadline = now_sec() + time_limit;
    ctx.null_r = null_r;
    ctx.eval_version = eval_version;

    uint64_t root_hash = zob_init_hash(board, player);

    int best_start = root_moves[0].start;
    int best_end = root_moves[0].end;
    double best_score = _NEG_INF;
    int depth_reached = 0;
    int64_t tot_nodes = 0, tot_prunes = 0;
    int aborted_overall = 0;

    for (int depth = 1; depth <= max_depth; depth++) {
        double alpha = _NEG_INF;
        double beta = _INF;
        double best_score_d = _NEG_INF;
        int best_start_d = root_moves[0].start;
        int best_end_d = root_moves[0].end;

        /* 从 TT 取 root_best */
        int root_best = -1;
        uint64_t idx = root_hash & (TT_SIZE - 1);
        if (tt[idx].used && tt[idx].key == root_hash) root_best = tt[idx].best;

        /* Python 每轮都从原始生成顺序稳定排序。 */
        gen_all_moves(board, player, root_moves);
        order_moves_c(root_moves, nroot, root_best, ctx.killer[0], player, &ctx);

        int64_t nodes_before = ctx.nodes;
        int64_t prunes_before = ctx.prunes;

        for (int i = 0; i < nroot; i++) {
            if (now_sec() >= ctx.deadline) { ctx.aborted = 1; break; }

            int s = root_moves[i].start, e = root_moves[i].end;
            board_apply(&board, s, e, player);
            uint64_t nh = zob_move_hash(root_hash, s, e, player);
            double score;
            int e_best_dummy;

            if (i == 0) {
                score = pvs_count(&board, 3 - player, depth - 1,
                                  player, alpha, beta, 1, nh, &ctx, &e_best_dummy);
            } else {
                score = pvs_count(&board, 3 - player, depth - 1,
                                  player, alpha, alpha + 1, 1, nh, &ctx, &e_best_dummy);
                if (!ctx.aborted && alpha < score && score < beta) {
                    score = pvs_count(&board, 3 - player, depth - 1,
                                      player, alpha, beta, 1, nh, &ctx, &e_best_dummy);
                }
            }
            board_apply(&board, e, s, player);

            if (!ctx.aborted && score > best_score_d) {
                best_score_d = score;
                best_start_d = s;
                best_end_d = e;
                if (best_score_d > alpha) alpha = best_score_d;
            }
            if (ctx.aborted) break;
        }

        int64_t depth_nodes = ctx.nodes - nodes_before;
        int64_t depth_prunes = ctx.prunes - prunes_before;
        tot_nodes += depth_nodes;
        tot_prunes += depth_prunes;

        if (ctx.aborted) {
            aborted_overall = 1;
            break;
        } else {
            best_start = best_start_d;
            best_end = best_end_d;
            best_score = best_score_d;
            depth_reached = depth;
            if (best_score >= WIN_SCORE - 1 || best_score <= -WIN_SCORE + 1) break;
        }
    }

    free(tt);

    res.start = best_start;
    res.end = best_end;
    res.score = best_score;
    res.depth_reached = depth_reached;
    res.nodes = tot_nodes;
    res.prunes = tot_prunes;
    res.aborted = aborted_overall;
    return res;
}

/* =========================================================
 * Python 绑定
 *   cc_search(board_list, player, time_limit, max_depth, eval_version, null_r)
 *     -> (start, end, score, depth, nodes, prunes, aborted)
 * ========================================================= */
static PyObject *py_cc_search(PyObject *self, PyObject *args) {
    PyObject *board_list;
    int player, max_depth, eval_version, null_r;
    double time_limit;

    if (!PyArg_ParseTuple(args, "Oidiii",
                          &board_list, &player, &time_limit,
                          &max_depth, &eval_version, &null_r)) {
        return NULL;
    }

    int board[81];
    if (PyList_Check(board_list)) {
        if (PyList_Size(board_list) != 81) {
            PyErr_SetString(PyExc_ValueError, "board must be length 81");
            return NULL;
        }
        for (int i = 0; i < 81; i++) {
            PyObject *v = PyList_GetItem(board_list, i);
            board[i] = (int)PyLong_AsLong(v);
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "board must be list");
        return NULL;
    }

    /* 搜索只使用本地 C 数据；释放 GIL，让 GUI 主线程继续处理事件。 */
    init_zob();
    init_prog_tables();
    ThinkResult r;
    Py_BEGIN_ALLOW_THREADS
    r = ccore_think(board, player, time_limit, max_depth, eval_version, null_r);
    Py_END_ALLOW_THREADS

    if (r.nodes == 0) {
        /* 无子可走 —— 返回 None 路径 */
        return PyTuple_Pack(7, Py_None, Py_None,
                            PyFloat_FromDouble(0.0),
                            PyLong_FromLong(0),
                            PyLong_FromLong(0),
                            PyLong_FromLong(0),
                            PyLong_FromLong(0));
    }

    return PyTuple_Pack(7,
        PyLong_FromLong(r.start),
        PyLong_FromLong(r.end),
        PyFloat_FromDouble(r.score),
        PyLong_FromLong(r.depth_reached),
        PyLong_FromLong((long)r.nodes),
        PyLong_FromLong((long)r.prunes),
        PyLong_FromLong(r.aborted ? 1 : 0)
    );
}

/* 暴露独立工具函数供验证 */
static PyObject *py_cc_gen_moves(PyObject *self, PyObject *args) {
    PyObject *board_list;
    int player;
    if (!PyArg_ParseTuple(args, "Oi", &board_list, &player)) return NULL;

    int board[81];
    for (int i = 0; i < 81; i++) {
        PyObject *v = PyList_GetItem(board_list, i);
        board[i] = (int)PyLong_AsLong(v);
    }
    Board b = board_from_list(board);
    Move moves[MAX_MOVES];
    int n = gen_all_moves(b, player, moves);

    PyObject *list = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyList_SetItem(list, i, PyTuple_Pack(2,
            PyLong_FromLong(moves[i].start),
            PyLong_FromLong(moves[i].end)));
    }
    return list;
}

static PyObject *py_cc_eval(PyObject *self, PyObject *args) {
    PyObject *board_list;
    int player, version;
    if (!PyArg_ParseTuple(args, "Oii", &board_list, &player, &version)) return NULL;
    int board[81];
    for (int i = 0; i < 81; i++) {
        PyObject *v = PyList_GetItem(board_list, i);
        board[i] = (int)PyLong_AsLong(v);
    }
    Board b = board_from_list(board);
    init_prog_tables();
    double s = evaluate(b, player, version);
    return PyFloat_FromDouble(s);
}

static PyMethodDef ccore_methods[] = {
    {"cc_search",  py_cc_search,  METH_VARARGS, "PVS+null_move+TT search"},
    {"cc_gen_moves", py_cc_gen_moves, METH_VARARGS, "Generate all legal moves"},
    {"cc_eval",    py_cc_eval,    METH_VARARGS, "Evaluate position"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ccoremodule = {
    PyModuleDef_HEAD_INIT, "ccore", NULL, -1, ccore_methods
};

PyMODINIT_FUNC PyInit_ccore(void) {
    return PyModule_Create(&ccoremodule);
}
