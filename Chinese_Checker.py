NEIGHBORS_81 = {
    0: [1, 2],
    1: [0, 2, 3, 4],
    2: [0, 1, 4, 5],
    3: [1, 4, 6, 7],
    4: [1, 2, 3, 5, 7, 8],
    5: [2, 4, 8, 9],
    6: [3, 7, 10, 11],
    7: [3, 4, 6, 8, 11, 12],
    8: [4, 5, 7, 9, 12, 13],
    9: [5, 8, 13, 14],
    10: [6, 11, 15, 16],
    11: [6, 7, 10, 12, 16, 17],
    12: [7, 8, 11, 13, 17, 18],
    13: [8, 9, 12, 14, 18, 19],
    14: [9, 13, 19, 20],
    15: [10, 16, 21, 22],
    16: [10, 11, 15, 17, 22, 23],
    17: [11, 12, 16, 18, 23, 24],
    18: [12, 13, 17, 19, 24, 25],
    19: [13, 14, 18, 20, 25, 26],
    20: [14, 19, 26, 27],
    21: [15, 22, 28, 29],
    22: [15, 16, 21, 23, 29, 30],
    23: [16, 17, 22, 24, 30, 31],
    24: [17, 18, 23, 25, 31, 32],
    25: [18, 19, 24, 26, 32, 33],
    26: [19, 20, 25, 27, 33, 34],
    27: [20, 26, 34, 35],
    28: [21, 29, 36, 37],
    29: [21, 22, 28, 30, 37, 38],
    30: [22, 23, 29, 31, 38, 39],
    31: [23, 24, 30, 32, 39, 40],
    32: [24, 25, 31, 33, 40, 41],
    33: [25, 26, 32, 34, 41, 42],
    34: [26, 27, 33, 35, 42, 43],
    35: [27, 34, 43, 44],
    36: [28, 37, 45],
    37: [28, 29, 36, 38, 45, 46],
    38: [29, 30, 37, 39, 46, 47],
    39: [30, 31, 38, 40, 47, 48],
    40: [31, 32, 39, 41, 48, 49],
    41: [32, 33, 40, 42, 49, 50],
    42: [33, 34, 41, 43, 50, 51],
    43: [34, 35, 42, 44, 51, 52],
    44: [35, 43, 52],
    45: [36, 37, 46, 53],
    46: [37, 38, 45, 47, 53, 54],
    47: [38, 39, 46, 48, 54, 55],
    48: [39, 40, 47, 49, 55, 56],
    49: [40, 41, 48, 50, 56, 57],
    50: [41, 42, 49, 51, 57, 58],
    51: [42, 43, 50, 52, 58, 59],
    52: [43, 44, 51, 59],
    53: [45, 46, 54, 60],
    54: [46, 47, 53, 55, 60, 61],
    55: [47, 48, 54, 56, 61, 62],
    56: [48, 49, 55, 57, 62, 63],
    57: [49, 50, 56, 58, 63, 64],
    58: [50, 51, 57, 59, 64, 65],
    59: [51, 52, 58, 65],
    60: [53, 54, 61, 66],
    61: [54, 55, 60, 62, 66, 67],
    62: [55, 56, 61, 63, 67, 68],
    63: [56, 57, 62, 64, 68, 69],
    64: [57, 58, 63, 65, 69, 70],
    65: [58, 59, 64, 70],
    66: [60, 61, 67, 71],
    67: [61, 62, 66, 68, 71, 72],
    68: [62, 63, 67, 69, 72, 73],
    69: [63, 64, 68, 70, 73, 74],
    70: [64, 65, 69, 74],
    71: [66, 67, 72, 75],
    72: [67, 68, 71, 73, 75, 76],
    73: [68, 69, 72, 74, 76, 77],
    74: [69, 70, 73, 77],
    75: [71, 72, 76, 78],
    76: [72, 73, 75, 77, 78, 79],
    77: [73, 74, 76, 79],
    78: [75, 76, 79, 80],
    79: [76, 77, 78, 80],
    80: [78, 79]
}


def get_step_moves(pos, board):
    valid_dest = []
    neighbors = NEIGHBORS_81[pos]
    for n in neighbors:
        if board[n] == 0:
            valid_dest.append(n)
    return valid_dest


POS_TO_XY = [
    (8,0),
    (7,1),(8,1),
    (6,2),(7,2),(8,2),
    (5,3),(6,3),(7,3),(8,3),
    (4,4),(5,4),(6,4),(7,4),(8,4),
    (3,5),(4,5),(5,5),(6,5),(7,5),(8,5),
    (2,6),(3,6),(4,6),(5,6),(6,6),(7,6),(8,6),
    (1,7),(2,7),(3,7),(4,7),(5,7),(6,7),(7,7),(8,7),
    (0,8),(1,8),(2,8),(3,8),(4,8),(5,8),(6,8),(7,8),(8,8),
    (0,9),(1,9),(2,9),(3,9),(4,9),(5,9),(6,9),(7,9),
    (0,10),(1,10),(2,10),(3,10),(4,10),(5,10),(6,10),
    (0,11),(1,11),(2,11),(3,11),(4,11),(5,11),
    (0,12),(1,12),(2,12),(3,12),(4,12),
    (0,13),(1,13),(2,13),(3,13),
    (0,14),(1,14),(2,14),
    (0,15),(1,15),
    (0,16),
]

XY_TO_POS = { xy: idx for idx, xy in enumerate(POS_TO_XY) }


def pos_to_xy(pos):
    return POS_TO_XY[pos]


def xy_to_pos(x,y):
    return XY_TO_POS.get((x,y), None)


def get_jump_moves(pos, board):
    jump_dests = []
    x1, y1 = pos_to_xy(pos)
    for mid in NEIGHBORS_81[pos]:
        if board[mid] == 0:
            continue
        x2, y2 = pos_to_xy(mid)
        dx = x2 - x1
        dy = y2 - y1
        x_dst = x2 + dx
        y_dst = y2 + dy

        dst = xy_to_pos(x_dst, y_dst)
        if dst is None:
            continue
        if 0 <= dst <= 80 and board[dst] == 0:
            jump_dests.append(dst)
    return jump_dests


def get_all_jump_paths(start_pos, board, player):
    all_paths = []

    def dfs(current_pos, path, cur_board):
        jump_dests = get_jump_moves(current_pos, cur_board)
        for dst in jump_dests:
            # 禁止回跳，否则出现 A→C→A→C 无限递归
            if dst in path:
                continue
            new_board = cur_board.copy()
            new_board[current_pos] = 0
            new_board[dst] = player  # 跳过的棋子不被吃掉

            new_path = path + [dst]
            all_paths.append(new_path)
            dfs(dst, new_path, new_board)

    dfs(start_pos, [start_pos], board.copy())
    # 同一个终点只保留一条路径
    seen_end = {}
    for p in all_paths:
        if p[-1] not in seen_end:
            seen_end[p[-1]] = p
    return list(seen_end.values())


def get_all_legal_moves(pos, board):
    player = board[pos]
    jump_paths = get_all_jump_paths(pos, board, player)
    step_paths = [[pos, dst] for dst in get_step_moves(pos, board)]
    return jump_paths + step_paths


def apply_move(board, path):
    new_board = board.copy()
    start = path[0]
    end = path[-1]
    piece = new_board[start]
    new_board[start] = 0
    new_board[end] = piece
    return new_board


def get_all_moves(board, player):
    all_moves = []
    for pos in range(81):
        if board[pos] == player:
            paths = get_all_legal_moves(pos, board)
            for p in paths:
                all_moves.append((pos, p))
    return all_moves


def is_game_over(board):
    # 快速版：一次 81 扫描，数双方已进入对方目标区的棋子数。
    # 跳棋无吃子（棋子守恒 10v10），某方 10 子全在对方三角 = 全进 = 获胜。
    p1_in = 0   # P1 棋子位于 P2 起始区 pos 71-80
    p2_in = 0   # P2 棋子位于 P1 起始区 pos 0-9
    for pos in range(81):
        pc = board[pos]
        if pc == 1:
            if pos >= 71:
                p1_in += 1
        elif pc == 2:
            if pos < 10:
                p2_in += 1
    if p1_in == 10:
        return 1
    if p2_in == 10:
        return 2
    return None


def init_board():
    board = [0] * 81
    for pos in range(0, 10):
        board[pos] = 1
    for pos in range(71, 81):
        board[pos] = 2
    return board
    