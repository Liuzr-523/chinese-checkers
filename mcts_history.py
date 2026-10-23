"""Search-only cycle avoidance; this does not change legal moves or game rules."""
def position_key(board, player):
    return tuple(board), player


def candidate_indices(children, seen, winner):
    # A real win takes precedence over the repetition heuristic.
    winning = [i for i, (_, _, _, child) in enumerate(children)
               if winner(child.board) == 3 - child.player]
    if winning:
        return winning
    fresh = [i for i, (_, _, _, child) in enumerate(children)
             if position_key(child.board, child.player) not in seen]
    # Never manufacture a no-legal-move loss when every child repeats.
    return fresh or list(range(len(children)))
