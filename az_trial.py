"""Bounded, reproducible self-play -> training -> paired evaluation experiment.
Only fresh, rule-terminal games train value; champions are never overwritten.
"""
import argparse, copy, hashlib, json, pickle, time
from pathlib import Path
import numpy as np
import torch
import Chinese_Checker as cc
from alphazero_nn import expand_game_records
from checkpoint_paths import ROOT, artifact_path
from nn_resnet import NetResNet, train
from self_play_resnet_mcts import mcts_think_resnet


def game(models, seed, simulations, max_moves=240, explore=False, opening=0):
    rng = np.random.default_rng(seed)
    board = cc.init_board()
    steps, seen = [], {}
    reason = 'move_limit'
    for ply in range(max_moves):
        player = 1 + ply % 2
        key = (tuple(board), player)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 3:
            reason = 'repetition'
            break
        if ply < opening:
            moves = [(s, p[-1], p) for s, p in cc.get_all_moves(board, player)]
            pi = np.full(len(moves), 1 / len(moves))
        else:
            (moves, pi), _ = mcts_think_resnet(
                models[player - 1], board, player, simulations,
                add_root_noise=explore, rng=rng, history=seen)
        if not moves:
            reason = 'no_moves'
            break
        probs = np.asarray(pi, dtype=np.float64)
        probs /= probs.sum()
        i = int(rng.choice(len(moves), p=probs)) if (explore and ply < 20) or ply < opening else int(np.argmax(probs))
        path = moves[i][2]
        steps.append(dict(player=player, move_path=list(path), moves=moves, pi=probs.tolist()))
        board = cc.apply_move(board, path)
        if cc.is_game_over(board) is not None:
            reason = 'terminal'
            break
    winner = cc.is_game_over(board)
    return dict(winner=winner, steps=steps, aborted=winner is None, reason=reason, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--games', type=int, default=6)
    ap.add_argument('--simulations', type=int, default=32)
    ap.add_argument('--steps', type=int, default=120)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(20261022)
    checkpoint = artifact_path(ROOT / 'params_resnet_champ.pt')
    baseline = NetResNet.load(checkpoint).eval()
    baseline.save(args.out / 'baseline.pt')
    config = dict(seed=20261022, games=args.games, simulations=args.simulations,
                  train_steps=args.steps, lr=2e-5, batch_size=16, max_moves=240,
                  baseline=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  source='existing champion, fresh self-play only; no opening book', eval_pairs=2)
    (args.out / 'config.json').write_text(json.dumps(config, indent=2))
    started = time.monotonic()
    records = []
    for i in range(args.games):
        rec = game([baseline, baseline], 20261022 + i, args.simulations, explore=True)
        records.append(rec)
        with (args.out / 'selfplay.pkl').open('wb') as f:
            pickle.dump(records, f)
        print(f"selfplay {i+1}/{args.games}: {rec['reason']} winner={rec['winner']} plies={len(rec['steps'])}", flush=True)
    complete = [r for r in records if r['winner'] in (1, 2)]
    samples = expand_game_records(complete)
    result = dict(completed=len(complete), unfinished=len(records)-len(complete), samples=len(samples))
    if samples:
        candidate = copy.deepcopy(baseline)
        result['last_train_loss'] = train(samples, candidate, steps=args.steps, lr=2e-5,
                                          bs=16, seed=20261022, log_every=30)
        candidate.eval().save(args.out / 'candidate.pt')
        matches = []
        wins = losses = unfinished = 0
        for pair in range(2):
            for side in (1, 2):
                models = [candidate, baseline] if side == 1 else [baseline, candidate]
                rec = game(models, 9100+pair, args.simulations, opening=4)
                rec['candidate_side'] = side
                matches.append(rec)
                wins += rec['winner'] == side
                losses += rec['winner'] == 3-side
                unfinished += rec['winner'] is None
                print(f"eval {len(matches)}/4: candidate=P{side}, winner={rec['winner']}, {rec['reason']}, plies={len(rec['steps'])}", flush=True)
                with (args.out / 'evaluation.pkl').open('wb') as f:
                    pickle.dump(matches, f)
        result.update(wins=wins, losses=losses, unfinished_eval=unfinished)
    else:
        result['blocked'] = 'No completed games: no trustworthy value labels, training skipped.'
    result.update(seconds=time.monotonic()-started, promoted=False)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
