"""检查开局训练/留出是否共享镜像或转方等价局面。"""
import argparse
import json
from pathlib import Path
from opening121 import MIRROR, OpeningBook
from train_opening121 import book_samples


def parse_decision(identity):
    raw, player = identity.split(':')
    if len(raw) != 121 or any(v not in '012' for v in raw) or player not in ('1', '2'):
        raise ValueError('decision_id须为121位棋盘及走棋方1或2')
    return list(map(int, raw)), int(player)


def decision_key(board, player):
    return ''.join(map(str, board)) + ':' + str(player)


def mirror(identity):
    board, player = parse_decision(identity)
    reflected = [0] * 121
    for i, value in enumerate(board):
        reflected[MIRROR[i]] = value
    return decision_key(reflected, player)


def rotate_swap(identity):
    board, player = parse_decision(identity)
    return decision_key([3 - v if v else 0 for v in reversed(board)], 3 - player)


def orbit(identity):
    swapped = rotate_swap(identity)
    return min(identity, mirror(identity), swapped, mirror(swapped))


def audit_split(train_ids, validation_ids):
    """例数按输入记录计数，具体ID保留重复，轨道数按归一键去重。"""
    def groups(identities):
        result = {}
        for identity in identities:
            result.setdefault(orbit(identity), []).append(identity)
        return result
    train, validation = groups(train_ids), groups(validation_ids)
    shared = sorted(train.keys() & validation.keys())
    details = [dict(orbit=k, train=sorted(train[k]), validation=sorted(validation[k])) for k in shared]
    return dict(train_cases=sum(map(len, train.values())),
                validation_cases=sum(map(len, validation.values())),
                overlap_orbits=len(shared),
                overlap_train_cases=sum(len(train[k]) for k in shared),
                overlap_validation_cases=sum(len(validation[k]) for k in shared),
                overlapping_orbits=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='审计JSON输出路径')
    parser.add_argument('--strict', action='store_true', help='存在跨组等价轨道时退出码为1')
    args = parser.parse_args()
    sets, _ = book_samples(OpeningBook())
    result = audit_split([s['decision_id'] for s in sets['train']],
                         [s['decision_id'] for s in sets['validation']])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f"跨组等价轨道{result['overlap_orbits']}个，训练{result['overlap_train_cases']}例，留出{result['overlap_validation_cases']}例")
    return int(args.strict and result['overlap_orbits'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
