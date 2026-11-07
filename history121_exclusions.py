"""从历史JSON/JSONL与安全数值PT建立121格对称排除清单；不读取.pkl。"""
import argparse
import hashlib
import json
import math
import re
import numpy as np
import torch
from collections import Counter
from pathlib import Path
import rules121 as r
from opening121_split_audit import orbit, decision_key

IDENTITY = re.compile(r'^[012]{121}:[12]$')


def safe_tensor_load(path):
    """只允许PyTorch数值对象及有限NumPy数值构造器；始终weights_only。"""
    numeric = [np.bool_, np.uint8, np.int8, np.int16, np.int32, np.int64,
               np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64]
    reconstruct = np._core.multiarray._reconstruct
    allowed = [np.ndarray, np.dtype, reconstruct,
               (reconstruct, 'numpy.core.multiarray._reconstruct')]
    allowed += list({type(np.dtype(t)) for t in numeric})
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', weights_only=True)


def decode_sample_inputs(value, encoding=None):
    """未知P2坐标方向时保守返回绝对/规范两种解码。"""
    x = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if x.shape != (6, 17, 17) or x.dtype.kind not in 'biuf' or not np.isfinite(x).all():
        raise ValueError('样本不是有限数值6×17×17编码')
    mask = np.zeros((17, 17), dtype=bool)
    for q, y in r.COORDS:
        mask[y+8, q+8] = True
    if not np.array_equal(x[2], mask):
        raise ValueError('棋盘mask与121格不符')
    if not np.isin(x[:2], [0, 1]).all() or np.any(x[:2, ~mask]):
        raise ValueError('棋子编码必须为mask内二值')
    if np.any((x[0] + x[1]) > 1):
        raise ValueError('双方占据同一格')
    identity = x[3][mask]
    if not (np.all(identity == 0) or np.all(identity == 1)):
        raise ValueError('玩家身份平面不恒定')
    player = 1 if identity[0] == 1 else 2
    if 'own-home-top' in str(encoding) or 'canonical' in str(encoding):
        orientations = [player == 2]
    elif encoding == 'axial17x17-6planes-pair121x121-v1':
        orientations = [False]
    else:
        orientations = [False, True] if player == 2 else [False]
    result = set()
    for reverse in orientations:
        a = x[:, ::-1, ::-1] if reverse else x
        board = [player if a[0,y+8,q+8] else 3-player if a[1,y+8,q+8] else 0 for q,y in r.COORDS]
        r.validate_board(board)
        result.add(decision_key(board, player))
    return result


def collect_history(root, exclude=()):
    root = Path(root).resolve()
    excluded = {Path(p).resolve() for p in exclude}
    states, file_hashes, seen_hashes = set(), {}, set()
    counts, issues = Counter(), []
    incomplete = []

    def add(identity):
        if isinstance(identity, str) and IDENTITY.fullmatch(identity):
            states.add(orbit(identity))
            counts['identity_occurrences'] += 1

    def moves(value):
        return isinstance(value, list) and all(isinstance(p, list) and len(p) >= 2 and
            all(isinstance(i, int) and 0 <= i < 121 for i in p) for p in value)

    def walk(obj, known121, location, encoding=None):
        if isinstance(obj, str):
            add(obj)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, known121, location, encoding)
        elif isinstance(obj, dict):
            encoding = obj.get('encoding', encoding)
            if 'state_dict' in obj and isinstance(obj['state_dict'], dict):
                counts['model_state_dict_records'] += 1
                return
            if 'x' in obj:
                try:
                    identities = decode_sample_inputs(obj['x'], encoding)
                    for identity in identities:
                        add(identity)
                    counts['decoded_tensor_samples'] += 1
                    counts['conservative_multi_orientation_samples'] += len(identities) > 1
                except (ValueError, TypeError) as error:
                    counts['undecoded_samples'] += 1
                    if known121 or tuple(getattr(obj['x'], 'shape', ())) == (6, 17, 17):
                        incomplete.append(dict(location=location, reason=str(error)))
                    else:
                        counts['non121_tensor_samples_skipped'] += 1
            for k in obj:
                add(k)
            board, player = obj.get('board'), obj.get('player')
            if isinstance(board, list) and len(board) == 121 and player in (1, 2) and all(v in (0, 1, 2) for v in board):
                add(decision_key(board, player))
            # Only replay records with evidence of the 121 rule profile.
            prefix = next((obj[k] for k in ('opening', 'prefix') if k in obj and moves(obj[k])), None)
            suffix = obj.get('paths')
            # PUCT paths enumerate alternative root actions, not successive plies.
            # Require the full search metadata shape; malformed marked records
            # remain incomplete rather than disappearing behind this classifier.
            candidate_marker = prefix is None and all(k in obj for k in ('paths','priors','policy','root_visits','child_visits'))
            candidate_metadata = False
            if candidate_marker:
                try:
                    n=len(suffix)
                    candidate_metadata = bool(n and moves(suffix) and len(obj['priors'])==len(obj['policy'])==n
                        and all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1 for v in obj['priors']+obj['policy'])
                        and abs(sum(obj['policy'])-1)<1e-5 and abs(sum(obj['priors'])-1)<1e-5
                        and obj['root_visits']==obj['child_visits'] and obj['root_visits']>0)
                except (TypeError,ValueError):candidate_metadata=False
                if candidate_metadata:
                    counts['search_candidate_records'] += 1
                    counts['search_candidate_paths'] += n
                else:
                    counts['invalid_search_candidate_records'] += 1
                    issues.append(dict(location=location,reason='非法搜索候选元数据'))
                    if known121:incomplete.append(issues[-1])
            sequences = []
            if prefix is not None:
                sequences.append(prefix + (suffix if moves(suffix) else []))
            elif not candidate_marker and moves(suffix) and obj.get('start_ply', 0) == 0:
                sequences.append(suffix)
            if sequences:
                if known121:
                    for sequence in sequences:
                        g = r.Game()
                        add(decision_key(g.board, g.player))
                        for index, path in enumerate(sequence):
                            if not r.valid_path(g.board, g.player, path):
                                counts['invalid_replay_records'] += 1
                                issues.append(dict(location=location, reason='非法121棋谱', ply=index))
                                incomplete.append(issues[-1])
                                break
                            try:
                                g.play(path)
                            except ValueError as error:
                                counts['invalid_replay_records'] += 1
                                issues.append(dict(location=location, reason=str(error), ply=index))
                                incomplete.append(issues[-1])
                                break
                            add(decision_key(g.board, g.player))
                        else:
                            counts['replayed_records'] += 1
                else:
                    counts['unclassified_or_non121_replay_records'] += 1
            elif not candidate_marker and moves(suffix) and suffix:
                counts['suffix_without_prefix_records'] += 1
                issues.append(dict(location=location, reason='非初始suffix缺少前缀'))
                if known121:
                    incomplete.append(issues[-1])
            for k, value in obj.items():
                walk(value, known121, location + '/' + str(k), encoding)

    profiles = {}
    def is121(path):
        for parent in (path.parent, *path.parents):
            if parent not in profiles:
                cfg = parent / 'config.json'
                try:
                    value = json.loads(cfg.read_text()) if cfg.is_file() else {}
                    profiles[parent] = str(value.get('profile', ''))
                except (ValueError, OSError):
                    profiles[parent] = ''
            if profiles[parent]:
                return '121' in profiles[parent]
            if parent == root:
                break
        return '121' in root.name or any('121' in part for part in path.relative_to(root).parts[:-1])

    for path in sorted(root.rglob('*')):
        if not path.is_file() or any(path.resolve() == e or e in path.resolve().parents for e in excluded):
            continue
        relative = path.relative_to(root)
        if any(part in ('source', '.git', '__pycache__') for part in relative.parts):
            counts['source_or_cache_files_skipped'] += 1
            continue
        if path.suffix not in ('.json', '.jsonl', '.pt'):
            if path.suffix in ('.pkl', '.pickle', '.npz'):
                counts['binary_files_not_parsed'] += 1
                if is121(path):
                    incomplete.append(dict(location=str(relative), reason='未读取121二进制pickle/npz'))
                else:
                    counts['non121_or_unclassified_binary_skipped'] += 1
            continue
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        file_hashes[str(relative)] = sha
        known121 = is121(path)
        dedup_key = (sha, known121)
        if dedup_key in seen_hashes:
            counts['duplicate_content_files_skipped'] += 1
            continue
        seen_hashes.add(dedup_key)
        try:
            if path.suffix == '.pt':
                records = [safe_tensor_load(path)]
                counts['safe_tensor_files_loaded'] += 1
            else:
                records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()] if path.suffix == '.jsonl' else [json.loads(raw)]
        except Exception as error:
            counts['parse_errors'] += 1
            issues.append(dict(location=str(relative), reason=str(error)))
            if known121 or path.suffix == '.pt':
                incomplete.append(dict(location=str(relative), reason='无法安全解析：' + str(error)))
            continue
        counts['parsed_files'] += 1
        known121 = is121(path)
        for index, record in enumerate(records):
            walk(record, known121, str(relative) + ':' + str(index))
    return dict(orbits=sorted(states), source_sha256=file_hashes, coverage=dict(counts), issues=issues, incomplete=incomplete,
                scope='JSON/JSONL局面及121棋谱所有状态、安全weights_only数值样本解码；模型state_dict仅分类。source跳过、data_sources保留，按SHA和规则分类去重。incomplete非空时不得宣称历史全覆盖。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('nn_checkpoints/experiments'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = collect_history(args.root, exclude=[args.output])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(orbits=len(result['orbits']), coverage=result['coverage']), ensure_ascii=False))


def resolve_legacy_endgames(manifest, root):
    """仅恢复curriculum121_100_20261029明确构造的末局前缀；不改历史原件。"""
    import copy
    result = copy.deepcopy(manifest)
    relative = 'curriculum121_100_20261029/endgame_evaluation.json'
    pending = [item for item in result.get('incomplete', [])
               if item.get('location', '').split(':')[0] == relative
               and item.get('reason') == '非初始suffix缺少前缀']
    if not pending:
        return result
    base = Path(root) / 'curriculum121_100_20261029'
    names = ['endgame_evaluation.json', 'teacher_games.jsonl', 'split.json']
    hashes = {}
    for name in names:
        raw = (base / name).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        key = 'curriculum121_100_20261029/' + name
        if manifest['source_sha256'].get(key) != sha:
            raise ValueError('历史来源SHA不一致：' + key)
        hashes[key] = sha
    teacher = [json.loads(line) for line in (base/'teacher_games.jsonl').read_text().splitlines() if line.strip()]
    split = json.loads((base/'split.json').read_text())
    selected = [g for g in teacher if g['group'] in split['test'] and g['winner'] in (1,2)][:10]
    endings = json.loads((base/'endgame_evaluation.json').read_text())
    if len(endings) != 2 * len(selected) or len(pending) != len(endings):
        raise ValueError('末局记录数量不符合原始构造')
    additional, evidence = set(), []
    for index, row in enumerate(endings):
        source = selected[index // 2]
        prefix = source['opening'] + source['paths'][:-8]
        if (row['model'] != ('prior', 'candidate')[index % 2]
                or row['group'] != source['group'] or row['side'] != source['winner']
                or row['start_ply'] != len(prefix)
                or row['plies'] != len(prefix) + len(row['paths'])):
            raise ValueError('末局来源、阵营、顺序或手数与原构造不符')
        g = r.Game()
        for path in prefix + row['paths']:
            additional.add(orbit(decision_key(g.board, g.player)))
            if not r.valid_path(g.board, g.player, path):
                raise ValueError('恢复末局出现非法走法')
            g.play(path)
        additional.add(orbit(decision_key(g.board, g.player)))
        evidence.append(dict(row=index, group=row['group'], model=row['model'],
                             side=row['side'], start_ply=row['start_ply'],
                             suffix_plies=len(row['paths']), prefix=prefix))
    before = set(result['orbits'])
    result['orbits'] = sorted(before | additional)
    result['incomplete'] = [item for item in result['incomplete'] if item not in pending]
    result.setdefault('resolved', []).append(dict(
        method='curriculum121.py:175—182：test胜局前10条，prefix=opening+paths[:-8]，prior/candidate依次续弈',
        original_incomplete=copy.deepcopy(pending), source_sha256=hashes,
        records=evidence, replayed_orbits=len(additional), added_orbits=len(additional-before)))
    result['coverage']['legacy_endgame_records_resolved'] = len(endings)
    return result


if __name__ == '__main__':
    main()
