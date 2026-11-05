"""只读统计121格开局/普通回放的阶段覆盖、重复监督与真实终局标签。"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cap_opening121_replay import load_data
from history121_exclusions import decode_sample_inputs
from opening121_split_audit import orbit
from az121 import CANONICAL_ENCODING

STAGES = ('1—3', '4—6', '7—10', '11—20', '21—40', '41及以后')


def stage_of(own_move):
    for label, maximum in zip(STAGES, (3, 6, 10, 20, 40, float('inf'))):
        if own_move <= maximum:
            return label


def describe(sample):
    x = np.asarray(sample['x'])
    identities = decode_sample_inputs(x, CANONICAL_ENCODING)
    if len(identities) != 1:
        raise ValueError('已知规范编码解码不唯一')
    identity = next(iter(identities))
    player = int(identity[-1])
    ply = round(float(x[4].max()) * 240)
    positive = tuple(sorted(set(sample.get('positive', [sample['actions'][sample['target']]]))))
    return dict(identity=identity, orbit=orbit(identity), player=player, ply=ply,
                own_move=ply//2+1, stage=stage_of(ply//2+1), labels=positive,
                z=float(sample['z']), value_weight=float(sample.get('value_weight', 1.)))


def duplicate_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['identity']].append(row)
    details = []
    for identity, ss in groups.items():
        if len(ss) < 2:
            continue
        signatures = {s['labels'] for s in ss}
        outcomes = {s['z'] for s in ss if s['value_weight'] > 0}
        disjoint = any(set(a).isdisjoint(b) for a in signatures for b in signatures)
        details.append(dict(decision_id=identity, records=len(ss), supervision_sets=len(signatures),
                            disjoint_policy_sets=disjoint, value_labels=sorted(outcomes),
                            distinct_ply=sorted({s['ply'] for s in ss}),
                            value_label_diversity=len(outcomes)>1))
    return dict(repeated_board_turn_groups=len(details),
                repeated_records=sum(d['records'] for d in details),
                differing_policy_groups=sum(d['supervision_sets']>1 for d in details),
                disjoint_policy_groups=sum(d['disjoint_policy_sets'] for d in details),
                differing_value_groups=sum(d['value_label_diversity'] for d in details), details=details)


def summarize(samples):
    rows = [describe(s) for s in samples]
    cells = []
    for player in (1, 2):
        for stage in STAGES:
            ss = [s for s in rows if s['player']==player and s['stage']==stage]
            cells.append(dict(player=player, stage=stage, records=len(ss),
                              unique_board_turn=len({s['identity'] for s in ss}),
                              unique_full_orbits=len({s['orbit'] for s in ss}),
                              active_value_labels=dict(Counter(str(s['z']) for s in ss if s['value_weight']>0)),
                              zero_value_weight=sum(s['value_weight']==0 for s in ss)))
    return dict(records=len(rows), unique_board_turn=len({s['identity'] for s in rows}),
                unique_full_orbits=len({s['orbit'] for s in rows}),
                players=dict(Counter(s['player'] for s in rows)), cells=cells,
                player_outcomes=[dict(player=p, active_labels=dict(Counter(str(s['z']) for s in rows if s['player']==p and s['value_weight']>0)), zero_value_weight=sum(s['player']==p and s['value_weight']==0 for s in rows)) for p in (1,2)],
                active_value_labels=dict(Counter(str(s['z']) for s in rows if s['value_weight']>0)),
                zero_value_weight=sum(s['value_weight']==0 for s in rows),
                duplicates=duplicate_summary(rows))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path('nn_checkpoints/experiments/fresh121_projection_20261105/data_sources'))
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args()
    _,old,normal,book=load_data(args.source)
    pools={'book_train':book['train'],'book_validation':book['validation'],
           'old_train':old['train'],'old_validation':old['validation'],
           'normal_train':normal['train'],'normal_validation':normal['validation']}
    report={name:summarize(ss) for name,ss in pools.items()}
    report['combined_training']=summarize(book['train']+old['train']+normal['train'])
    sources=[args.source/n for n in ('fixed32.pt','old_teacher.jsonl','old_split.json','normal_games.jsonl')]
    sources += [Path(n) for n in ('opening121_book.json','cap_opening121_replay.py','train_opening121.py','curriculum121.py','cycle121_targeted.py','history121_exclusions.py','az121.py','diagnostics/diagnose121_coverage.py')]
    raw_sources={}
    split=json.loads((args.source/'old_split.json').read_text())
    for filename in ('old_teacher.jsonl','normal_games.jsonl'):
        grouped=defaultdict(list)
        for line in (args.source/filename).read_text().splitlines():
            row=json.loads(line)
            part=row.get('split') if filename=='normal_games.jsonl' else next((p for p in ('train','validation') if row['group'] in split[p]),'other')
            grouped[part].append(row)
        raw_sources[filename]={part:dict(games=len(rs), reasons=dict(Counter(str(g['reason']) for g in rs)), winners=dict(Counter(str(g['winner']) for g in rs)), draw_records=[dict(reason=g['reason'], paths=len(g['paths']), included_in_samples=filename!='old_teacher.jsonl', expected_value_weight=(0 if g['reason']=='max_ply' else 1) if filename!='old_teacher.jsonl' else None) for g in rs if g['winner']==0]) for part,rs in grouped.items()}
    result=dict(pools=report, raw_source_game_outcomes=raw_sources, source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                definitions=dict(stage='己方步数=floor(总ply/2)+1；ply取规范输入plane4×240四舍五入，player取plane3；全部为已知canonical解码。',
                                 unique='精确唯一为完整board:player，不含ply/repetition；全轨道额外归一左右镜像及180度旋转换色。',
                                 outcomes='仅value_weight>0的z计入价值标签，z为当前走棋方视角真实终局结果。开局root的z=0是占位且value_weight=0，不记为和棋。普通真实终局由既有load_data终局重放校验。',
                                 conflicts='同棋盘不同棋局可有不同合法着法及不同真实终局。这里报告监督多样性与共享参数张力，不直接将其标为污染；不同ply/repetition可能对应不同完整输入。',
                                 stage_symmetry='每个player阶段cell分别统计；合并全轨道数可能小于双方轨道数之和。'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for name, row in report.items():
        print(name, row['records'], row['unique_board_turn'], row['unique_full_orbits'], row['active_value_labels'])
        print('7—10', [c for c in row['cells'] if c['stage']=='7—10'])
        print('duplicates', {k:v for k,v in row['duplicates'].items() if k!='details'})


if __name__ == '__main__':
    main()
