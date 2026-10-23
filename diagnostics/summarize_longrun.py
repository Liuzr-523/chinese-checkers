"""Read live artifacts and regenerate a concise, human-readable progress report."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'nn_checkpoints/experiments/longrun_20261022'
state=json.loads((run/'status.json').read_text())
events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines() if line.strip()]
games=[e for e in events if e['event']=='game']
# Resumed stages log a game only once after its atomic save.
lines=['# 自对弈长跑进度', '', f"更新时间：{state['updated']}（UTC，+8小时为北京时间）",f"状态：{state['status']}，第{state['round']}轮，阶段 {state['phase']}", '',
       '所有截断/重复局面均记为未知，以下胜负仅来自真实终局。GUI冠军未覆盖。','',
       '| 轮次 | 自对弈完赛 | 候选对本轮基线 | 候选对原冠军 | 内部晋升 |','|---|---|---|---|---|']
for p in sorted(run.glob('round_*/result.json')):
 r=json.loads(p.read_text())
 if r.get('skipped'):continue
 def score(s):return f"{s['wins']}胜{s['losses']}负{s['unfinished']}未结束"
 lines.append(f"| {r['round']} | {r['selfplay_complete']}/{r['selfplay_games']} | {score(r['incumbent'])} | {score(r['frozen'])} | {r['accepted']} |")
lines+=['','## 对局日志','', '| 时间 UTC | 轮次 | 阶段 | 局次 | 胜方 | 原因 | 手数 |', '|---|---|---|---|---|---|---|']
for e in games:lines.append(f"| {e['time'][11:19]} | {e['round']} | {e['phase']} | {e['game']} | {e['winner']} | {e['reason']} | {e['plies']} |")
lines+=['','## 文件', '', '- `events.jsonl`：按时间追加的完整事件日志。','- `status.json`：运行PID、阶段、手数、更新时间，可恢复定位。','- `round_*/selfplay.pkl`：新自对弈棋谱，每局保存并校验。','- `round_*/validation.json`：整局隔离的验证指标与种子。','- `round_*/candidate.pt`：独立候选模型。','- `round_*/incumbent_eval.pkl`、`frozen_eval.pkl`：选模与独立对战棋谱。']
(run/'PROGRESS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(state=state,finished_games=len(games),terminal_games=sum(e['winner'] is not None for e in games)),ensure_ascii=False))
