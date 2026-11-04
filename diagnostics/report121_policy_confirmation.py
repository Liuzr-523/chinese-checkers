"""完整验收收尾：重放全部200棋谱，中文报告和对战结果图。"""
import json,sys,hashlib,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from curriculum121 import replay,atomic
from iterate121 import summary,paired_gate
from checkpoint121_save import digest

def finalize(folder):
 p=Path(folder);cfg=json.loads((p/'config.json').read_text());res=json.loads((p/'result.json').read_text());logs={};reasons={};stages={}
 for name,n in [('confirmation',120),('fixed_c_control',40),('fixed_c_candidate',40)]:
  rows=[json.loads(s) for s in (p/(name+'.jsonl')).read_text().splitlines()]
  if len(rows)!=n:raise ValueError('对局数量不符：'+name)
  for row in rows:
   g=replay(row['opening'])
   for path in row['paths']:g.play(path)
   if g.result()!=(row['winner'],row['reason']) or g.ply!=row['plies']:raise ValueError('终局重放不符')
  logs[name]=rows;reasons[name]={reason:sum(r['reason']==reason for r in rows) for reason in sorted({r['reason'] for r in rows})}
  stages[name]=dict(mean_plies=sum(r['plies'] for r in rows)/len(rows),maximum_plies=max(r['plies'] for r in rows))
 if summary(logs['confirmation'])!=res['confirmation'] or paired_gate(logs['confirmation'])!=res['paired_gate']:raise ValueError('汇总分数不符')
 for name in ('control','candidate'):
  if summary(logs['fixed_c_'+name])!=res['fixed_c'][name]:raise ValueError('对C汇总不符')
 for name,sha in cfg['source_sha256'].items():
  if digest(p/'source'/name)!=sha:raise ValueError('源码快照变化')
 for name,sha in cfg['data_sha256'].items():
  if digest(p/'data_sources'/name)!=sha:raise ValueError('输入快照变化')
 if digest(p/'candidate.pt')!=cfg['candidate_sha256'] or digest(cfg['model'])!=cfg['model_sha256']:raise ValueError('模型变化')
 atomic(p/'full_replay_audit.json',dict(games_verified=200,failures=0,termination_reasons=reasons,game_lengths=stages,model_and_source_hashes_verified=True))
 adoption=json.loads((p/'adoption.json').read_text());res['promoted']=adoption.get('adopted',False);atomic(p/'result.json',res)
 lines=['# 神经网络候选：完整对局验收','',f'验收结论：{"通过，已登记新版本" if res["promoted"] else "未通过，保留v3"}。',
 '候选固定为'+cfg['candidate_source']+'。双方MCTS32，固定120局对v3与新旧各40局对C-PVS；C每手30毫秒、最大深度10。双方开局库配置：'+str(cfg.get('opening_book_for_both',False))+'。所有开局按组交换先后手，跑满预算。',
 '|对战|胜|负|和|积分|','|---|---:|---:|---:|---:|']
 for title,v in [('候选对v3',res['confirmation']),('v3对C-PVS',res['fixed_c']['control']),('候选对C-PVS',res['fixed_c']['candidate'])]:
  lines.append('|'+title+'|'+str(v['wins'])+'|'+str(v['losses'])+'|'+str(v['draws'])+'|'+str(v['points'])+'|')
 gate=res['paired_gate'];lines+=['', '开局配对检验：'+json.dumps(gate,ensure_ascii=False),'','升级规则预先固定：候选对v3配对单侧p<=0.01，且对C-PVS积分不低于v3。通过后完整哈希/棋谱审计与GUI接口烟测；v3存档保留。',
 '本轮200局已逐手合法重放，终局和汇总分数全部核对通过。终局类型与对局长度见full_replay_audit.json。',
 '后续决定：'+('从新冠军生成后续自对弈数据，继续保留独立验收。' if res['promoted'] else '结束该候选。优先分析本轮完整输棋记录中的开局、中盘与收官表现，下一次改动针对实际输棋问题。'),
 '完整棋谱：confirmation.jsonl、fixed_c_control.jsonl、fixed_c_candidate.jsonl；中文日志、源码/数据/模型快照及开局去重日程均保存。']
 lines.append('中文战绩图：完整对局验收.png / 完整对局验收.svg。')
 (p/'REPORT.md').write_text('\n'.join(lines))
 plotted=subprocess.run(['/Library/Frameworks/Python.framework/Versions/3.10/bin/python3','diagnostics/plot121_policy_confirmation.py',str(p)],capture_output=True,text=True)
 atomic(p/'plot_status.json',dict(exit_code=plotted.returncode,stderr=plotted.stderr))
 if plotted.returncode:raise RuntimeError('中文战绩图绘制失败')
 with (p/'中文日志.md').open('a') as f:f.write('\n200局完整重放与哈希审计通过。最终结果：'+json.dumps(res,ensure_ascii=False)+'\n')
if __name__=='__main__':finalize(Path(sys.argv[1]))
