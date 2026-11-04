import json,sys
from pathlib import Path
from plot_style_zh import configure
configure()
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);s=json.loads((p/'summary.json').read_text());e=json.loads((p/'expert_summary.json').read_text());fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for key,label,color in [('win','候选胜局','#228833'),('loss','候选败局','#cc6677')]:
 h=s['stage_summary'][key];axes[0].plot([int(x) for x in h],[r['mean_assignment_deficit'] for r in h.values()],marker='o',label=label,color=color)
axes[0].axhline(0,color='gray');axes[0].set(title='推进差距在中盘扩大',xlabel='总手数',ylabel='候选相对对手的剩余距离缺口（正值落后）');axes[0].grid(alpha=.2);axes[0].legend()
labels=['开局','中盘','收官'];values=[e['by_phase'][p]['gap4'] for p in ('opening','middle','endgame')];bars=axes[1].bar(labels,values,color=['#4477aa','#cc6677','#aaaaaa']);axes[1].bar_label(bars);axes[1].set(title='针对性专家复查：每阶段抽查八个动作',ylabel='与专家分差至少四分的动作数',ylim=(0,8));fig.suptitle('120局输棋分析：双败开局重点复查，优先处理中盘调度')
fig.savefig(p/'输棋阶段分析.png',dpi=170);fig.savefig(p/'输棋阶段分析.svg');plt.close(fig)

svg=p/'输棋阶段分析.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
