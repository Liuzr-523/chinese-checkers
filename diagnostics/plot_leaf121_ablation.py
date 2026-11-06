"""叶值消融归档中文图，不运行搜索。"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diagnostics.plot_style_zh import configure
configure('叶子评价消融改变选择根动作覆盖率模拟次数局面数原价值零值手工值纯策略实战胜负真和截断不是独立棋力验收')
import matplotlib.pyplot as plt
folder=Path(sys.argv[1]);r=json.loads((folder/'artifacts/result.json').read_text())
labels={'nn':'原价值','zero':'零值','hand':'手工值','pure':'纯策略'}
fig,axes=plt.subplots(1,3,figsize=(13,4.5))
for arm,color in [('nn','#317e9e'),('zero','#a2adb5'),('hand','#d69835')]:
    xs=[next(x for x in r['cases'] if x['split']=='all' and x['arm']==arm and x['simulations']==n) for n in (32,128,512)]
    axes[0].plot(range(3),[x['changed_from_policy'] for x in xs],marker='o',color=color,label=labels[arm])
    axes[1].plot(range(3),[100*x['mean_coverage'] for x in xs],marker='o',color=color,label=labels[arm])
for ax in axes[:2]:ax.set_xticks(range(3),[32,128,512]);ax.set_xlabel('模拟次数');ax.legend(fontsize=8)
axes[0].set_title('相对纯策略改变选择');axes[0].set_ylabel('局面数 / 48')
axes[1].set_title('根动作覆盖率');axes[1].set_ylabel('覆盖率（%）')
bottom=[0]*4;arms=['nn','zero','hand','pure']
for key,label,color in [('nn_wins','胜','#317e9e'),('nn_losses','负','#bf7766'),('true_draws','真和','#bdc6cd'),('truncated','截断','#e0bc6e')]:
    vs=[r['arena'][a][key] for a in arms];axes[2].bar(range(4),vs,bottom=bottom,color=color,label=label);bottom=[x+y for x,y in zip(bottom,vs)]
axes[2].set_xticks(range(4),[labels[a] for a in arms]);axes[2].set_ylabel('局数 / 12');axes[2].set_title('对固定参考引擎实战');axes[2].set_ylim(0,16);axes[2].legend(fontsize=8)
fig.suptitle('固定策略先验的叶子评价消融（已有诊断局面）');fig.tight_layout();fig.savefig(folder/'叶值消融.png',dpi=160)
