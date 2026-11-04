import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from plot_style_zh import configure
configure()
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);arms=['root_policy','policy_replay','full_objective'];fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for arm,color in zip(arms,['#4477aa','#228833','#cc6677']):
 h=json.loads((p/(arm+'_curve.json')).read_text());x=[r['step'] for r in h]
 axes[0,0].plot(x,[r['train32']['policy_ce'] for r in h],color=color,label={'root_policy':'仅根策略','policy_replay':'加普通策略回放','full_objective':'完整训练目标'}[arm])
 axes[0,1].plot(x,[r['train32']['accuracy'] for r in h],color=color,label={'root_policy':'仅根策略','policy_replay':'加普通策略回放','full_objective':'完整训练目标'}[arm])
 axes[1,0].plot(x,[r['validation40']['accuracy'] for r in h],color=color,label={'root_policy':'仅根策略','policy_replay':'加普通策略回放','full_objective':'完整训练目标'}[arm])
 axes[1,1].plot(x,[r['retention']['policy_ce'] for r in h],color=color,label={'root_policy':'仅根策略','policy_replay':'加普通策略回放','full_objective':'完整训练目标'}[arm])
for ax,title in zip(axes.flat,['固定32局面：策略交叉熵','固定32局面：专家首选一致率','未见40局面：专家首选一致率','旧能力保留：策略交叉熵']):
 ax.set(title=title,xlabel='参数更新次数');ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[0,1].set_ylim(0,1);axes[1,0].set_ylim(0,1)
fig.suptitle('相同32决策与根样本暴露次数；仅作拟合诊断')
fig.savefig(p/'fitting_curves.png',dpi=170);fig.savefig(p/'fitting_curves.svg');plt.close(fig)
