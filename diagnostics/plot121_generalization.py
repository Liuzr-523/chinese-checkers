"""Standalone matplotlib renderer; uses system Python, no Torch needed."""
import csv,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from plot_style_zh import configure
configure()
import matplotlib.pyplot as plt
folder=Path(sys.argv[1]);arms=['small72','expanded160'];curves={a:json.loads((folder/(a+'_curve.json')).read_text()) for a in arms}
rows=[]
for arm,history in curves.items():
    for record in history:
        for domain,values in record.items():
            if domain=='step':continue
            rows.append(dict(arm=arm,step=record['step'],domain=domain,policy_ce=values['policy_ce'],policy_accuracy=values.get('accuracy',values.get('policy_accuracy')),preference_accuracy=values.get('preference_accuracy',''),value_mse=values.get('value_mse','')))
with (folder/'learning_curves.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for arm,color in [('small72','#4477aa'),('expanded160','#cc6677')]:
    h=curves[arm];steps=[x['step'] for x in h]
    axes[0,0].plot(steps,[x['seen_train']['policy_ce'] for x in h],color=color,ls='--',label=('72训练根' if arm=='small72' else '160训练根')+'／训练集')
    axes[0,0].plot(steps,[x['validation40']['policy_ce'] for x in h],color=color,label=('72训练根' if arm=='small72' else '160训练根')+'／验证集')
    axes[0,1].plot(steps,[x['seen_train']['preference_accuracy'] for x in h],color=color,ls='--',label=('72训练根' if arm=='small72' else '160训练根')+'／训练集')
    axes[0,1].plot(steps,[x['validation40']['preference_accuracy'] for x in h],color=color,label=('72训练根' if arm=='small72' else '160训练根')+'／验证集')
    axes[1,0].plot(steps,[x['retention_validation']['policy_ce'] for x in h],color=color,label='72训练根' if arm=='small72' else '160训练根')
    axes[1,1].plot(steps,[x['retention_validation']['value_mse'] for x in h],color=color,label='72训练根' if arm=='small72' else '160训练根')
for ax,title in zip(axes.flat,['专家决策交叉熵','好棋相对原坏棋的偏好正确率','旧能力保留：策略交叉熵','旧能力保留：价值均方误差']):
    ax.set(title=title,xlabel='参数更新次数');ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[0,1].set_ylim(0,1)
fig.suptitle('72与160个训练决策对照；相同40验证局面与训练预算')
fig.savefig(folder/'learning_curves.png',dpi=170);fig.savefig(folder/'learning_curves.svg');plt.close(fig)
