import csv,json,sys
from pathlib import Path
from plot_style_zh import configure
configure()
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);cfg=json.loads((p/'config.json').read_text());result=json.loads((p/'result.json').read_text());fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained');rows=[]
for seed,ls in zip(cfg['seeds'],['-','--']):
 for arm,color,label in [('control','#4477aa','原方案'),('dynamic','#cc6677','动态实际差棋纠错')]:
  name=f'{arm}_{seed}';h=json.loads((p/(name+'_curve.json')).read_text());x=[r['step'] for r in h];legend=label+'／种子'+str(seed)
  axes[0,0].plot(x,[r['train']['accuracy'] for r in h],ls=ls,color=color,label=legend);axes[0,1].plot(x,[r['retention']['policy_ce'] for r in h],ls=ls,color=color,label=legend);axes[1,0].plot(x,[r['retention']['value_mse'] for r in h],ls=ls,color=color,label=legend)
  for r in h:rows.append({'方案':label,'种子':seed,'更新次数':r['step'],'训练专家首选一致率':r['train']['accuracy'],'旧策略交叉熵':r['retention']['policy_ce'],'旧价值均方误差':r['retention']['value_mse']})
labels=[];bad=[]
for seed in cfg['seeds']:
 for arm,label in [('control','原方案'),('dynamic','动态纠错')]:labels.append(label+'\n种子'+str(seed));bad.append(result['final'][f'{arm}_{seed}']['fresh40_actual']['bad_moves'])
bars=axes[1,1].bar(labels,bad,color=['#4477aa','#cc6677']*2);axes[1,1].bar_label(bars);axes[1,1].set(title='40个全新开发状态：实际首选差棋数',ylabel='专家分差至少4的局面数',ylim=(0,40))
for ax,title in zip(axes.flat,['训练根：专家首选一致率','旧能力保留：策略交叉熵','旧能力保留：价值均方误差']):ax.set(title=title,xlabel='参数更新次数');ax.legend(fontsize=8);ax.grid(alpha=.2)
axes[0,0].set_ylim(0,1);fig.suptitle('动态纠错对照：训练与旧能力曲线；全新开发只评固定最终模型')
fig.savefig(p/'动态纠错对照.png',dpi=170);fig.savefig(p/'动态纠错对照.svg');plt.close(fig)
with (p/'动态纠错曲线.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
