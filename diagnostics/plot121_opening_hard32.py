import json,sys,csv
from pathlib import Path
from plot_style_zh import configure
configure()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
p=Path(sys.argv[1]);fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');csvrows=[]
for arm,label,color in [('root_only','纯专项学习','#4477aa'),('mixed','保留旧能力的混合训练','#cc6677')]:
 h=json.loads((p/(arm+'_curve.json')).read_text());x=[r['step'] for r in h]
 for ax,domain,key in [(axes[0,0],'fixed32','top1_in_book'),(axes[0,1],'validation','top1_in_book'),(axes[1,0],'retention','policy_ce'),(axes[1,1],'retention','value_mse')]:ax.plot(x,[r[domain][key] for r in h],label=label,color=color)
 for r in h:csvrows.append({'方案':label,'更新次数':r['step'],'固定32命中率':r['fixed32']['top1_in_book'],'留出128命中率':r['validation']['top1_in_book'],'旧策略交叉熵':r['retention']['policy_ce'],'旧价值均方误差':r['retention']['value_mse']})
for ax,title in zip(axes.flat,['32个后段训练难例拟合','原开局留出集：推荐命中率','旧能力：策略交叉熵','旧能力：价值均方误差']):ax.set(title=title,xlabel='参数更新次数');ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[0,0].axhline(.9,ls=':',color='gray');axes[0,0].set_ylim(0,1);axes[0,1].set_ylim(0,1);fig.suptitle('后段32难例：同初值、同专项样本序列与系数、同2400更新')
fig.savefig(p/'后段32难例对照.png',dpi=170);fig.savefig(p/'后段32难例对照.svg');plt.close(fig)
metadata=json.loads((p/'fixed32.json').read_text())['metadata'];ids=[s['decision_id'] for s in metadata];labels=[f'例{i+1:02d}·P{s["player"]}第{s["own_move"]}步' for i,s in enumerate(metadata)]
fig,axes=plt.subplots(1,2,figsize=(13,11),layout='constrained')
for ax,arm,label in [(axes[0],'root_only','纯专项学习'),(axes[1],'mixed','保留旧能力的混合训练')]:
 rows=json.loads((p/(arm+'_cases.json')).read_text());steps=sorted({r['step'] for r in rows});lookup={(r['decision_id'],r['step']):r['correct'] for r in rows};array=np.array([[lookup[did,t] for t in steps] for did in ids]);ax.imshow(array,aspect='auto',cmap=ListedColormap(['#edb4b7','#69ad7d']),vmin=0,vmax=1);ax.set(title=label,yticks=range(32),yticklabels=labels,xticks=range(0,len(steps),2),xticklabels=steps[::2],xlabel='参数更新次数');ax.tick_params(axis='x',rotation=60);ax.tick_params(axis='y',labelsize=8)
fig.suptitle('逐例实际首选：绿色为库内推荐，粉色为库外动作；每100步记录一次')
fig.savefig(p/'逐例掌握过程.png',dpi=170);plt.close(fig)
svg=p/'后段32难例对照.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
with (p/'后段32难例对照.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(csvrows[0]),lineterminator='\n');w.writeheader();w.writerows(csvrows)
