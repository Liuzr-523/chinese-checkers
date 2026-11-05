import json,sys,csv
from pathlib import Path
from plot_style_zh import configure
configure('专项回放梯度范数上限原配方对照更新次数开局命中率旧策略交叉熵价值均方误差实际')
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');rows=[]
for arm,label in [('control','原完整混合'),('capped','回放梯度范数上限')]:
    curve=json.loads((p/f'{arm}_curve.json').read_text())
    for ax,domain,key in [(axes[0,0],'fixed32','top1_in_book'),(axes[0,1],'validation','top1_in_book'),(axes[1,0],'retention','policy_ce'),(axes[1,1],'retention','value_mse')]:ax.plot([r['step'] for r in curve],[r[domain][key] for r in curve],label=label)
    for r in curve:rows.append(dict(方案=label,更新次数=r['step'],固定难例命中率=r['fixed32']['top1_in_book'],原留出命中率=r['validation']['top1_in_book'],旧策略交叉熵=r['retention']['policy_ce'],旧价值均方误差=r['retention']['value_mse']))
for ax,title in zip(axes.flat,['固定32难例命中率','原128开局留出命中率','旧策略交叉熵','旧价值均方误差']):ax.set(title=title,xlabel='参数更新次数');ax.grid(alpha=.2);ax.legend(fontsize=9)
axes[0,0].set_ylim(0,1);axes[0,1].set_ylim(0,1);fig.suptitle('原完整混合与回放梯度上限：同初值、同样本序列、同更新次数')
fig.savefig(p/'回放梯度上限对照.png',dpi=170);fig.savefig(p/'回放梯度上限对照.svg');plt.close(fig)
svg=p/'回放梯度上限对照.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
with (p/'回放梯度上限对照.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
