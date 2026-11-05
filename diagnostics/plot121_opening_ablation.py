import sys,json,csv
from pathlib import Path
from plot_style_zh import configure
configure('专项回放保护完整混合训练命中率更新次数旧策略交叉熵价值均方误差')
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');rows=[]
for arm,label in [('root_only','纯专项'),('with_replay','专项＋策略回放'),('with_aux','专项＋价值及旧输出保护'),('full','完整混合')]:
 curve=json.loads((p/(arm+'_curve.json')).read_text())
 for ax,domain,key in [(axes[0,0],'fixed32','top1_in_book'),(axes[0,1],'validation','top1_in_book'),(axes[1,0],'retention','policy_ce'),(axes[1,1],'retention','value_mse')]:ax.plot([r['step'] for r in curve],[r[domain][key] for r in curve],label=label)
 for r in curve:rows.append(dict(方案=label,更新次数=r['step'],难例命中率=r['fixed32']['top1_in_book'],留出命中率=r['validation']['top1_in_book'],旧策略交叉熵=r['retention']['policy_ce'],旧价值均方误差=r['retention']['value_mse']))
for ax,title in zip(axes.flat,['固定32难例命中率','128开局留出命中率','旧策略交叉熵','旧价值均方误差']):ax.set(title=title,xlabel='参数更新次数');ax.legend(fontsize=8);ax.grid(alpha=.2)
axes[0,0].set_ylim(0,1);axes[0,1].set_ylim(0,1);fig.suptitle('固定32难例：策略回放与价值保护拆分')
fig.savefig(p/'四组损失拆分.png',dpi=170);fig.savefig(p/'四组损失拆分.svg');plt.close(fig)
svg=p/'四组损失拆分.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
with (p/'四组损失拆分.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
