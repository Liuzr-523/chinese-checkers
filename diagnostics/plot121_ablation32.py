import csv,json,sys
from pathlib import Path
from plot_style_zh import configure
configure()
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');rows=[]
for arm,label,color in [('control','原方案','#4477aa'),('ties_ce_only','仅接受同分好棋','#228833'),('best_rank_only','仅最优棋对差棋排序','#cc6677')]:
 h=json.loads((p/(arm+'_curve.json')).read_text());x=[r['step'] for r in h]
 for ax,domain,key in [(axes[0,0],'train32','accuracy'),(axes[0,1],'validation40','accuracy'),(axes[1,0],'retention','policy_ce'),(axes[1,1],'retention','value_mse')]:ax.plot(x,[r[domain][key] for r in h],label=label,color=color)
 for v in h:rows.append({'方案':label,'更新次数':v['step'],'训练专家首选一致率':v['train32']['accuracy'],'旧开发验证首选一致率':v['validation40']['accuracy'],'旧策略交叉熵':v['retention']['policy_ce'],'旧价值均方误差':v['retention']['value_mse']})
for ax,title in zip(axes.flat,['固定32训练：专家首选一致率','旧40开发验证：专家首选一致率','旧能力保留：策略交叉熵','旧能力保留：价值均方误差']):ax.set(title=title,xlabel='参数更新次数');ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[0,0].set_ylim(0,1);axes[0,1].set_ylim(0,1);fig.suptitle('监督因素分别对照：同分动作评分另见报告；不代表棋力验收')
fig.savefig(p/'监督拆分曲线.png',dpi=170);fig.savefig(p/'监督拆分曲线.svg');plt.close(fig)
with (p/'监督拆分曲线.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
