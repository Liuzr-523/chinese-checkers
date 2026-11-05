import sys,json,csv
from pathlib import Path
from plot_style_zh import configure
configure('训练开局命中率旧策略交叉熵旧价值均方误差更新次数原混合回放方向投影新开发差棋数平均分差')
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained');rows=[]
for arm,label in [('control','原混合'),('projected','回放方向投影')]:
    curve=json.loads((p/f'{arm}_curve.json').read_text())
    for ax,domain,key in [(axes[0],'book_train','top1_in_book'),(axes[1],'retention','policy_ce'),(axes[2],'retention','value_mse')]:ax.plot([r['step'] for r in curve],[r[domain][key] for r in curve],label=label)
    for r in curve:rows.append(dict(方案=label,更新次数=r['step'],训练开局命中率=r['book_train']['top1_in_book'],旧策略交叉熵=r['retention']['policy_ce'],旧价值均方误差=r['retention']['value_mse']))
for ax,title in zip(axes,['训练开局推荐命中率','旧策略交叉熵','旧价值均方误差']):ax.set(title=title,xlabel='参数更新次数');ax.legend();ax.grid(alpha=.2)
fig.savefig(p/'广开局投影训练.png',dpi=170);plt.close(fig)
r=json.loads((p/'result.json').read_text());names=['v3','control','projected'];labels=['原冠军','原混合','回放方向投影'];fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
for ax,k,title in [(axes[0],'bad','全新64局面：专家分差至少4的差棋数'),(axes[1],'mean_gap','全新64局面：平均专家分差')]:
    values=[r['new_development'][n][k] for n in names];ax.bar(labels,values,color=['#888888','#4477aa','#dd8844']);ax.set_title(title)
    for i,v in enumerate(values):ax.text(i,v,f'{v:.2f}' if k=='mean_gap' else str(v),ha='center',va='bottom')
fig.savefig(p/'全新开发专家质量.png',dpi=170);plt.close(fig)
with (p/'训练指标.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
