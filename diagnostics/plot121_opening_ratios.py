import json,sys,csv
from pathlib import Path
from plot_style_zh import configure
configure()
import matplotlib.pyplot as plt
import numpy as np
p=Path(sys.argv[1]);r=json.loads((p/'result.json').read_text());fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');rows=[]
for arm,color in [('book25','#4477aa'),('book50','#228833'),('book75','#cc6677')]:
 h=json.loads((p/(arm+'_curve.json')).read_text());x=[v['step'] for v in h];label='开局份额'+arm[4:]+'%'
 for ax,domain,key in [(axes[0,0],'book_train','top1_in_book'),(axes[0,1],'book_validation','top1_in_book'),(axes[1,0],'retention','policy_ce')]:ax.plot(x,[v[domain][key] for v in h],label=label,color=color)
 for v in h:rows.append({'开局份额':arm[4:]+'%','更新次数':v['step'],'训练命中率':v['book_train']['top1_in_book'],'留出命中率':v['book_validation']['top1_in_book'],'标准前三步双方命中率':v['standard_first6']['top1_in_book'],'旧策略交叉熵':v['retention']['policy_ce'],'旧价值均方误差':v['retention']['value_mse']})
for ax,title in zip(axes.flat,['训练局面：推荐着法命中率','留出局面：推荐着法命中率','旧能力保留：策略交叉熵']):ax.set(title=title,xlabel='参数更新次数');ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[0,0].axhline(.9,color='gray',ls=':',label='训练目标90%');axes[0,0].set_ylim(0,1);axes[0,1].set_ylim(0,1)
values=np.array([[r['final'][arm]['train_stages'][stage]['top1_in_book'] for stage in ('moves1_3','moves4_6','moves7_10')] for arm in ('book25','book50','book75')]);im=axes[1,1].imshow(values,vmin=0,vmax=1,cmap='YlGn');axes[1,1].set(title='固定最终模型：训练开局分阶段掌握率',xticks=range(3),xticklabels=['己方第1—3步','己方第4—6步','己方第7—10步'],yticks=range(3),yticklabels=['开局25%','开局50%','开局75%'])
for i in range(3):
 for j in range(3):axes[1,1].text(j,i,f'{values[i,j]:.1%}',ha='center',va='center')
fig.suptitle('开局样本份额对照：相同CNN、初值、学习率与2400更新；原始网络关闭外部开局库')
fig.savefig(p/'开局份额对照.png',dpi=170);fig.savefig(p/'开局份额对照.svg');plt.close(fig);svg=p/'开局份额对照.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
with (p/'开局份额对照.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
