"""第3步双权重价值对照中文图，仅读取结果JSON。"""
import json,sys,csv
from pathlib import Path
from plot_style_zh import configure
configure('路线图第三步价值学习训练开发开局组宏平均终局附近双方旧策略保护更新次数')
import numpy as np
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);r=json.loads((p/'result.json').read_text());names=['v3','head_only','joint'];labels=['原冠军','仅价值头','联合更新'];values={'v3':r['baseline'],**r['final']};fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');rows=[]
for arm,label,color in [('head_only','仅价值头','#4477aa'),('joint','联合更新','#dd8844')]:
    curve=json.loads((p/f'{arm}_curve.json').read_text());axes[0,0].plot([x['step'] for x in curve],[x['value_train']['group_macro']['mse'] for x in curve],label=label,color=color)
    for x in curve:rows.append(dict(方案=label,更新次数=x['step'],训练组宏价值误差=x['value_train']['group_macro']['mse'],训练策略交叉熵=x['policy_train']['policy_ce']))
axes[0,0].set(title='训练集：价值组宏误差',xlabel='更新次数',ylabel='均方误差');axes[0,0].legend();axes[0,0].grid(alpha=.2)
vv=[values[n]['development']['group_macro_mse'] for n in names];axes[0,1].bar(labels,vv,color=['#888888','#4477aa','#dd8844']);b=values['v3']['development'];axes[0,1].axhline(b['zero_group_macro_mse'],ls='--',color='#444444',label='中性价值0');axes[0,1].axhline(b['prior_group_macro_mse'],ls=':',color='#aa6677',label='训练阵营先验');axes[0,1].set_title('开发8组：价值组宏误差');axes[0,1].legend(fontsize=8)
for i,v in enumerate(vv):axes[0,1].text(i,v,f'{v:.3f}',ha='center',va='bottom')
for off,n,label,color in [(-.25,'v3','原冠军','#888888'),(0,'head_only','仅价值头','#4477aa'),(.25,'joint','联合更新','#dd8844')]:
    vv=[values[n]['development']['late16']['players'][str(pl)]['group_macro_mse'] for pl in (1,2)];axes[1,0].bar(np.arange(2)+off,vv,width=.24,label=label,color=color)
axes[1,0].set(xticks=[0,1],xticklabels=['P1','P2'],title='真实终局前16手：分方组宏误差');axes[1,0].legend(fontsize=8)
vv=[values[n]['retention']['policy_ce'] for n in names];axes[1,1].bar(labels,vv,color=['#888888','#4477aa','#dd8844']);axes[1,1].axhline(vv[0]+.05,ls='--',color='#444444',label='预定保持上限');axes[1,1].set_title('旧策略留出交叉熵');axes[1,1].legend(fontsize=8)
for i,v in enumerate(vv):axes[1,1].text(i,v,f'{v:.3f}',ha='center',va='bottom')
fig.suptitle('路线图第3步：双权重价值学习固定预算对照（未晋升）');fig.savefig(p/'双权重价值对照.png',dpi=170);fig.savefig(p/'双权重价值对照.svg');plt.close(fig)
svg=p/'双权重价值对照.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
with (p/'训练过程.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
