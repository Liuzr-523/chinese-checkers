"""路线图第一步只读诊断的简体中文图；从已保存JSON绘图，不运行模型。"""
import json,sys,csv
from pathlib import Path
from plot_style_zh import configure
configure('路线图第一步数据价值策略诊断训练留出胜标签比例仅胜负样本固定先手身份敏感性')
import numpy as np
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);coverage=json.loads((p/'coverage.json').read_text());value=json.loads((p/'value.json').read_text());transfer=json.loads((p/'transfer.json').read_text());identity=json.loads((p/'identity.json').read_text());fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');export=[]
for offset,pool,label,color in [(-.18,'normal_train','训练：12局','#4477aa'),(.18,'normal_validation','留出：4局','#cc6677')]:
    vals=[]
    for row in coverage['pools'][pool]['player_outcomes']:
        z=row['active_labels'];vals.append(z.get('1.0',0)/(z.get('1.0',0)+z.get('-1.0',0)));export.append(dict(指标='胜标签比例',数据=pool,模型=f'P{row["player"]}',数值=vals[-1]))
    bars=axes[0,0].bar(np.arange(2)+offset,vals,width=.35,label=label,color=color)
    for b,v in zip(bars,vals):axes[0,0].text(b.get_x()+b.get_width()/2,v,f'{v:.1%}',ha='center',va='bottom')
axes[0,0].set(xticks=[0,1],xticklabels=['P1','P2'],ylim=(0,1),title='普通数据：阵营胜标签比例翻转',ylabel='仅有胜负样本；同局样本相关');axes[0,0].legend()
names=['v3','control','projected'];labels=['原冠军','原混合','投影组'];vals=[next(r['mse'] for r in value['rows'] if r['source']=='normal' and r['split']=='validation' and r['model']==m and r['stage']=='总计' and r['player']==0) for m in names]
axes[0,1].bar(labels,vals,color=['#888888','#4477aa','#dd8844']);axes[0,1].axhline(1.,ls='--',color='#333333',label='始终预测0');axes[0,1].set(title='价值头：普通留出误差',ylabel='均方误差（越低越好）',ylim=(0,1.35));axes[0,1].legend()
for i,v in enumerate(vals):axes[0,1].text(i,v,f'{v:.3f}',ha='center',va='bottom');export.append(dict(指标='价值均方误差',数据='普通留出',模型=names[i],数值=v))
vals=[transfer['distribution'][key]['late_moves7_10']['mean_exited_home'] for key in ('train','fresh_development')];axes[1,0].bar(['开局库训练','随机开发'],vals,color=['#4477aa','#dd8844']);axes[1,0].set(title='同为己方第7—10步：棋形进度不同',ylabel='平均当前出营棋子数',ylim=(0,8))
for i,v in enumerate(vals):axes[1,0].text(i,v,f'{v:.2f}',ha='center',va='bottom');export.append(dict(指标='出营棋子数',数据=['库训练','随机开发'][i],模型='局面统计',数值=v))
vals=[transfer['summary'][m]['fresh_development']['overall']['mean_target_rank'] for m in names];axes[1,1].bar(labels,vals,color=['#888888','#4477aa','#dd8844']);axes[1,1].set(title='既有64开发：单一专家建议的名次',ylabel='平均名次（越低越好）',ylim=(0,10))
for i,v in enumerate(vals):axes[1,1].text(i,v,f'{v:.2f}',ha='center',va='bottom');export.append(dict(指标='建议平均名次',数据='既有开发64',模型=names[i],数值=v))
fig.suptitle('路线图第1步：数据覆盖、价值判断与策略转化');fig.savefig(p/'第一步诊断总览.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,key,title in [(axes[0],'mean_value_abs_difference','几何转方后价值输出的平均绝对变化'),(axes[1],'top1_changed','几何转方后首选变化数／64')]:
    vals=[identity['results'][m]['used_fresh_development']['overall'][key] for m in names];ax.bar(labels,vals,color=['#888888','#4477aa','#dd8844']);ax.set_title(title)
    for i,v in enumerate(vals):ax.text(i,v,f'{v:.3f}' if key.startswith('mean') else str(v),ha='center',va='bottom')
fig.suptitle('身份敏感性诊断：严格规则固定P1先手，几何转方不是完整博弈等价');fig.savefig(p/'身份敏感性.png',dpi=180);plt.close(fig)
with (p/'诊断图表数据.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['指标','数据','模型','数值'],lineterminator='\n');w.writeheader();w.writerows(export)
