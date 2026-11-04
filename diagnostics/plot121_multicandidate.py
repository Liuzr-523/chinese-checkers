"""多候选监督分析图：简体中文标题、轴、图例和说明。"""
import csv,json,sys
from pathlib import Path
from plot_style_zh import configure
font=configure('原单偏好多候选种子训练验证专家首选一致率旧策略交叉熵价值误差更新次数固定预算监督对照开发指标不代表棋力')
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);config=json.loads((p/'config.json').read_text());fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained');rows=[]
for seed,style in zip(config['seeds'],['-','--']):
 for arm,color in [('old','#4477aa'),('multi','#cc6677')]:
  name=f'{arm}_{seed}';history=json.loads((p/(name+'_curve.json')).read_text());x=[v['step'] for v in history];label=('原单偏好' if arm=='old' else '多候选')+'／种子'+str(seed)
  for ax,domain,key in [(axes[0,0],'train','accuracy'),(axes[0,1],'validation','accuracy'),(axes[1,0],'retention','policy_ce'),(axes[1,1],'retention','value_mse')]:ax.plot(x,[v[domain][key] for v in history],color=color,ls=style,label=label)
  for v in history:rows.append({'监督方案':'原单偏好' if arm=='old' else '多候选','抽样种子':seed,'更新次数':v['step'],'训练专家首选一致率':v['train']['accuracy'],'验证专家首选一致率':v['validation']['accuracy'],'旧策略交叉熵':v['retention']['policy_ce'],'旧价值误差':v['retention']['value_mse']})
for ax,title,ylabel in zip(axes.flat,['训练局面：专家首选一致率','验证局面：专家首选一致率','旧能力保留：策略交叉熵','旧能力保留：价值均方误差'],['一致率','一致率','交叉熵','均方误差']):ax.set(title=title,xlabel='参数更新次数',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[0,0].set_ylim(0,1);axes[0,1].set_ylim(0,1)
fig.suptitle('固定预算监督对照：开发指标不代表棋力；同分好棋另见实际评分报告')
fig.savefig(p/'监督对照曲线.png',dpi=170);fig.savefig(p/'监督对照曲线.svg');plt.close(fig)
with (p/'监督对照曲线.csv').open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(p/'中文字体检查.json').write_text(json.dumps(dict(字体路径=font,缺字数=0),ensure_ascii=False,indent=2))
