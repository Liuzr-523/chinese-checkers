import csv,json,sys
from pathlib import Path
from plot_style_zh import configure
configure('难例学习过程专家评分分差参数更新次数局部梯度普通回放专项平均余弦隔离单步探针完整目标仅专项仅普通策略回放仅价值与正则移除普通回放移除专项间隔变化正值改善')
import numpy as np
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);result=json.loads((p/'result.json').read_text());cases=json.loads((p/'cases.json').read_text());hard=[c for c in cases if c['decision_id'] in result['hard_ids']];steps=[r['step'] for r in hard[0]['trajectory']];fig,axes=plt.subplots(3,1,figsize=(12,12),layout='constrained')
a=np.array([[r['regret'] for r in c['trajectory']] for c in hard]);im=axes[0].imshow(a,aspect='auto',cmap='YlOrRd',vmin=0);axes[0].set_xticks(range(len(steps)),steps,rotation=45);axes[0].set_yticks(range(len(hard)),['局面'+str(c['decision_id']) for c in hard]);axes[0].set(title='七个最终难例的学习过程：0表示与专家最优分相同',xlabel='参数更新次数');fig.colorbar(im,ax=axes[0],label='与固定专家最优分的差距')
g=result['gradient'];x=[int(s) for s in g];axes[1].plot(x,[v['mean_cosine'] for v in g.values()],marker='o',label='专项与普通策略梯度平均余弦');axes[1].axhline(0,color='gray',lw=1);axes[1].set(title='五阶段固定八批梯度诊断：负值表示局部方向冲突',xlabel='参数更新次数',ylabel='平均余弦');axes[1].legend();axes[1].grid(alpha=.2)
keys=['full','root_only','replay_only','aux_only','without_replay','without_root'];labels=['完整目标','仅专项','仅普通策略回放','仅价值与正则','移除普通回放','移除专项'];values=[result['one_step'][k]['mean_margin_delta'] for k in keys];axes[2].bar(labels,values,color=['#4477aa' if v>=0 else '#cc6677' for v in values]);axes[2].axhline(0,color='gray',lw=1);axes[2].set(title='同模型与优化器起点的隔离单步探针：仅说明局部作用',ylabel='难例平均间隔变化（正值改善）');axes[2].tick_params(axis='x',rotation=15)
fig.suptitle('训练难例诊断：固定专家评分与局部探针均不等于实战棋力')
fig.savefig(p/'难例过程诊断.png',dpi=170);fig.savefig(p/'难例过程诊断.svg');plt.close(fig)
with (p/'难例过程数据.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['局面编号','开局组','更新次数','实际动作','专家评分差距']);w.writeheader()
 for c in cases:
  for r in c['trajectory']:w.writerow({'局面编号':c['decision_id'],'开局组':c['pair'],'更新次数':r['step'],'实际动作':r['action'],'专家评分差距':r['regret']})
