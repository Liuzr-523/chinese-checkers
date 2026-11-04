import json,sys
from pathlib import Path
from plot_style_zh import configure
configure('优化器历史隔离单步探针难例平均间隔变化正值改善继承原动量零当前梯度清一阶动量全新无动量梯度下降不能据此推断长期棋力')
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);r=json.loads((p/'动量补充审计/result.json').read_text());labels=['原优化器\n当前梯度置零','原优化器\n完整目标','仅清一阶动量\n完整目标','全新优化器\n完整目标','无动量梯度下降\n完整目标'];v=[r['zero_gradient_inherited_delta']]+[r['summary'][k]['full']['mean_delta'] for k in ('inherited','reset_first_moment','fresh_adam','sgd')]
fig,ax=plt.subplots(figsize=(11,5),layout='constrained');bars=ax.bar(labels,v,color=['#cc6677' if x<0 else '#4477aa' for x in v]);ax.axhline(0,color='gray');ax.set(title='相同最终模型的单步探针：隔离优化器历史',ylabel='七个难例平均间隔变化（正值改善）');ax.bar_label(bars,labels=[f'{x:.6f}' for x in v],padding=4);ax.grid(axis='y',alpha=.2);fig.suptitle('局部机制诊断，不能据此推断重置优化器能提高长期棋力',fontsize=11)
fig.savefig(p/'动量隔离诊断.png',dpi=170);fig.savefig(p/'动量隔离诊断.svg');plt.close(fig)
