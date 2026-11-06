"""从归档汇总绘制中文搜索诊断图，不重新运行搜索。"""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diagnostics.plot_style_zh import configure
configure('搜索诊断根动作覆盖率模拟次数相对纯策略改变选择局面数价值贡献开启关闭已知偏好与实际选择稳定偏好方稳定非偏好方候选外未知不等于棋力提升')
import matplotlib.pyplot as plt

folder=Path(sys.argv[1]);result=json.loads((folder/'artifacts/result.json').read_text())
rows=[x for x in result['cases'] if x['split']=='all'];ns=[32,128,512]
fig,axes=plt.subplots(1,3,figsize=(13,4.5))
for scale,color,label in [(1.,'#317e9e','价值贡献开启'),(0.,'#d9a047','价值贡献关闭')]:
    xs=[next(x for x in rows if x['simulations']==n and x['value_scale']==scale) for n in ns]
    axes[0].plot(range(3),[100*x['mean_coverage'] for x in xs],marker='o',color=color,label=label)
    axes[1].plot(range(3),[x['changed_from_policy'] for x in xs],marker='o',color=color,label=label)
for ax in axes[:2]:ax.set_xticks(range(3),ns);ax.set_xlabel('模拟次数');ax.legend(fontsize=8)
axes[0].set_title('根动作覆盖率');axes[0].set_ylabel('已访问动作占比（%）');axes[0].set_ylim(0,60)
axes[1].set_title('相对纯策略改变选择');axes[1].set_ylabel('局面数 / 48');axes[1].set_ylim(0,8)
bottom=0
for value,color,label in [(3,'#317e9e','稳定偏好方'),(4,'#c46f61','稳定非偏好方'),(4,'#bdc6cd','候选外未知')]:
    axes[2].bar([0,1],[value,value],bottom=bottom,color=color,label=label);bottom+=value
axes[2].set_xticks([0,1],['纯策略','512次搜索']);axes[2].set_ylabel('局面数 / 11');axes[2].set_title('已知偏好与实际选择');axes[2].set_ylim(0,15);axes[2].legend(fontsize=8)
fig.suptitle('固定v3搜索诊断：更多模拟尚未纠正选择');fig.tight_layout()
fig.savefig(folder/'搜索诊断.png',dpi=160)
