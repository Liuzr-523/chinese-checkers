"""从归档概率与损失绘制中文图，不重新训练或搜索。"""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diagnostics.plot_style_zh import configure
configure('胜和负首轮闭环开发概率误差初始均匀训练先验最终候选越低越好按整盘平均训练更新次数损失策略真实终局置信度实际胜率未校准对局已完成')
import matplotlib.pyplot as plt

folder=Path(sys.argv[1]);a=folder/'artifacts'
baseline=json.loads((a/'baseline.json').read_text());final=json.loads((a/'final.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(13,4.5))
iteration=json.loads((a/'config.json').read_text()).get('iteration',1)
names=['初始均匀' if iteration==1 else '父模型','训练先验','最终候选']
for index,key in enumerate(('brier3','nll')):
    xs=[baseline['wdl']['game_macro'][key],baseline['prior_metrics']['game_macro'][key],final['wdl']['game_macro'][key]]
    axes[index].bar(range(3),xs,color=['#bac8d2','#dba646','#317f9e']);axes[index].set_xticks(range(3),names)
    axes[index].set_title(('三类概率误差' if key=='brier3' else '真实结果负对数损失')+'（越低越好）')
    axes[index].set_ylabel('按整盘平均')
records=final['wdl']['records'];bins=[]
for i in range(10):
    rows=[r for r in records if min(int(r['probabilities'][0]*10),9)==i]
    if rows:bins.append((np.mean([r['probabilities'][0] for r in rows]),np.mean([r['label']==0 for r in rows]),len(rows)))
axes[2].plot([0,1],[0,1],ls='--',color='#a0aab0')
if bins:axes[2].scatter([b[0] for b in bins],[b[1] for b in bins],s=[20+5*np.sqrt(b[2]) for b in bins],color='#317f9e')
axes[2].set(xlim=(0,1),ylim=(0,1),xlabel='候选预计胜率',ylabel='实际胜局标签比例',title='胜率分箱诊断（未校准）')
fig.suptitle(f'胜／和／负闭环第{iteration}轮 · 留出完整对局检验');fig.tight_layout();fig.savefig(folder/'概率验证.png',dpi=160)
losses=json.loads((a/'losses.json').read_text());fig,ax=plt.subplots(figsize=(8,4.5))
for key,label in [('wdl_ce','真实终局损失'),('policy_ce','策略损失'),('old_kl','旧策略保持损失')]:
    ax.plot([r['step'] for r in losses],[r[key] for r in losses],label=label)
ax.set(xlabel='训练更新次数',ylabel='损失',title=f'第{iteration}轮固定2000次更新（仅训练指标）');ax.legend();fig.tight_layout();fig.savefig(folder/'训练曲线.png',dpi=160)
