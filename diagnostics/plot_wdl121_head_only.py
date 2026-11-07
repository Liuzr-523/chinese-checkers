"""Render the frozen evaluation and continuation summaries."""
import json
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diagnostics.plot_style_zh import configure
configure('残局中局按整盘平均胜负预测误差越低越好父模型阵营先验最终候选实际胜局比例完整续弈仅供诊断搜索')
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
a = out/'artifacts'
audit = json.loads((a/'independent_audit.json').read_text())
probe = json.loads((a/'probe_summary.json').read_text())
fig, axes = plt.subplots(1,2,figsize=(10.6,4.4))
metrics = audit['decisive_brier']
for offset,name,label,color in [(-.24,'parent','父模型','#a9bac7'),
                               (0,'prior','阵营先验','#daa43d'),
                               (.24,'candidate','候选','#317f9e')]:
    axes[0].bar(np.arange(2)+offset,[metrics[stage][name] for stage in ('late','middle')],
                width=.23,label=label,color=color)
axes[0].set_xticks(range(2),['残局末16手','中局'])
axes[0].set_ylabel('胜负预测误差（按盘平均）')
axes[0].set_title('留出对局 · 越低越好')
axes[0].set_ylim(0,max(v for row in metrics.values() for v in row.values())*1.18)
axes[0].legend(loc='upper center',ncol=3,fontsize=9)
axes[1].plot([0,1],[0,1],ls='--',color='#a0aab0')
for name,label,color in [('parent','父模型','#9bacb8'),('candidate','候选','#317f9e')]:
    roots = [r for r in probe['roots'] if r['arms'][0]['n']]
    axes[1].scatter([r['predictions'][name]['raw_wdl'][0] for r in roots],
                    [r['arms'][0]['empirical_win'] for r in roots],
                    label=label,color=color,alpha=.8)
axes[1].set(xlim=(0,1),ylim=(0,1),xlabel='模型预计胜率',ylabel='实际续弈胜局比例',
            title=f"残局复核 · {len(probe['roots'])}根，每根8次")
axes[1].legend()
fig.tight_layout()
fig.savefig(out/'残局胜率复核.png',dpi=160)
