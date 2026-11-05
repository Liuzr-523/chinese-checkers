"""48局数据试点中文图；仅读取已完成的JSON。"""
import json,sys,csv
from pathlib import Path
from plot_style_zh import configure
configure('路线图第二步专家数据训练开发真实胜负和棋限步有效价值监督终局前样本过滤')
import numpy as np
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);result=json.loads((p/'result.json').read_text());games=[json.loads(l) for l in (p/'games.jsonl').read_text().splitlines()];rows=json.loads((p/'sample_metadata.json').read_text());fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');names=['train','development'];labels=['训练集','开发集'];data=[]
bottom=np.zeros(2)
for label,color,fn in [('P1胜','#df6673',lambda g:g['winner']==1),('P2胜','#26a6b5',lambda g:g['winner']==2),('真实和棋','#999999',lambda g:g['winner']==0 and g['reason']!='max_ply'),('限步未完','#e5b44d',lambda g:g['reason']=='max_ply')]:
    vals=np.array([sum(fn(g) for g in games if g['split']==n) for n in names]);axes[0,0].bar(labels,vals,bottom=bottom,label=label,color=color);bottom+=vals
axes[0,0].set(title='完整48局：自然终局结果',ylabel='对局数');axes[0,0].legend(fontsize=8)
for off,key,label,color in [(-.18,'raw','原始价值记录','#aaaaaa'),(.18,'eligible','过滤后有效价值','#4477aa')]:
    vals=[sum(r['split']==n and (r['terminal_reason']!='max_ply' if key=='raw' else r['value_weight']>0) for r in rows) for n in names];axes[0,1].bar(np.arange(2)+off,vals,width=.35,label=label,color=color)
    for i,v in enumerate(vals):axes[0,1].text(i+off,v,str(v),ha='center',va='bottom')
axes[0,1].set(xticks=[0,1],xticklabels=labels,title='历史及跨组状态过滤前后',ylabel='价值监督条数');axes[0,1].legend(fontsize=8)
for off,n,label,color in [(-.18,'train','训练集','#4477aa'),(.18,'development','开发集','#dd8844')]:
    vals=[sum(r['split']==n and r['player']==pl and r['value_weight']>0 for r in rows) for pl in (1,2)];axes[1,0].bar(np.arange(2)+off,vals,width=.35,label=label,color=color)
axes[1,0].set(xticks=[0,1],xticklabels=['P1','P2'],title='有效价值样本的双方覆盖',ylabel='样本条数');axes[1,0].legend()
vals=[result['summary'][n]['endgame16_samples'] for n in names];axes[1,1].bar(labels,vals,color=['#4477aa','#dd8844']);axes[1,1].set(title='过滤后真实终局前16手覆盖',ylabel='有效价值样本条数')
for i,v in enumerate(vals):axes[1,1].text(i,v,str(v),ha='center',va='bottom')
fig.suptitle('路线图第2步：48局固定专家数据试点（未训练）');fig.savefig(p/'48局数据质量总览.png',dpi=170);plt.close(fig)
for n in names:
    v=result['summary'][n];data.append(dict(划分='训练集' if n=='train' else '开发集',完整对局=v['games'],真实终局=v['terminal_games'],真实终局开局组=v['terminal_groups'],有效价值样本=v['effective_value_samples'],终局前16手样本=v['endgame16_samples']))
with (p/'数据质量.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
