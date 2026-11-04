import json,sys
from pathlib import Path
from plot_style_zh import configure
configure()
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);nn=json.loads((p/'nn_result.json').read_text());arena=json.loads((p/'arena_result.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
labels=['原冠军','普通决策对照','开局库模仿'];v=[nn['baseline']['book_validation']['top1_in_book']]+[nn['final'][a]['book_validation']['top1_in_book'] for a in ('prior_decisions','opening_book')];bars=axes[0].bar(labels,v,color=['#aaaaaa','#4477aa','#228833']);axes[0].bar_label(bars,labels=[f'{x:.1%}' for x in v]);axes[0].set(title='关闭开局库：原始网络留出局面命中率',ylabel='首选属于库推荐集合的比例',ylim=(0,1))
s=arena['branches'];bars=axes[1].bar(['胜','负','和'],[s['wins'],s['losses'],s['draws']],color=['#228833','#cc6677','#aaaaaa']);axes[1].bar_label(bars);axes[1].set(title='同一个v3：带库对无库，20分支交换先后手',ylabel='对局数',ylim=(0,40));fig.suptitle('开局库实战收益与原始NN学习效果分别检查')
fig.savefig(p/'开局库与NN效果.png',dpi=170);fig.savefig(p/'开局库与NN效果.svg');plt.close(fig);svg=p/'开局库与NN效果.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
