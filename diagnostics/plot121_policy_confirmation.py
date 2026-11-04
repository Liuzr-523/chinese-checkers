import json,sys
from pathlib import Path
from plot_style_zh import configure
configure()
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);r=json.loads((p/'result.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for ax,rows,title,total in [(axes[0],[('候选对v3',r['confirmation'])],'候选与现役冠军：120局',120),(axes[1],[('v3对C-PVS',r['fixed_c']['control']),('候选对C-PVS',r['fixed_c']['candidate'])],'相同开局对C-PVS：各40局',40)]:
 for y,(label,s) in enumerate(rows):
  left=0
  for key,name,color in [('wins','胜','#228833'),('draws','和','#aaaaaa'),('losses','负','#cc6677')]:
   v=s[key];ax.barh(y,v,left=left,color=color,label=name if y==0 else None)
   if v:ax.text(left+v/2,y,str(v),ha='center',va='center',color='white')
   left+=v
 ax.set(title=title,xlabel='对局数',yticks=range(len(rows)),yticklabels=[x[0] for x in rows],xlim=(0,total));ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3)
fig.suptitle('完整对局验收：双方交换先后手，固定模型与搜索预算')
fig.savefig(p/'完整对局验收.png',dpi=170);fig.savefig(p/'完整对局验收.svg');plt.close(fig)
