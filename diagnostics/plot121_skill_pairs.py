"""已有正反棋例的简体中文静态示意；不查询模型或搜索。"""
import json,sys
from pathlib import Path
from plot_style_zh import configure
configure('落子次序专项历史正反例应先解放后排应先兑现前锋长跳建议方案原走法固定续弈胜负棋子下轮接力入口被占对手先后不等于普适棋理')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,FancyArrowPatch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rules121 as r
p=Path(sys.argv[1]);data=json.loads((p/'result.json').read_text());byid={x['seed_id']:x for x in data['details']}
fig,axes=plt.subplots(1,2,figsize=(14,9));fig.subplots_adjust(top=.87,bottom=.25,wspace=.12)
for ax,key,title in zip(axes,(13,40),('正例：先解放后排','反例：先兑现前锋长跳')):
    row=byid[key]
    def xy(i):q,y=r.COORDS[i];return q+y/2,-y
    for start,ns in r.NEIGHBORS.items():
        for end in ns:
            if end>start:ax.plot(*zip(xy(start),xy(end)),color='#d4dadd',lw=.6,zorder=0)
    for i,v in enumerate(row['board']):
        color=('#e45b65' if v==1 else '#26a5b8') if v else ('#f7dce0' if i in r.HOME[1] else '#d6eef2' if i in r.HOME[2] else '#f3f1e9')
        ax.add_patch(Circle(xy(i),.24,facecolor=color,edgecolor='#a4aeb4',lw=.8,zorder=1))
    lines=[]
    for role,color,label in [('expert','#15824c','建议方案'),('played','#df8d0a','原走法')]:
        arm=row['branches'][role];path=arm['path']
        for a,b in zip(path,path[1:]):ax.add_patch(FancyArrowPatch(xy(a),xy(b),arrowstyle='-|>',mutation_scale=13,lw=2,color=color,zorder=3,shrinkA=3,shrinkB=3))
        for i in (path[0],path[-1]):
            x,y=xy(i);ax.text(x+.28,y+.1,str(i),fontsize=9,color=color,fontweight='bold',bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1),zorder=4)
        win='胜' if arm['winner']==row['player'] else '和' if arm['winner']==0 else '负';lines.append((f"{label}：{'→'.join(map(str,path))}；续弈{win}",color))
    ax.set(title=f"{title}（P{row['player']}，已走{row['ply']}手）",xlim=(-7,7),ylim=(-8.7,8.7));ax.set_aspect('equal');ax.axis('off')
    for j,(line,color) in enumerate(lines):ax.text(.02,-.045-j*.05,line,transform=ax.transAxes,color=color,fontsize=11)
fig.suptitle('落子次序专项：同样推进落后子，正反情形都要学',fontsize=18)
fig.text(.08,.035,'左图：8→16后，下轮可跳16→39→58；原走法分支中对手随后占据16。\n右图：先走前锋长跳、下一轮再搬后排，固定续弈结果反而更好。\n结果来自已有的固定配置续弈；用于解释案例，不作为全新棋力验收。',fontsize=11,linespacing=1.7)
fig.savefig(p/'落子次序正反例.png',dpi=160);plt.close(fig)
