"""落子次序新根专项的中文图表；只读取已保存工件。"""
import json,sys,csv
from pathlib import Path
from plot_style_zh import configure
configure('落子次序专项新根数据覆盖训练开发后排优前锋优平衡准确率原模型候选模型误差变化保持上限训练更新次数成对交叉熵未训练未通过')
import numpy as np
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);r=json.loads((p/'result.json').read_text());d=json.loads((p/'collection/dataset.json').read_text());fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for i,split in enumerate(('train','development')):
    vals=[sum(s['split']==split and s['label']==label for s in d['rows']) for label in (1,-1,0)];axes[0,0].bar(np.arange(3)+(i-.5)*.3,vals,width=.3,label=('训练' if i==0 else '开发'))
axes[0,0].set(xticks=range(3),xticklabels=['后排优','前锋优','不适用'],ylabel='根局面组数',title='冻结后数据覆盖');axes[0,0].legend()
if r.get('training_started'):
    before=r['before'];after=r['after'];c=json.loads((p/'curve.json').read_text());axes[0,1].plot([s['step'] for s in c],[s['train']['pair_ce'] for s in c],color='#4477aa');axes[0,1].set(xlabel='更新次数',ylabel='成对交叉熵',title='仅训练曲线');axes[0,1].grid(alpha=.2)
    for off,name,label,color in [(-.17,'before','原模型','#888888'),(.17,'after','候选模型','#dd8844')]:
        z=r[name]['order'];v=[z['directions'][q]['accuracy'] for q in ('rear','front')]+[z['balanced_accuracy']];axes[1,0].bar(np.arange(3)+off,v,width=.32,label=label,color=color)
    axes[1,0].set(xticks=range(3),xticklabels=['后排优','前锋优','平衡准确率'],ylim=(0,1.05),title='新开发组：两方向排序');axes[1,0].legend()
    v=[after['retention']['policy_ce']-before['retention']['policy_ce'],after['retention']['value_mse']-before['retention']['value_mse'],after['normal']['policy_ce']-before['normal']['policy_ce']];axes[1,1].bar(range(3),v,color='#4477aa');axes[1,1].axhline(.05,ls='--',color='#aa4444',label='保持上限');axes[1,1].axhline(0,color='#aaaaaa',lw=.6);axes[1,1].set(xticks=range(3),xticklabels=['旧策略交叉熵','旧价值误差','普通策略交叉熵'],ylabel='候选减原模型',title='旧能力变化');axes[1,1].legend()
    with (p/'训练曲线.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['更新次数','训练成对交叉熵','训练平衡准确率']);w.writerows((s['step'],s['train']['pair_ce'],s['train']['balanced_accuracy']) for s in c)
    verdict='通过专项开发门槛' if r['development_pass'] else '未通过专项开发门槛'
else:
    review=json.loads((p/'data_review.json').read_text()) if (p/'data_review.json').exists() else None
    if review:
        t=review['rear_template_audit']['summary'];n=sum(t[q]['all']['static_forward_multijump'] for q in ('train','development'));axes[0,1].bar(['具备静态接力潜力','未观察到'],[n,48-n],color=['#4477aa','#bbbbbb']);axes[0,1].set(title='后排减距不等于接力入口',ylabel='根局面组数')
    else:axes[0,1].axis('off')
    probe=p/'historical_teacher_probe/result.json'
    if probe.exists():
        t=json.loads(probe.read_text())['by_depth'];x=np.arange(3)
        for off,key,den,label,color in [(-.17,'rear_winning_agreement',3,'后排更优的3例','#4477aa'),(.17,'front_preferred_agreement',4,'前锋更优的4例','#dd8844')]:
            axes[1,0].bar(x+off,[t[str(depth)][key]/den for depth in (4,5,6)],width=.32,label=label,color=color)
        axes[1,0].set(xticks=x,xticklabels=['深度4','深度5','深度6'],ylim=(0,1.1),title='历史7例：教师与续弈方向相符比例');axes[1,0].legend(fontsize=9)
    else:axes[1,0].axis('off')
    axes[1,1].axis('off');axes[1,1].text(.05,.65,'开发后排优：1例，低于2例下限\n训练：未启动\n冠军：保留原模型\n先修正任务模板与标签语义。',fontsize=15,linespacing=1.8,transform=axes[1,1].transAxes);verdict='数据覆盖未通过'
fig.suptitle('路线图第3步：落子次序专项——'+verdict+'（未晋升）');fig.savefig(p/'落子次序专项结果.png',dpi=160);plt.close(fig)
