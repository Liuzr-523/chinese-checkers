"""中文图：固定48根真实续弈的标签覆盖。"""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diagnostics.plot_style_zh import configure
configure('接力专项稳定偏好根数训练开发后排优前锋优无稳定偏好配置方向相反仅一配置有偏好两配置均无偏好覆盖门槛')
import matplotlib.pyplot as plt

folder=Path(sys.argv[1]);data=json.loads((folder/'resume_artifacts/label_diagnostics.json').read_text())['classification']
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
axes[0].bar([0,1],[5,3],label='后排优',color='#347c9b')
axes[0].bar([0,1],[2,1],bottom=[5,3],label='前锋优',color='#dfa54b')
axes[0].axhline(8,color='#b04444',ls='--',label='总数门槛：8根')
axes[0].set_xticks([0,1],['训练','开发']);axes[0].set_ylim(0,10)
axes[0].set_title('稳定偏好覆盖不足');axes[0].set_ylabel('根数');axes[0].legend()
names=['两配置均无偏好','仅一配置有偏好','两配置方向相反'];bottom=[0,0]
for name,color in zip(names,['#d7dfe5','#9caebb','#607d8b']):
    values=[data[s][name] for s in ('train','development')]
    axes[1].bar([0,1],values,bottom=bottom,label=name,color=color)
    bottom=[a+b for a,b in zip(bottom,values)]
axes[1].set_xticks([0,1],['训练','开发']);axes[1].set_ylabel('根数');axes[1].set_title('37根无稳定偏好的原因');axes[1].legend()
fig.suptitle('接力专项：192局完成，未达到训练门槛');fig.tight_layout()
fig.savefig(folder/'续弈标签覆盖.png',dpi=160)
