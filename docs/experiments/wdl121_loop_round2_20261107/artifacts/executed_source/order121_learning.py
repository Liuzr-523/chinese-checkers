"""同一根局面的后排/前锋落子次序监督；不是全局胜负价值标签。"""
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
import rules121 as r
from az121 import encode,action_id
from curriculum121 import replay,batch


def configure(model):
    for name,p in model.named_parameters():p.requires_grad_(not name.startswith('value.'))
    return model


configure_policy_training=configure

def value_state_hash(model):
    h=hashlib.sha256()
    for name,v in model.state_dict().items():
        if name.startswith('value.'):
            h.update(name.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prepare_samples(samples,canonical=True):
    """在采样前预编码一次。允许开发根，但开发根不允许进入训练loss。"""
    prepared=[]
    for row in samples:
        g=replay(row['opening'])
        if g.board!=row['board'] or g.player!=row['player'] or g.ply!=row['ply']:raise ValueError('prefix does not reproduce root')
        for name in ('rear_path','front_path'):
            if not r.valid_path(g.board,g.player,row[name]):raise ValueError('illegal paired action')
        a,b=pair_ids(row,canonical)
        if a==b:raise ValueError('paired endpoints identify same policy action')
        prepared.append({**row,'_order_x':encode(g.board,g.player,g.ply,g.repetitions,canonical=canonical),'_order_canonical':bool(canonical)})
    return prepared


def order_batch(samples,canonical=True):
    if not samples:raise ValueError('empty root batch')
    if any('_order_x' not in s or s.get('_order_canonical')!=bool(canonical) for s in samples):samples=prepare_samples(samples,canonical)
    return torch.from_numpy(np.stack([s['_order_x'] for s in samples]))


def pair_ids(sample,canonical=True):
    player=sample['player']
    if player not in (1,2):raise ValueError('invalid player')
    paths=[sample[k] for k in ('rear_path','front_path')]
    if any(len(path)<2 or any(not isinstance(i,int) or not 0<=i<121 for i in path) for path in paths):raise ValueError('invalid action path')
    return tuple(action_id(path[0],path[-1],player,canonical) for path in paths)


def paired_policy_loss(logits,samples,canonical=True):
    """只平均train且eligible的±1标签；mask0无损失梯度，开发输入立即拒绝。

    同根lr-lf消去共同logit偏移及共同legal-softmax归一项；不假设模型
    对身份完全不敏感，也不把同根偏好差称作全局胜负监督。
    """
    if logits.ndim!=2 or len(logits)!=len(samples) or not samples:raise ValueError('matching nonempty logits/root batch required')
    terms=[]
    for i,row in enumerate(samples):
        if row['split']!='train':raise ValueError('development root cannot enter training loss')
        label=row['label']
        if label not in (-1,0,1):raise ValueError('invalid preference label')
        if not row['eligible'] and label!=0:raise ValueError('masked root resurrected by nonzero label')
        if label==0:continue
        a,b=pair_ids(row,canonical)
        if a==b or max(a,b)>=logits.shape[1]:raise ValueError('invalid paired policy indices')
        if not bool(torch.isfinite(logits[i,[a,b]]).all()):raise ValueError('nonfinite paired logits')
        terms.append(F.softplus(-float(label)*(logits[i,a]-logits[i,b])))
    return torch.stack(terms).mean() if terms else logits.sum()*0.


def retention_terms(model,reference,samples):
    """保留旧合法策略及旧value函数，不使用旧胜负作为本专项标签。

    value头参数冻结不妨碍此项梯度经过其运算流回共享CNN。
    """
    if not samples:raise ValueError('empty retention batch')
    x,ids,mask,*_=batch(samples);l,v=model(x);lp=l.gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)
    with torch.no_grad():rl,rv=reference(x);rp=rl.gather(1,ids).masked_fill(~mask,-1e9).log_softmax(1)
    return dict(policy_kl=(rp.exp()*(rp-lp)).sum(1).mean(),value_preservation=F.mse_loss(v,rv))
