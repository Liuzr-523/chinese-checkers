"""strict121 CNN的独立胜/和/负头；不注册为冠军，不把限步当作和棋。"""
import copy,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from az121 import Net121,ENCODING,CANONICAL_ENCODING
import rules121 as r

FORMAT='wdl121-v1'
CLASSES=('win','draw','loss')

class WDLNet121(Net121):
    def __init__(self,width=32,value_head='pooled',canonical=False,metadata=None):
        super().__init__(width,value_head,canonical)
        self.wdl=nn.Sequential(nn.Linear(width,32),nn.ReLU(),nn.Linear(32,3))
        nn.init.zeros_(self.wdl[-1].weight);nn.init.zeros_(self.wdl[-1].bias)
        self.value.requires_grad_(False)
        self.metadata=copy.deepcopy(metadata or {})

    @classmethod
    def from_v3(cls,model,parent_sha256=None,iteration=0):
        if isinstance(model,cls):raise ValueError('from_v3只接受原双头CNN')
        result=cls(model.width,model.value_head,model.canonical,dict(parent_sha256=parent_sha256,iteration=iteration))
        loaded=result.load_state_dict(model.state_dict(),strict=False)
        if loaded.unexpected_keys or any(not name.startswith('wdl.') for name in loaded.missing_keys):raise ValueError('父CNN结构不兼容')
        result.train(model.training)
        return result

    def forward_wdl(self,x):
        h=F.relu(self.stem(x))
        for block in self.blocks:h=F.relu(h+block(h))
        cells=h[:,:,self.rows,self.cols].transpose(1,2);bias=self.bias(cells)
        logits=torch.bmm(self.src(cells),self.dst(cells).transpose(1,2))/4
        logits=logits+bias[:,:,0].unsqueeze(2)+bias[:,:,1].unsqueeze(1)
        return logits.flatten(1),self.wdl(cells.mean(1))

    def predict_wdl(self,x):return self.forward_wdl(x)[1].softmax(-1)

    def forward(self,x):
        logits,wdl=self.forward_wdl(x);prob=wdl.softmax(-1)
        return logits,prob[:,0]-prob[:,2]

    def save(self,path,metadata=None):
        meta=copy.deepcopy(self.metadata)
        if metadata:meta.update(metadata)
        torch.save(dict(format=FORMAT,encoding=self.encoding,profile=r.PROFILE,width=self.width,value_head=self.value_head,classes=CLASSES,state_dict=self.state_dict(),metadata=meta),path)

    @classmethod
    def load(cls,path):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data.get('format')!=FORMAT or data.get('profile')!=r.PROFILE or data.get('encoding') not in (ENCODING,CANONICAL_ENCODING) or tuple(data.get('classes',()))!=CLASSES:raise ValueError('不是兼容的strict121 WDL checkpoint')
        model=cls(data['width'],data['value_head'],data['encoding']==CANONICAL_ENCODING,data.get('metadata'))
        model.load_state_dict(data['state_dict']);return model.eval()


def terminal_class(winner,player,reason):
    if player not in (1,2):raise ValueError('走棋方必须为1或2')
    if reason=='max_ply' or winner is None:return None
    if winner not in (0,1,2):raise ValueError('非法终局winner')
    draws={'equal_turn_goal_draw','repetition','no_legal_moves'};wins={'all_pieces_in_goal','anti_block_goal_full'}
    if winner==0:
        if reason not in draws:raise ValueError('未经确认的真实和棋原因')
        return 1
    if reason not in wins:raise ValueError('未经确认的胜负终局原因')
    return 0 if winner==player else 2


def sample_class(sample):
    # 如带原终局字段，优先核实真实终局；截断不能由手工class复活。
    if 'winner' in sample or 'reason' in sample:
        label=terminal_class(sample.get('winner'),sample['player'],sample.get('reason',''))
        if label is not None and 'wdl_class' in sample and sample['wdl_class']!=label:raise ValueError('WDL标签与真实终局不一致')
        return label
    label=sample.get('wdl_class')
    if label is not None and (type(label) is not int or label not in (0,1,2)):raise ValueError('WDL class必须为0/1/2或None')
    return label


def training_batch(samples):
    if not samples:raise ValueError('空训练batch')
    x=torch.from_numpy(np.stack([s['x'] for s in samples])).float();width=max(len(s['actions']) for s in samples)
    ids=torch.zeros((len(samples),width),dtype=torch.long);mask=torch.zeros_like(ids,dtype=torch.bool);target=torch.zeros((len(samples),width));pw=[];vw=[];labels=[]
    for i,s in enumerate(samples):
        actions=s['actions'];n=len(actions)
        if not n or len(set(actions))!=n or any(type(a) is not int or not 0<=a<14641 for a in actions):raise ValueError('无效合法动作集合')
        ids[i,:n]=torch.tensor(actions);mask[i,:n]=True
        if 'policy' in s:
            p=torch.as_tensor(s['policy'],dtype=torch.float32)
            if p.shape!=(n,) or not torch.isfinite(p).all() or (p<0).any() or not torch.isclose(p.sum(),torch.tensor(1.),atol=1e-5):raise ValueError('无效soft policy目标')
            target[i,:n]=p
        else:
            index=s['target']
            if type(index) is not int or not 0<=index<n:raise ValueError('无效策略target')
            target[i,index]=1.
        label=sample_class(s);labels.append(0 if label is None else label)
        pw.append(float(s.get('policy_weight',1.)));vw.append(float(s.get('wdl_weight',s.get('value_weight',1.))) if label is not None else 0.)
    weights=torch.tensor([pw,vw],dtype=torch.float32)
    if not torch.isfinite(weights).all() or (weights<0).any():raise ValueError('无效损失权重')
    return dict(x=x,ids=ids,mask=mask,policy=target,labels=torch.tensor(labels),policy_weight=weights[0],wdl_weight=weights[1])


def training_loss(model,samples,reference=None,policy_weight=1.,wdl_weight=1.,kl_weight=0.):
    if any(not math.isfinite(v) or v<0 for v in (policy_weight,wdl_weight,kl_weight)):raise ValueError('无效loss系数')
    b=training_batch(samples);logits,wdl=model.forward_wdl(b['x']);lp=logits.gather(1,b['ids']).masked_fill(~b['mask'],-1e9).log_softmax(1)
    pw,vw=b['policy_weight'],b['wdl_weight'];pce=(-(lp*b['policy']).sum(1)*pw).sum()/pw.sum().clamp_min(1e-12)
    vce=(F.cross_entropy(wdl,b['labels'],reduction='none')*vw).sum()/vw.sum().clamp_min(1e-12)
    kl=logits.sum()*0.
    if kl_weight and reference is None:raise ValueError('旧policy KL需要reference')
    if reference is not None:
        with torch.no_grad():old,_=reference(b['x']);oldlp=old.gather(1,b['ids']).masked_fill(~b['mask'],-1e9).log_softmax(1)
        kl=((oldlp.exp()*(oldlp-lp)).sum(1)*pw).sum()/pw.sum().clamp_min(1e-12)
    return dict(total=policy_weight*pce+wdl_weight*vce+kl_weight*kl,policy_ce=pce,wdl_ce=vce,policy_kl=kl,policy_n=int((pw>0).sum()),wdl_n=int((vw>0).sum()))
