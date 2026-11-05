"""单向消除普通策略回放与当前专项梯度的负内积，不做范数限幅。"""
import torch
from cap_opening121_replay import policy_gradients,norm_of,dot_of


def projection_coefficient(dot,root_squared_norm,enabled=True):
    with torch.no_grad():
        if not bool(torch.isfinite(dot)) or not bool(torch.isfinite(root_squared_norm)) or float(root_squared_norm)<0:
            raise ValueError('投影内积/范数非法')
        if not enabled or float(root_squared_norm)==0 or float(dot)>=0:
            return torch.zeros_like(dot)
        value=(-dot/root_squared_norm).detach()
        if not bool(torch.isfinite(value)):raise ValueError('投影系数非有限')
        return value


def projected_objective(root,replay,aux,params,enabled=True):
    a=policy_gradients(root,params);b=policy_gradients(replay,params);squared=dot_of(a,a);dot=dot_of(a,b);coefficient=projection_coefficient(dot,squared,enabled)
    loss=(1+coefficient)*root+replay+aux
    return loss,a,b,coefficient
