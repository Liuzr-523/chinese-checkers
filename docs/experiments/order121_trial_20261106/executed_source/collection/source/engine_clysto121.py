"""Clysto-evaluation-derived expert candidate, not the original 81-cell engine."""
import math
from engine_c121 import CSearch121

class ClystoDerived121(CSearch121):
    name='Clysto评估派生121（候选，非原版）'
    def __init__(self,seconds=.04,max_depth=10,max_ply=240):
        if not math.isfinite(seconds) or seconds<=0 or not 1<=max_depth<=62 or max_ply<0:
            raise ValueError('invalid search limits')
        import clysto121_native
        self.native=clysto121_native
        self.seconds,self.max_depth,self.max_ply=seconds,max_depth,max_ply
    def think(self,*args,**kwargs):
        path,info=super().think(*args,**kwargs)
        info['eval_version']='clysto-table-clamped-wings-121-v1'
        info['original_clysto']=False
        return path,info
