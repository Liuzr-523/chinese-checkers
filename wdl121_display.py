"""独立后台WDL分析器；不改变棋手或冠军，不把旧标量伪装成胜率。"""
import hashlib,json,math,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parent
DEFAULT_POINTER=ROOT/'nn_checkpoints/experiments/wdl121_loop_20261107/display_model.json'

def position_key(game,generation):
    history=tuple(sorted((tuple(b),p,n) for (b,p),n in game.repetitions.items() if n))
    return generation,tuple(game.board),game.player,game.ply,history

def format_probabilities(probabilities,player):
    probs=[float(x) for x in probabilities]
    if len(probs)!=3 or any(not math.isfinite(x) or x<0 or x>1 for x in probs) or abs(sum(probs)-1)>1e-4:raise ValueError('胜和负概率无效')
    win,draw,loss=probs
    p1,p2=(win,loss) if player==1 else (loss,win)
    return f'实验候选·模型估计·未校准：P1胜 {p1:.1%}  和 {draw:.1%}  P2胜 {p2:.1%}'

def terminal_text(game):
    winner,reason=game.result()
    if winner is None:return None
    if reason=='max_ply':return '达到限步，结果截断；不作真实和棋胜率标签'
    if winner==0:return '规则确定结果：和棋 100%'
    return f'规则确定结果：P{winner}胜 100%'

def top_candidates(info,limit=5):
    if not 3<=limit<=5:raise ValueError('候选数须为3至5')
    paths=info['paths'];visits=info['visits'];values=info['child_wdl']
    if len(paths)!=len(visits) or len(paths)!=len(values):raise ValueError('候选统计长度不符')
    result=[]
    for rank,index in enumerate(sorted(range(len(paths)),key=lambda i:(-visits[i],i))[:limit],1):
        value=values[index] if visits[index]>0 else None
        if value is not None:
            # Reuse validation only; statistics are already root-player perspective.
            format_probabilities(value,1)
            value=[float(x) for x in value]
        estimated=int(info.get('child_wdl_visits',visits)[index])
        result.append(dict(rank=rank,path=list(paths[index]),visits=int(visits[index]),wdl=value,estimated_visits=estimated,partial=estimated<visits[index]))
    return result

class ModelPredictor:
    """只在分析线程中加载自己的模型实例，绝不共享棋手模型。"""
    def __init__(self,pointer=DEFAULT_POINTER):self.pointer=Path(pointer);self.fingerprint=None;self.model=None
    def __call__(self,board,player,ply,repetitions):
        explicit=os.environ.get('CC121_WDL_MODEL')
        if explicit:
            path=Path(explicit).expanduser().resolve();expected=None
        else:
            if not self.pointer.exists():return '胜率模型尚未生成'
            record=json.loads(self.pointer.read_text());path=Path(record['checkpoint']).expanduser()
            if not path.is_absolute():raise ValueError('分析模型指针须为绝对路径')
            expected=record['sha256']
        if not path.is_file():return '胜率模型尚未生成'
        stat=path.stat();fingerprint=(str(path),stat.st_mtime_ns,stat.st_size,expected)
        if self.fingerprint!=fingerprint:
            if expected and hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('分析模型哈希不符')
            from wdl121 import WDLNet121
            model=WDLNet121.load(path);model.eval();model.requires_grad_(False)
            self.model=model;self.fingerprint=fingerprint
        import torch
        from az121 import encode,search
        with torch.inference_mode():
            x=torch.from_numpy(encode(board,player,ply,repetitions,canonical=self.model.canonical)).unsqueeze(0)
            probs=self.model.predict_wdl(x)[0].tolist()
        path,info=search(self.model,board,player,ply,repetitions,simulations=128,wdl_stats=True)
        limit=int(os.environ.get('CC121_WDL_TOP_K','5'))
        return dict(text=format_probabilities(probs,player),candidates=top_candidates(info,limit),player=player,simulations=info['nodes'])

class WDLAnalysis:
    def __init__(self,predictor=None,executor=None):
        self.predictor=predictor or ModelPredictor();self.pool=executor or ThreadPoolExecutor(max_workers=1,thread_name_prefix='胜率分析')
        self.future=None;self.pending_key=None;self.current_key=None;self.completed_key=None;self.text='胜率模型尚未生成';self.candidates=[]
    def update(self,game,generation,animation=False):
        key=position_key(game,generation)
        if key!=self.current_key:self.current_key=key;self.text='模型估计更新中…';self.completed_key=None;self.candidates=[]
        if self.future is not None and self.future.done():
            try:value=self.future.result()
            except Exception as exc:value='胜率分析不可用：'+str(exc)[:45]
            if self.pending_key==key:
                if isinstance(value,dict):self.text=value['text'];self.candidates=value['candidates']
                else:self.text=value;self.candidates=[]
                self.completed_key=key
            self.future=None;self.pending_key=None
        known=terminal_text(game)
        if known is not None:self.candidates=[];return known
        if animation:self.candidates=[];self.completed_key=None;return '走棋动画中，落子后更新胜率'
        if self.completed_key!=key and self.future is None:
            self.pending_key=key
            self.future=self.pool.submit(self.predictor,game.board.copy(),game.player,game.ply,game.repetitions.copy())
        return self.text
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)
