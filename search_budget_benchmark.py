"""Resumable fixed-network MCTS comparison using local inference only."""
import argparse
import fcntl
import hashlib
import json
import os
import pickle
import shutil
import time
import traceback
from pathlib import Path
import torch
from az_longrun import play, atomic_json, atomic_pickle, stamp
from nn_resnet import NetResNet

PAIRS = [(256,64), (512,64), (512,256)]


def schedule(openings):
    return [(high,low,1200000+opening,side)
            for opening in range(openings) for high,low in PAIRS for side in (1,2)]


def compatible_extension(old, new):
    return (new['openings'] >= old['openings'] and
            {k:v for k,v in old.items() if k != 'openings'} ==
            {k:v for k,v in new.items() if k != 'openings'})


def report(root, records, total, status):
    rows=[]
    for high,low in PAIRS:
        games=[g for g in records if g['comparison']==[high,low]]
        wins=sum(g['winner']==g['high_side'] for g in games)
        losses=sum(g['winner']==3-g['high_side'] for g in games)
        unfinished=sum(g['winner'] is None for g in games)
        seconds=[0.,0.];moves=[0,0]
        for g in games:
            for j,side in enumerate((g['high_side'],3-g['high_side'])):
                seconds[j]+=g['search_seconds'][side-1];moves[j]+=g['search_moves'][side-1]
        rows.append(dict(high=high,low=low,games=len(games),wins=wins,losses=losses,
                         unfinished=unfinished,seconds_per_move=[s/max(n,1) for s,n in zip(seconds,moves)]))
    atomic_json(root/'summary.json',dict(status=status,finished=len(records),total=total,comparisons=rows))
    lines=['# MCTS 搜索预算对照','',f'状态：{status}；已完成 {len(records)}/{total} 局。',
           '固定同一校准模型，折扣0.995；无开局库、无探索噪声、无重训。每个开局交换先后手。',
           '这是按搜索量比较，耗时不同；截断不算规则和棋。小样本结果不作晋升依据。','',
           '| 更多模拟 vs 更少模拟 | 胜 | 负 | 未结束 | 秒/手（多 / 少） |',
           '|---|---:|---:|---:|---|']
    for r in rows:
        a,b=r['seconds_per_move']
        lines.append(f"| {r['high']} vs {r['low']} | {r['wins']} | {r['losses']} | {r['unfinished']} | {a:.3f} / {b:.3f} |")
    lines+=['','每局棋谱和合法性校验保存在 games/；进度 status.json；事件 events.jsonl；标准输出 run.log。',
            '程序全部完成后自动退出，不调用 Codex 或其他远程接口。',
            '如需提前停止，在本目录创建 STOP 文件，程序完成当前局后停止；删除 STOP 后可用同一命令恢复。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--model',type=Path,default=Path('nn_checkpoints/experiments/longrun_20261022/bootstrap.pt'))
    ap.add_argument('--openings',type=int,default=8)
    ap.add_argument('--extend',action='store_true')
    ap.add_argument('--wait-lock',action='store_true')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if args.openings<1:ap.error('--openings must be positive')
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX if args.wait_lock else fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1)
    tasks=schedule(args.openings)
    state=dict(pid=os.getpid(),status='running',started=stamp(),total=len(tasks),finished=0)
    def update(**kw):
        state.update(kw,updated=stamp());atomic_json(root/'status.json',state)
    def event(kind,**kw):
        obj=dict(time=stamp(),event=kind,**kw)
        with (root/'events.jsonl').open('a') as f:f.write(json.dumps(obj)+'\n')
        print(json.dumps(obj),flush=True)
    try:
        config=dict(openings=args.openings,pairs=PAIRS,discount=.995,max_moves=320,opening_plies=4,seed_base=1200000)
        # JSON normalizes pairs to lists, so normalize before comparing a resumed config.
        config=json.loads(json.dumps(config))
        if (root/'config.json').exists():
            old=json.loads((root/'config.json').read_text())
            if old['experiment']!=config:
                if not args.extend or not compatible_extension(old['experiment'],config):
                    raise ValueError('Existing experiment configuration differs')
                event('extended',previous_openings=old['experiment']['openings'],openings=args.openings)
                old['experiment']=config
                atomic_json(root/'config.json',old)
                shutil.copy(__file__,root/'source/search_budget_benchmark_extended.py')
        else:
            shutil.copy(args.model,root/'model.pt')
            atomic_json(root/'config.json',dict(experiment=config,source_model=str(args.model.resolve()),
                model_sha256=hashlib.sha256((root/'model.pt').read_bytes()).hexdigest()))
            sources=['search_budget_benchmark.py','az_longrun.py','self_play_resnet_mcts.py',
                     'mcts_history.py','nn_resnet.py','alphazero_nn.py','Chinese_Checker.py','checkpoint_paths.py']
            (root/'source').mkdir(exist_ok=True)
            for source in sources:shutil.copy(source,root/'source'/source)
        if args.resume:
            (root/'STOP').unlink(missing_ok=True)
        expected=json.loads((root/'config.json').read_text())['model_sha256']
        if hashlib.sha256((root/'model.pt').read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen model hash changed')
        model=NetResNet.load(root/'model.pt').eval()
        (root/'games').mkdir(exist_ok=True)
        records=[];update();report(root,records,len(tasks),'running');event('started',pid=os.getpid(),total=len(tasks))
        for i,(high,low,seed,side) in enumerate(tasks):
            path=root/'games'/f'{i+1:03d}.pkl'
            if path.exists():
                rec=pickle.loads(path.read_bytes())
                if (rec['comparison'],rec['seed'],rec['high_side'])!=([high,low],seed,side):
                    raise ValueError('Saved game does not match schedule')
                records.append(rec);continue
            if (root/'STOP').exists():
                update(status='stopped',finished=len(records));report(root,records,len(tasks),'stopped');return
            budgets=[high,low] if side==1 else [low,high]
            update(game=i+1,finished=len(records),budgets=budgets,seed=seed,ply=0)
            start=time.monotonic()
            rec=play([model,model],seed,budgets,[.995,.995],opening=4,
                     progress=lambda ply:update(ply=ply))
            rec.update(comparison=[high,low],high_side=side,elapsed_seconds=time.monotonic()-start)
            atomic_pickle(path,rec);records.append(rec)
            event('game',game=i+1,budgets=budgets,winner=rec['winner'],reason=rec['reason'],
                  plies=len(rec['steps']),seconds=rec['elapsed_seconds'])
            update(finished=len(records));report(root,records,len(tasks),'running')
        update(status='complete',finished=len(records));report(root,records,len(tasks),'complete');event('complete',games=len(records))
    except BaseException:
        update(status='failed',error=traceback.format_exc());event('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
