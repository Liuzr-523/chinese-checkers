"""Paired, local-only search/value ablation, resumable one game at a time."""
import argparse, fcntl, hashlib, json, os, pickle, shutil, sys, time, traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nn_resnet import NetResNet
from az_longrun import play,atomic_json,atomic_pickle,stamp
from diagnostics.value_ablation import NoValue

GROUPS=('value_on','value_zero')
def schedule(openings=20,seed_base=1300100):
    return [(group,seed_base+i,side) for i in range(openings) for group in GROUPS for side in (1,2)]

def summarize(records):
    result={}
    for group in GROUPS:
        rows=[r for r in records if r['group']==group]
        result[group]=dict(games=len(rows),wins=sum(r['winner']==r['high_side'] for r in rows),
            losses=sum(r['winner']==3-r['high_side'] for r in rows),unfinished=sum(r['winner'] is None for r in rows))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1)
    source=Path('nn_checkpoints/experiments/search_budget_20261022/model.pt')
    config=dict(openings=20,seed_base=1300100,opening_plies=4,max_moves=320,
                high_simulations=512,low_simulations=64,discount=.995,
                baseline='64 simulations with value head enabled',caveat='40 games per setting; each pair swaps colors; outcomes depend on opening sample')
    if (root/'config.json').exists():
        old=json.loads((root/'config.json').read_text())
        if old['settings']!=config:raise ValueError('Existing experiment settings differ')
    else:
        shutil.copyfile(source,root/'model.pt')
        atomic_json(root/'config.json',dict(settings=config,source_model=str(source.resolve()),
                    sha256=hashlib.sha256((root/'model.pt').read_bytes()).hexdigest()))
        (root/'source').mkdir(exist_ok=True)
        for name in ['diagnostics/value_ablation_80.py','diagnostics/value_ablation.py','az_longrun.py','self_play_resnet_mcts.py','mcts_history.py','nn_resnet.py','alphazero_nn.py','Chinese_Checker.py','checkpoint_paths.py']:
            target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,target)
    expected=json.loads((root/'config.json').read_text())['sha256']
    if hashlib.sha256((root/'model.pt').read_bytes()).hexdigest()!=expected:raise ValueError('Frozen model hash mismatch')
    model=NetResNet.load(root/'model.pt').eval();zero=NoValue(model).eval();records=[]
    (root/'games').mkdir(exist_ok=True)
    state=dict(status='running',pid=os.getpid(),started=stamp(),total=80,finished=0)
    started=time.monotonic()
    def update(**kwargs):
        state.update(kwargs,updated=stamp());atomic_json(root/'status.json',state)
    def write_report():
        result=summarize(records)
        atomic_json(root/'summary.json',dict(status=state['status'],finished=len(records),total=80,counts=result))
        lines=['# 价值头对照实验','',f"状态：{state['status']}，完成 {len(records)}/80 局。",'',
               '固定相同模型和开局，每组各40局，512次搜索对64次搜索；每个开局交换先后手。',
               '未结束单列，价值屏蔽仅影响512次搜索一方的非终局网络价值。','',
               '| 512次搜索 | 胜 | 负 | 未结束 |','|---|---:|---:|---:|']
        for key,label in [('value_on','使用价值头'),('value_zero','屏蔽价值头')]:
            row=result[key];lines.append(f"| {label} | {row['wins']} | {row['losses']} | {row['unfinished']} |")
        lines.extend(['','每局棋谱：games/；状态：status.json；运行日志：run.log。','创建 STOP 文件可在本局结束后停止，删除后用原命令重启可续跑。'])
        (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    try:
        update();write_report()
        for i,(group,seed,side) in enumerate(schedule()):
            path=root/'games'/f'{i+1:03}.pkl'
            if path.exists():
                rec=pickle.loads(path.read_bytes())
                if (rec['group'],rec['seed'],rec['high_side'])!=(group,seed,side):raise ValueError('Saved game mismatches schedule')
            else:
                if (root/'STOP').exists():
                    update(status='stopped',finished=len(records));write_report();return
                high=model if group=='value_on' else zero
                models=[high,model] if side==1 else [model,high]
                budgets=[512,64] if side==1 else [64,512]
                update(game=i+1,group=group,seed=seed,high_side=side,ply=0,finished=len(records))
                t=time.monotonic()
                rec=play(models,seed,budgets,[.995,.995],opening=4,progress=lambda ply:update(ply=ply))
                rec.update(group=group,high_side=side,elapsed_seconds=time.monotonic()-t)
                atomic_pickle(path,rec)
                print(json.dumps(dict(game=i+1,group=group,seed=seed,side=side,winner=rec['winner'],reason=rec['reason'],plies=len(rec['steps']),seconds=rec['elapsed_seconds']),ensure_ascii=False),flush=True)
            records.append(rec);update(finished=len(records));write_report()
        update(status='complete',finished=len(records),elapsed_seconds=time.monotonic()-started);write_report()
    except BaseException:
        update(status='failed',error=traceback.format_exc(),finished=len(records));write_report();raise

if __name__=='__main__':main()
