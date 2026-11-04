"""Finish reporting from completed1600-step checkpoints; never collect or train."""
import json,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from az121 import Net121
from corrected121_decisions import root_metrics
from generalization121_curve import predict_preferences
from curriculum121 import atomic
from checkpoint121_save import digest,REGISTRY

def main():
    torch.set_num_threads(1);p=Path(sys.argv[1]);began=time.monotonic();cfg=json.loads((p/'config.json').read_text());prior=json.loads((p/'status.json').read_text())
    if prior['stage']!='failed':raise ValueError('only finalize an interrupted report')
    for name,sha in cfg['source_sha256'].items():
        if digest(p/'source'/name)!=sha:raise ValueError('original source snapshot changed')
    for name,sha in cfg['data_sha256'].items():
        if digest(p/'data_sources'/name)!=sha:raise ValueError('input data changed')
    if digest(REGISTRY)!=cfg['registry_sha256'] or digest(cfg['model'])!=cfg['model_sha256']:raise ValueError('active baseline changed')
    shutil.copyfile(p/'status.json',p/'status_before_finalization.json')
    data=torch.load(p/'decisions.pt',weights_only=True);roots=data['sets'];histories={};predictions={};checkpoint_hashes={}
    for arm in ('small72','expanded160'):
        histories[arm]=json.loads((p/(arm+'_curve.json')).read_text());last=histories[arm][-1]
        if last['step']!=1600:raise ValueError('training incomplete; do not fabricate final result')
        path=p/(arm+'_step1600.pt');checkpoint_hashes[arm]=digest(path);model=Net121.load(path)
        measured=root_metrics(model,roots['validation'])
        for key in ('policy_ce','accuracy','preference_accuracy','hinge'):
            if abs(measured[key]-last['validation40'][key])>2e-5:raise ValueError('saved checkpoint and logged metrics mismatch')
        predictions[arm]=predict_preferences(model,roots['validation'])
    atomic(p/'validation_predictions.json',predictions);quality=json.loads((p/'data_quality.json').read_text())
    elapsed=prior['elapsed']+time.monotonic()-began
    result=dict(training_started=True,primary_checkpoint=1600,final={arm:h[-1] for arm,h in histories.items()},data=dict(roots=200,groups=quality['total_opening_groups'],train160=160,validation40=40),
        promoted=False,arena_run=False,seconds=elapsed,report_recovered=True,note='Fixed-budget learning curves; no playing-strength inference or automatic1000 expansion.')
    atomic(p/'result.json',result)
    plot=subprocess.run([cfg['plot_python'],'diagnostics/plot121_generalization.py',str(p)],capture_output=True,text=True)
    atomic(p/'plot_status.json',dict(exit_code=plot.returncode,stderr=plot.stderr[-2000:]))
    if plot.returncode:raise RuntimeError(plot.stderr)
    lines=['# 200去重决策与泛化学习曲线\n',f'数据覆盖{quality["total_opening_groups"]}个开局组（训练{quality["groups"]["train"]}、验证{quality["groups"]["validation"]}），共160训练/40验证决策；新局面中位第{quality["median_ply"]}手。\n',
           '两组从相同v3参数出发，普通棋谱和抽样顺序相同，各1600步，根决策占25%。小组仅用160训练池中嵌套的72样本，大组使用完整160。原始曲线和模型均已完成保存，汇总阶段漏导入F导致中断；本次只修复汇总，没有重跑数据采集或训练。\n',
           '| 方案 | 训练偏好正确率 | 验证偏好正确率 | 验证专家首选一致率 | 训练CE | 验证CE |', '|---|---:|---:|---:|---:|---:|']
    for arm,h in histories.items():
        x=h[-1];a=x['seen_train'];b=x['validation40'];lines.append(f'| {arm} | {a["preference_accuracy"]:.1%} | {b["preference_accuracy"]:.1%} | {b["accuracy"]:.1%} | {a["policy_ce"]:.3f} | {b["policy_ce"]:.3f} |')
    lines+=['\n扩大数据后，验证偏好正确率27.5%→35%，专家首选一致率17.5%→25%，各相差3/40个验证决策；提示可能有覆盖收益，但样本小且同开局内相关，不能据此认定棋力提升。两组训练与验证仍有差距；在当前混合目标、保留约束和训练预算下尚未充分拟合，不能直接断言CNN表达能力不足。',
            '\n只研究刻意筛出的早期困难决策，初始低准确率并不代表所有跳棋走法。根标签是固定专家搜索偏好，无胜负价值标签。40验证根按开局隔离，全数据棋盘/走棋方去重，并移除普通微调数据中的精确验证输入重合。原始暖启动历史不等于完全陌生棋形。',
            '\n72样本组的common_train160指标包含88未训练决策，不能把它当作其全部见过的训练数据。主比较固定1600步，没有依据验证最优点挑模型。',
            '\n本轮没有后台对战或升级，GUI仍v3。结果用于确定下一轮拟合/泛化实验，未自动扩大到1000。',
            '\n文件：learning_curves.png、learning_curves.svg、learning_curves.csv、两组curve.json、validation_predictions.json、抽样审计与中文日志。']
    (p/'REPORT.md').write_text('\n'.join(lines));shutil.copyfile(__file__,p/'finalization_source.py')
    atomic(p/'finalization_audit.json',dict(no_collection_or_training_rerun=True,checkpoint_metrics_match=True,checkpoints_sha256=checkpoint_hashes,baseline_and_registry_unchanged=True,source_sha256=digest(__file__)))
    with (p/'中文日志.md').open('a') as f:f.write('\n- 汇总恢复：补齐损失函数导入，直接读取两组已完成1600步检查点核对指标，生成曲线/CSV/报告；未重跑采样或训练，现役v3不变。\n')
    atomic(p/'status.json',dict(stage='complete',elapsed=elapsed,result=result));print(json.dumps(dict(stage='complete',groups=quality['total_opening_groups'],roots=200,report=str(p/'REPORT.md'))))
if __name__=='__main__':main()
