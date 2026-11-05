"""只读检查既有真实终局样本的价值输出；不训练、不查询教师。"""
import argparse,hashlib,json,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from az121 import Net121
from cap_opening121_replay import load_data
from checkpoint121_save import REGISTRY
SOURCE=Path('nn_checkpoints/experiments/fresh121_projection_20261105')
STAGES=('总计','0—19手','20—39手','40—79手','80手及以后','终局前16手（old标记）')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metadata(sample):
    x=np.asarray(sample['x']);player=1 if float(x[3].max())>0 else 2;ply=round(float(x[4].max())*240)
    stage='0—19手' if ply<20 else '20—39手' if ply<40 else '40—79手' if ply<80 else '80手及以后'
    return dict(player=player,ply=ply,stage=stage)

def in_stage(sample,stage):
    return stage=='总计' or (bool(sample.get('endgame',False)) if stage=='终局前16手（old标记）' else metadata(sample)['stage']==stage)

def summarize(pred,target,train_mean):
    pred=np.asarray(pred,dtype=np.float64);target=np.asarray(target,dtype=np.float64)
    if len(pred)!=len(target) or not len(pred):raise ValueError('empty/mismatched values')
    if not np.isfinite(pred).all() or not np.isfinite(target).all():raise ValueError('nonfinite value')
    nonzero=target!=0;correct=(np.sign(pred[nonzero])==np.sign(target[nonzero]));ties=pred[nonzero]==0
    correlation=None if np.std(pred)<1e-12 or np.std(target)<1e-12 else float(np.corrcoef(pred,target)[0,1])
    return dict(n=len(pred),prediction_mean=float(pred.mean()),prediction_std=float(pred.std()),prediction_min=float(pred.min()),prediction_max=float(pred.max()),target_mean=float(target.mean()),target_std=float(target.std()),positive_target_fraction=float((target>0).mean()),negative_target_fraction=float((target<0).mean()),draw_target_fraction=float((target==0).mean()),mse=float(np.mean((pred-target)**2)),mae=float(np.mean(abs(pred-target))),zero_baseline_mse=float(np.mean(target**2)),train_mean=train_mean,train_mean_baseline_mse=float(np.mean((target-train_mean)**2)),correlation=correlation,sign_samples=int(nonzero.sum()),sign_accuracy_half_ties=float((correct.astype(float)+.5*ties).mean()) if nonzero.any() else None,zero_prediction_fraction=float((pred==0).mean()),abs_prediction_below_01_fraction=float((abs(pred)<.1).mean()),saturated_abs_ge_095_fraction=float((abs(pred)>=.95).mean()))

@torch.no_grad()
def predict_values(model,samples):
    values=[]
    for start in range(0,len(samples),32):
        x=torch.from_numpy(np.stack([s['x'] for s in samples[start:start+32]]));_,v=model(x);values.extend(v.cpu().tolist())
    return np.asarray(values)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=SOURCE);ap.add_argument('--output',type=Path,default=Path('docs/experiments/roadmap121_step1_20261105/value.json'));a=ap.parse_args();torch.set_num_threads(1)
    cfg=json.loads((a.source/'config.json').read_text());registry_before=sha(REGISTRY);paths={'v3':Path(cfg['model']),'control':a.source/'control_step2400.pt','projected':a.source/'projected_step2400.pt'};hashes={k:sha(p) for k,p in paths.items()}
    assert hashes['v3']==cfg['model_sha256'] and registry_before==cfg['registry_sha256']
    for n,digest in cfg['data_sha256'].items():assert sha(a.source/'data_sources'/n)==digest
    _,old,normal,_=load_data(a.source/'data_sources');pools={'old':old,'normal':normal};rows=[];coverage={};raw_old_target_checks=0;normal_target_checks=0
    from cycle121_targeted import expert_samples,value_label
    normal_outcomes={part:Counter() for part in ('train','validation')}
    for line in (a.source/'data_sources/normal_games.jsonl').read_text().splitlines():
        record=json.loads(line);normal_outcomes[record['split']][f"winner={record['winner']};reason={record['reason']}"]+=1
        for sample in expert_samples(record):
            expected=value_label(record['winner'],record['reason'],metadata(sample)['player']);assert (sample['z'],sample['value_weight'])==expected;normal_target_checks+=1
    for source,pool in pools.items():
        genuine={part:[s for s in samples if s.get('value_weight',1.)>0] for part,samples in pool.items()};train_mean=float(np.mean([s['z'] for s in genuine['train']]))
        coverage[source]={part:dict(all=len(pool[part]),genuine=len(ss),excluded_zero_weight=len(pool[part])-len(ss)) for part,ss in genuine.items()}
        for ss in genuine.values():
            for s in ss:
                if s['z'] not in (-1.,0.,1.):raise ValueError('nonterminal target scale')
                if 'winner' in s:
                    assert s['z']==(1. if s['winner']==metadata(s)['player'] else -1.);raw_old_target_checks+=1
        for name,path in paths.items():
            m=Net121.load(path)
            for part,samples in genuine.items():
                pred=predict_values(m,samples);target=np.array([s['z'] for s in samples]);meta=[metadata(s) for s in samples]
                for stage in STAGES:
                    for player in (0,1,2):
                        ix=[i for i,x in enumerate(meta) if in_stage(samples[i],stage) and (player==0 or x['player']==player)]
                        if not ix:continue
                        train_stratum=[s['z'] for s in genuine['train'] if in_stage(s,stage) and (player==0 or metadata(s)['player']==player)]
                        row=dict(source=source,split=part,model=name,stage=stage,player=player,**summarize(pred[ix],target[ix],train_mean));row['matched_train_stratum_mean']=float(np.mean(train_stratum)) if train_stratum else None;row['matched_train_stratum_baseline_mse']=float(np.mean((target[ix]-np.mean(train_stratum))**2)) if train_stratum else None;rows.append(row)
    deps=['diagnostics/diagnose121_value.py','az121.py','cap_opening121_replay.py','curriculum121.py','canonical121_trial.py','cycle121_targeted.py']
    result=dict(scope='只读旧终局样本，不训练、不新增教师查询；每行按样本等权，非独立对局统计。阶段按总手数划分，不等同真实残局。',source=str(a.source),model_sha256=hashes,data_sha256=cfg['data_sha256'],config_sha256=sha(a.source/'config.json'),source_sha256={n:sha(n) for n in deps},registry_sha256=registry_before,coverage=coverage,normal_game_outcomes={k:dict(v) for k,v in normal_outcomes.items()},target_audit=dict(definition='z=+1当真实终局胜者等于当前走棋方，-1当其对手获胜；真实和棋为0，max_ply样本value_weight=0被排除。',canonical='canonical_sample仅旋转坐标，不改z；身份平面保留原player；value_label同样使用当前player。',old_explicit_winner_label_checks=raw_old_target_checks,normal_explicit_winner_label_checks=normal_target_checks,sign='目标为0的样本排除sign统计；预测恰好0计半分；饱和阈值|v|>=0.95。'),rows=rows)
    assert all(sha(path)==hashes[k] for k,path in paths.items()) and sha(REGISTRY)==registry_before
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');write_report(result,a.output.with_suffix('.md'));print(json.dumps(dict(coverage=coverage,models=hashes,rows=len(rows)),ensure_ascii=False))

def write_report(result,path):
    lines=['# 价值头只读诊断','','预测真实棋谱最终结果，标签从当前走棋方视角定义。早期真实±1标签是该条轨迹结果，不是局面的必然胜负。所有统计按样本等权，连续棋谱样本相关，以下不是独立棋力验收。','','|数据|模型|样本|预测均值±标准差|MSE|常数0 MSE|训练均值 MSE|符号准确率|相关系数|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['rows']:
        if row['split']=='validation' and row['stage']=='总计' and row['player']==0:
            correlation='无定义' if row['correlation'] is None else f"{row['correlation']:.3f}";sign='无定义' if row['sign_accuracy_half_ties'] is None else f"{row['sign_accuracy_half_ties']:.1%}"
            lines.append(f"|{row['source']}留出|{row['model']}|{row['n']}|{row['prediction_mean']:.3f}±{row['prediction_std']:.3f}|{row['mse']:.3f}|{row['zero_baseline_mse']:.3f}|{row['train_mean_baseline_mse']:.3f}|{sign}|{correlation}|")
    lines+=['','## 结论及阵营分布核查','','v3价值输出幅度很小，约四分之三留出样本|v|<0.1；它不是严格常量，但整体未优于常数0基线。control/projected输出幅度增大后，留出误差和负相关反而加重，饱和比例仍为0；问题不是tanh输出饱和。','','normal的训练与留出阵营结果方向翻转：训练12局为P1胜7、P2胜3、真实等轮和2；留出4局为P1胜1、P2胜3。总体正负接近均衡会掩盖这种偏置。v3预测在两组始终P1偏正、P2偏负，符合训练阵营倾向，却与留出倾向相反。总体训练相关不能直接视为局面区分能力；分玩家后相关接近0。','','|来源划分|玩家|样本|目标正/负/和比例|目标均值|预测均值|相关|','|---|---:|---:|---:|---:|---:|---:|']
    for row in result['rows']:
        if row['model']=='v3' and row['source']=='normal' and row['stage']=='总计' and row['player'] in (1,2):
            corr='无定义' if row['correlation'] is None else f"{row['correlation']:.3f}"
            lines.append(f"|normal {row['split']}|{row['player']}|{row['n']}|{row['positive_target_fraction']:.1%}/{row['negative_target_fraction']:.1%}/{row['draw_target_fraction']:.1%}|{row['target_mean']:.3f}|{row['prediction_mean']:.3f}|{corr}|")
    lines+=['','按总手数分层没有跨来源稳定有效的阶段：old 80手后仅弱正相关，而normal同阶段负相关。old真实终局前16手112例总体MSE1.006、相关0.096；P1的56例有正排序相关0.496，但预测普遍偏正，MSE仍1.036，不能当作可靠胜负判断。早期轨迹终局±1具有高不确定性，不要求每例必然正确；这里用常数与训练先验基线检验实际信息增益。','','优先核查并平衡普通数据的阵营/对局来源分布，按对局组及几何轨道保守隔离重建评估，再诊断价值监督如何提供局面信息。几何轨道隔离用于防止近似样本泄漏，不等于断言博弈对称。strict规则固定P1先手，等轮判和带有身份语义；身份平面应保留，转方后的输出差仅是身份敏感性证据，不据此判编码错误、删除身份或翻转标签。当前样本少且时序相关，不据此再做大训练。']
    lines+=['','## v3按阶段与双方留出（最后16手是额外交叠子集）','','|数据|总手阶段|玩家（0为双方）|样本|预测均值±标准差|MSE|训练阶段/玩家均值基线|符号准确率|相关|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['rows']:
        if row['split']=='validation' and row['model']=='v3':
            baseline='无对应训练样本' if row['matched_train_stratum_baseline_mse'] is None else f"{row['matched_train_stratum_baseline_mse']:.3f}";corr='无定义' if row['correlation'] is None else f"{row['correlation']:.3f}";sign='无定义' if row['sign_accuracy_half_ties'] is None else f"{row['sign_accuracy_half_ties']:.1%}"
            lines.append(f"|{row['source']}|{row['stage']}|{row['player']}|{row['n']}|{row['prediction_mean']:.3f}±{row['prediction_std']:.3f}|{row['mse']:.3f}|{baseline}|{sign}|{corr}|")
    lines+=['','阶段按总手数划分，另补old数据标记的终局前16手交叠子集；normal样本不保留此标记，不推定该子集。完整三模型训练/留出、阶段、双方统计见value.json，包括范围、MAE、目标比例、近零及饱和比例。零目标不参与符号准确率；预测严格等于0计半分。相关系数在任一变量常量时记null。常数基线使用相应来源训练集均值，另列匹配阶段/玩家的训练均值基线，不用留出集拟合基线。','','old标签逐例核对真实winner与编码player；normal由expert_samples/value_label生成，重放验证真实终局，限步和棋零权重排除。源码与样本、模型哈希见JSON。']
    path.write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
