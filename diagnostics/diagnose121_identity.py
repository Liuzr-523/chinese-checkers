"""事后只读：几何转方身份敏感性；strict固定P1先手，非完整博弈对称性检验。"""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import rules121 as r
from az121 import Net121,encode,action_id,CANONICAL_ENCODING
from history121_exclusions import decode_sample_inputs
from opening121_split_audit import parse_decision,rotate_swap
from cap_opening121_replay import load_data
from curriculum121 import replay,batch
from checkpoint121_save import REGISTRY,digest
SOURCE=Path('nn_checkpoints/experiments/fresh121_projection_20261105')


def terminal_asymmetry_witness():
    """固定P1先手的等轮判和不随几何转方交换；身份不是可直接删除的冗余。"""
    for target in sorted(r.GOAL[2]):
        for start in r.NEIGHBORS[target]:
            if start in r.GOAL[1] or start in r.GOAL[2]:continue
            board=[0]*121
            for i in r.GOAL[1]:board[i]=1
            for i in r.GOAL[2]-{target}:board[i]=2
            board[start]=2
            if not r.valid_path(board,2,[start,target]):continue
            twin=[3-v if v else 0 for v in reversed(board)]
            return dict(board=board,player=2,last_step=[start,target],original_terminal=r.terminal(board,2),rotated_board=twin,rotated_player=1,rotated_terminal=r.terminal(twin,1))
    raise ValueError('无法构造strict终局非对称见证')


def paired_input(sample):
    x=np.asarray(sample['x'],dtype=np.float32);identity=next(iter(decode_sample_inputs(x,CANONICAL_ENCODING)));original,player=parse_decision(identity);board,other=parse_decision(rotate_swap(identity));ply=round(float(x[4].max())*240);repeat=round(float(x[5].max())*3)
    twin=encode(board,other,ply,{(tuple(board),other):repeat},canonical=True);expected=x.copy();expected[3]=x[2]-x[3]
    if not np.array_equal(twin,expected):raise ValueError('实际转方编码不等于仅切换identity平面')
    legal={action_id(s,p[-1],other,True) for s,p in r.get_all_moves(board,other)}
    if legal!=set(sample['actions']):raise ValueError('真实转方合法canonical动作集合不一致')
    return twin,player


def divergence(logits,other,actions):
    a=logits[actions].log_softmax(0);b=other[actions].log_softmax(0);pa,pb=a.exp(),b.exp();middle=(pa+pb)/2
    return float((.5*(pa*(a-middle.log())).sum()+.5*(pb*(b-middle.log())).sum()).clamp_min(0))


@torch.no_grad()
def inspect(model,samples,twins,players):
    records=[]
    for start in range(0,len(samples),32):
        ss=samples[start:start+32];x,*_=batch(ss);l,v=model(x);tl,tv=model(torch.from_numpy(np.stack(twins[start:start+32])))
        for j,s in enumerate(ss):
            actions=s['actions'];positive=s.get('positive',[actions[s['target']]]);i=start+j;a=actions[int(l[j,actions].argmax())];b=actions[int(tl[j,actions].argmax())]
            mass=lambda logits:float((torch.logsumexp(logits[positive],0)-torch.logsumexp(logits[actions],0)).exp())
            records.append(dict(index=i,player=players[i],top1_changed=a!=b,original_action=a,paired_canonical_action=b,policy_js=divergence(l[j],tl[j],actions),value=float(v[j]),paired_value=float(tv[j]),value_abs_difference=abs(float(v[j]-tv[j])),reference_mass=mass(l[j]),paired_reference_mass=mass(tl[j])))
    def summarize(rows):return dict(n=len(rows),top1_changed=sum(x['top1_changed'] for x in rows),mean_policy_js=float(np.mean([x['policy_js'] for x in rows])),mean_value_abs_difference=float(np.mean([x['value_abs_difference'] for x in rows])),max_value_abs_difference=max(x['value_abs_difference'] for x in rows),value_difference_over_005=sum(x['value_abs_difference']>.05 for x in rows),mean_reference_mass_change=float(np.mean([x['paired_reference_mass']-x['reference_mass'] for x in rows])))
    return dict(overall=summarize(records),by_original_player={str(p):summarize([x for x in records if x['player']==p]) for p in (1,2)},records=records)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();torch.set_num_threads(1);registry=digest(REGISTRY)
    fixed,old,normal,sets=load_data(SOURCE/'data_sources');labels=json.loads((SOURCE/'development.json').read_text());fresh=[]
    for row in labels:
        g=replay(row['prefix']);actions=[action_id(s,p[-1],g.player,True) for s,p in r.get_all_moves(g.board,g.player)];preferred=action_id(row['preferred'][0],row['preferred'][-1],g.player,True)
        fresh.append(dict(x=encode(g.board,g.player,g.ply,g.repetitions,canonical=True),actions=actions,target=actions.index(preferred),z=0.,positive=[preferred]))
    pools=dict(book_train=sets['train'],old_value_validation=old['validation'],normal_value_validation=normal['validation'],used_fresh_development=fresh);pairs={}
    for name,samples in pools.items():
        values=[paired_input(s) for s in samples];pairs[name]=([a for a,p in values],[p for a,p in values])
    paths=dict(v3=Path(json.loads((SOURCE/'config.json').read_text())['model']),control=SOURCE/'control_step2400.pt',projected=SOURCE/'projected_step2400.pt');result={}
    for name,path in paths.items():
        model=Net121.load(path);result[name]={pool:inspect(model,ss,*pairs[pool]) for pool,ss in pools.items()}
    if digest(REGISTRY)!=registry:raise ValueError('注册表变化')
    args.output.parent.mkdir(parents=True,exist_ok=True);payload=dict(scope='事后只读机制诊断，不做模型选择/新开发验收，不查询教师。几何旋转换色后canonical除identity平面外完全相同，固定原动作顺序消除同分排序差异。strict规则固定P1先手且等轮判和非对称，因此这些配对不是完整博弈等价；输出差说明身份依赖，不单独证明编码bug/应删除身份，也不证明价值应完全相同。reference_mass沿用原标签集合，仅测敏感性，不当转方教师正确率。',source_sha256={str(p):digest(p) for p in [SOURCE/'development.json',Path(__file__),Path('az121.py'),Path('rules121.py')]},model_sha256={n:digest(p) for n,p in paths.items()},registry_sha256=registry,results=result)
    payload['strict_terminal_asymmetry_witness']=terminal_asymmetry_witness()
    args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n');print(json.dumps({n:{p:v['overall'] for p,v in pools.items()} for n,pools in result.items()},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
