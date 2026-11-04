"""Supplement original probe grid with deterministic P2 turns; no new games.
Original grid (even openings + offsets 0,30,60) covered P1 only. Preserve it,
add offsets 31/61 from the same trajectories and report both separately.
"""
import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from az121 import Net121,Neural121,encode
from curriculum121 import replay,atomic
from diagnostics.policy_value121_cross import CrossPolicyValue
RUN=Path('nn_checkpoints/experiments/policy_value121_cross_20261101')
def main():
    torch.set_num_threads(1);cfg=json.loads((RUN/'config.json').read_text());old=Net121.load(cfg['models']['v3']['path']);new=Net121.load(cfg['models']['new']['path']);hybrid=CrossPolicyValue(new,old)
    probes=[]
    for line in (RUN/'new_vs_v3.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['side']!=1:continue
        g=replay(row['opening'])
        for offset,path in enumerate(row['paths'],1):
            g.play(path)
            if offset in (31,61) and g.result()[0] is None:probes.append((row['pair'],copy.deepcopy(g)))
    records=[]
    with torch.no_grad():
        for pair,g in probes:
            assert g.player==2
            x=torch.from_numpy(encode(g.board,g.player,g.ply,g.repetitions,canonical=True)).unsqueeze(0);_,a=old(x);_,b=new(x)
            before,info_new=Neural121(new,32).think(g.board,g.player,g.ply,g.repetitions);after,info_old=Neural121(hybrid,32).think(g.board,g.player,g.ply,g.repetitions)
            if info_new['priors']!=info_old['priors']:raise ValueError('policy priors differ')
            records.append(dict(pair=pair,player=2,ply=g.ply,old_value=float(a[0]),new_value=float(b[0]),new_path=before,hybrid_path=after,
                                changed_move=(before[0],before[-1])!=(after[0],after[-1])))
    atomic(RUN/'probes_p2_supplement.json',records)
    original=json.loads((RUN/'probes.json').read_text());combined=original+records;stats={}
    for player in (1,2):
        group=[x for x in combined if x['player']==player];a=np.array([x['old_value'] for x in group]);b=np.array([x['new_value'] for x in group])
        stats[str(player)]=dict(n=len(group),changed_moves=sum(x['changed_move'] for x in group),old_mean=float(a.mean()),old_std=float(a.std()),new_mean=float(b.mean()),new_std=float(b.std()),mean_absolute_difference=float(abs(a-b).mean()))
    result=dict(original_grid='P1 only; even opening lengths and even offsets',supplement='Same 24 trajectories, offsets31/61, P2 only; no new games or changes to original outcome analysis',per_player=stats,
                total=len(combined),changed_moves=sum(x['changed_move'] for x in combined))
    atomic(RUN/'probe_analysis.json',result);(RUN/'probe_supplement_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result))
if __name__=='__main__':main()
