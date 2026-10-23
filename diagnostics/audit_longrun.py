"""Full replay and split audit of the completed long run."""
import sys,json,pickle,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from az_longrun import validate
from checkpoint_paths import artifact_path,ROOT
root=ROOT/'experiments/longrun_20261022'
all_train=set();all_val=set();summary={};total_plies=0
for directory in sorted(root.glob('round_*')):
    validation=json.loads((directory/'validation.json').read_text())
    all_train.update(validation['train_seeds']);all_val.update(validation['val_seeds'])
    for label in ('selfplay','incumbent_eval','frozen_eval'):
        records=pickle.loads((directory/(label+'.pkl')).read_bytes())
        audits=[validate(rec) for rec in records]
        total_plies+=sum(a['plies'] for a in audits)
        summary[directory.name+'/'+label]=dict(games=len(records),terminal=sum(r['winner'] is not None for r in records),
            plies=sum(a['plies'] for a in audits),repeated_positions=sum(a['repeated_positions'] for a in audits),errors=0)
assert not(all_train & all_val)
champ_hash=hashlib.sha256(artifact_path(ROOT/'params_resnet_champ.pt').read_bytes()).hexdigest()
frozen_hash=hashlib.sha256((root/'frozen_baseline.pt').read_bytes()).hexdigest()
assert champ_hash==frozen_hash
out=dict(groups=summary,games=sum(s['games'] for s in summary.values()),
         terminal_games=sum(s['terminal'] for s in summary.values()),plies=total_plies,
         errors=0,train_games=len(all_train),validation_games=len(all_val),split_overlap=0,
         gui_champion_unchanged=True,sha256=champ_hash)
(root/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='groups'}))
