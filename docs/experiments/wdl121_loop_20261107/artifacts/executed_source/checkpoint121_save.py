"""Verified local save slots for the deployed strict121 neural engine."""
import argparse,hashlib,json,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REGISTRY=ROOT/'nn_checkpoints/models/champions/neural121.json'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(folder):
    folder=Path(folder);m=json.loads((folder/'manifest.json').read_text())
    for name,sha in m['files'].items():
        if digest(folder/name)!=sha:raise ValueError('save checksum mismatch: '+name)
    return m

def save(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    registry=json.loads(REGISTRY.read_text());model=ROOT/registry['checkpoint']
    if digest(model)!=registry['sha256']:raise ValueError('active checkpoint mismatch')
    shutil.copyfile(REGISTRY,folder/'registry.json');shutil.copyfile(model,folder/'model.pt')
    # Standalone source snapshot includes untracked code; never changes Git HEAD.
    for src in ROOT.rglob('*'):
        relative=src.relative_to(ROOT)
        if any(part in {'.git','nn_checkpoints','__pycache__','.venv','venv','node_modules'} for part in relative.parts):continue
        if src.is_file() and src.suffix in {'.py','.c','.h','.md','.json','.toml'}:
            target=folder/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
    (folder/'git_head.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT))
    (folder/'tracked_changes.patch').write_bytes(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT))
    (folder/'git_status.txt').write_bytes(subprocess.check_output(['git','status','--short'],cwd=ROOT))
    files={str(p.relative_to(folder)):digest(p) for p in folder.rglob('*') if p.is_file()}
    manifest=dict(version=registry['version'],checkpoint=registry['checkpoint'],created_unix=time.time(),files=files,model_sha256=registry['sha256'],registry_sha256=digest(REGISTRY))
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2));verify(folder);return manifest

def restore(folder):
    folder=Path(folder);m=verify(folder);registry=json.loads((folder/'registry.json').read_text());target=ROOT/registry['checkpoint']
    if target.exists() and digest(target)!=m['model_sha256']:raise ValueError('immutable checkpoint collision; preserve it for investigation')
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():shutil.copyfile(folder/'model.pt',target)
    backup=REGISTRY.with_name('neural121_before_restore_'+str(time.time_ns())+'.json')
    if REGISTRY.exists():shutil.copyfile(REGISTRY,backup)
    temporary=REGISTRY.with_suffix('.restore.tmp');shutil.copyfile(folder/'registry.json',temporary);temporary.replace(REGISTRY)
    return dict(restored=registry['version'],sha256=digest(target))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('operation',choices=['save','verify','restore']);ap.add_argument('folder',type=Path);a=ap.parse_args()
    print(json.dumps(globals()[a.operation](a.folder),ensure_ascii=False))
