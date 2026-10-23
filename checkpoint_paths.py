"""Paths for model weights and training data; legacy basenames remain accepted."""
from pathlib import Path
import fnmatch

ROOT = Path(__file__).resolve().parent / 'nn_checkpoints'


def category(name):
    if name.startswith('params_'):
        if 'champ' in name or 'chosen_by_eval' in name:
            return Path('models/champions')
        if '_ckpt_' in name or '_step' in name:
            return Path('models/checkpoints')
        return Path('models/final')
    if name.startswith('az_'):
        return Path('data/selfplay')
    if name.startswith('states_'):
        return Path('data/positions')
    if name.startswith(('dag_', 'dgrad_')):
        return Path('data/dagger')
    if name == 'replay.pkl' or name.startswith('replay_'):
        return Path('data/replays')
    return Path('other')


def artifact_path(path):
    """Route old nn_checkpoints/name.pkl paths; explicit other paths stay intact.

    Creates category directories so generators and model saves also use the layout.
    Does not load, copy or overwrite any file.
    """
    p = Path(path)
    if p.suffix not in ('.pkl', '.pt') or p.parent.resolve() != ROOT.resolve():
        return p
    result = ROOT / category(p.name) / p.name
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def artifact_glob(directory, pattern):
    """Find legacy basename globs across category folders, without duplicates."""
    directory = Path(directory)
    if directory.resolve() != ROOT.resolve():
        return sorted(directory.glob(pattern))
    if Path(pattern).is_absolute():
        import glob
        return [Path(p) for p in sorted(glob.glob(pattern))]
    if '/' in pattern:
        return sorted(directory.glob(pattern))
    return sorted(p for p in directory.rglob('*') if p.is_file() and fnmatch.fnmatch(p.name, pattern))
