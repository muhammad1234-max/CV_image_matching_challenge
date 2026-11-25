"""Exclusive experiment registration; only a genuinely frozen benchmark is accepted."""
import argparse,json
from pathlib import Path
from benchmark_contract import validate_experiment,save_new
def register(manifest,benchmark,root):
    validate_experiment(manifest,benchmark)
    identifier=manifest['experiment_id']
    if not identifier or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in identifier):raise ValueError('Unsafe experiment ID')
    path=Path(root)/manifest['role']/(identifier+'.json')
    save_new(path,manifest)
    return path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--benchmark',required=True);p.add_argument('--root',default='experiments');a=p.parse_args()
    print(register(json.loads(Path(a.manifest).read_text()),json.loads(Path(a.benchmark).read_text()),a.root))
