"""Standalone container entry point. Default action is a CPU-only boundary probe."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

INPUT = Path('/input')
OUTPUT = Path('/output')
CONFIG = 'conf/pipeline/imc2025/mast3rhybrid/mast3rhybrid-022-c.yaml'


def probe():
    root = '/content/ns64/project'
    forbidden = ['/content/ns64/data/train/ETs', '/content/imc_ets_images.zip', '/content/drive',
                 '/kaggle/input', '/private.json', '/input/../private.json', '/proc/1/root/content/drive',
                 root, root+'/configs/baselines/organizer_v1/development_scoring/ETs/train_labels.csv',
                 root+'/configs/baselines/top10_reproduction_v1/pilot.json',
                 root+'/configs/baselines/top10_reproduction_v1/ns64_pilot_source.zip.bin',
                 '/content/ns64/ets_attempt/inference/predictions.csv','../../../content/ns64/data/train/ETs',
                 '/input/../content/ns64/project', '/var/run/docker.sock', '/private/admin-canary']
    link = OUTPUT/'escape-test'
    link.symlink_to('/private/admin-canary')
    results = {}
    try:
        for name in [*forbidden,str(link)]:
            try:
                fd = os.open(name,os.O_RDONLY)
            except (FileNotFoundError, PermissionError, NotADirectoryError) as error:
                results[name] = type(error).__name__
            else:
                os.close(fd)
                results[name] = 'READABLE'
    finally:
        link.unlink()
    status = Path('/proc/self/status').read_text().splitlines()
    mounts = [line.split()[4] for line in Path('/proc/self/mountinfo').read_text().splitlines()]
    ok = ('READABLE' not in results.values() and os.getuid()==65534 and
          'NoNewPrivs:\t1' in status and 'CapEff:\t0000000000000000' in status and
          INPUT.is_mount() and OUTPUT.is_mount() and os.statvfs(INPUT).f_flag & os.ST_RDONLY)
    import subprocess
    environment = subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
                                   '--format=csv,noheader'],text=True)
    evidence = dict(environment=environment, gpu=gpu,status='PASS' if ok else 'BLOCKED', reads=results, uid=os.getuid(),
                    gid=os.getgid(), mountpoints=mounts, namespace=os.readlink('/proc/self/ns/mnt'),
                    worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUTPUT/'boundary.json').write_text(json.dumps(evidence,indent=2))
    if not ok:
        raise RuntimeError('Boundary check failed')


def infer():
    import pandas as pd
    import torch
    import yaml
    source = INPUT/'source'
    sys.path.insert(0,str(source))
    definitions = yaml.safe_load((source/'conf/models.yaml').read_text())
    for alias,path in json.loads((INPUT/'aliases.json').read_text()).items():
        definitions[alias] = dict(local=path,kernel=path)
    aliases = OUTPUT/'models.json'
    aliases.write_text(json.dumps(definitions))
    os.environ.update(DEFAULT_MODEL_LIST_PATH=str(aliases), DEFAULT_EXTRA_DIR=str(OUTPUT/'extra'))
    from config import PipelineConfig
    from data import IMC2025TestData
    from pipeline import create_pipeline
    names = sorted(p.name for p in (INPUT/'test/ETs').iterdir())
    if names != [f'image_{i:04d}.png' for i in range(22)]:
        raise ValueError('Opaque inventory mismatch')
    frame = pd.DataFrame([dict(image_id=str(i), dataset='ETs',scene='cluster0',image=name,
                               rotation_matrix='',translation_vector='') for i,name in enumerate(names)])
    pipeline = create_pipeline(PipelineConfig.load_config(source/CONFIG), device=torch.device('cuda:0'))
    predictions = pipeline.run(frame, IMC2025TestData(frame,data_root_dir=INPUT))
    predictions.to_csv(OUTPUT/'opaque.csv',index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer',action='store_true')
    args = parser.parse_args()
    probe()
    if args.infer:
        infer()

