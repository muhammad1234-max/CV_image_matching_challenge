"""Bounded artifact and runtime preflight for the published ns64 ETs pilot."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
import zipfile
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def run(plan, input_root, output):
    started = time.monotonic()
    report = dict(team=plan['team'], dataset='ETs', status='BLOCKED',
                  hardware={}, checkpoints={}, stages={}, predictions=None, scores=None)
    stage = 'hardware'
    try:
        gpu = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            timeout=10, text=True)
        report['hardware'] = dict(gpus=gpu.splitlines(), python=platform.python_version())
        packages = {}
        for name in ['torch', 'pycolmap', 'kornia', 'asmk', 'curope', 'mast3r']:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        report['packages'] = packages
        probe = subprocess.run([__import__('sys').executable, '-c',
            'import torch; print(torch.ones(1, device="cuda").item()); import pycolmap, asmk, curope, mast3r'],
            capture_output=True, text=True, timeout=90)
        report['import_cuda_probe'] = dict(returncode=probe.returncode, stderr=probe.stderr[-2000:])
        report['stages'][stage] = 'PASS'
        stage = 'source'
        source = next(input_root.rglob('ns64_pilot_source.zip.bin'), None)
        if source is None or digest(source) != plan['source_zip_sha256']:
            raise ValueError('Mount the exact ns64_pilot_source.zip.bin from the preparation bundle')
        report['source_sha256'] = digest(source)
        report['stages'][stage] = 'PASS'
        stage = 'inputs'
        candidates = [p for p in input_root.rglob('train/ETs') if p.is_dir()]
        if len(candidates) != 1:
            raise ValueError('Expected one competition train/ETs directory')
        for name, expected in plan['images'].items():
            if digest(candidates[0] / name) != expected:
                raise ValueError('ETs image identity mismatch')
        report['stages'][stage] = 'PASS'
        stage = 'checkpoints'
        for key, filename in plan['weight_files'].items():
            paths = list(input_root.rglob(filename))
            report['checkpoints'][key] = [{'path':str(p), 'sha256':digest(p)} for p in paths if p.is_file()]
        raise RuntimeError('Original checkpoint versions and complete compatible environment are not pinned. Review the inventory before inference.')
    except Exception as exc:
        report.update(failed_stage=stage, error=str(exc))
        report['stages'][stage] = 'BLOCKED'
    report['elapsed_seconds'] = time.monotonic() - started
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in [('plan', plan), ('result', report)]:
            archive.writestr(name+'.json', json.dumps(data, indent=2))
    return report
