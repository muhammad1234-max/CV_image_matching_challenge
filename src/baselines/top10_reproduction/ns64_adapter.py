"""Run in a fresh Linux process. Supply the reviewed source and models only."""
import functools
import time
import math
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def stage_ets(plan, image_dir, work):
    if image_dir.name != 'ETs' or image_dir.parent.name != 'train' or len(plan['images']) != 22:
        raise ValueError('Only train/ETs with 22 pinned identities is allowed')
    target = work/'test'/'ETs'
    target.mkdir(parents=True, exist_ok=False)
    mapping = {}
    for index, (name, expected) in enumerate(sorted(plan['images'].items())):
        if Path(name).name != name:
            raise ValueError('Invalid image identity')
        source = image_dir/name
        if source.is_symlink() or digest(source) != expected:
            raise ValueError('Image digest mismatch')
        opaque = f'image_{index:04d}.png'
        shutil.copyfile(source, target/opaque)
        mapping[opaque] = name
    return mapping


def check_model_paths(paths, assets):
    for key, asset in assets.items():
        p = Path(paths[asset['alias']])
        weight = p/asset['filename'] if key=='dinov2' else p
        if weight.name != asset['filename'] or digest(weight) != asset['sha256']:
            raise ValueError('Unverified '+key)
        for name, expected in asset.get('sidecars',{}).items():
            if digest(p/name)!=expected:
                raise ValueError('Changed processor/configuration '+name)


def infer(plan, image_dir, work, source, model_paths):
    """Internal entry after dependency verification; launch inside a deadline-bound worker."""
    import zipfile
    import pandas as pd
    import torch
    project=Path(__file__).resolve().parents[3]
    bundle=project/'configs/baselines/top10_reproduction_v1/ns64_pilot_source.zip.bin'
    if digest(bundle)!='c6035c596a484331ebfaf8d3b6368b4a32acb00b3bebe670a8215c78b6c12549':
        raise ValueError('Unexpected source archive')
    source=source.resolve()
    work=work.resolve()
    source.mkdir(exist_ok=False,parents=True)
    with zipfile.ZipFile(bundle) as z:z.extractall(source)
    config_path = source/plan['config']
    if digest(config_path) != plan['config_sha256']:
        raise ValueError('Changed submission configuration')
    pin = Path(__file__).resolve().parents[3]/'configs/baselines/top10_reproduction_v1/pilot.json'
    if digest(pin) != '5951162b32f22537e547b6c2d42f1df9db48666bf92d1d2c24f7a99ff054ffbf' or plan!=json.loads(pin.read_text()):
        raise ValueError('Expected frozen ETs pilot plan')
    if work.exists() and any(work.iterdir()):
        raise ValueError('Use an empty per-attempt workspace')
    mapping = stage_ets(plan, image_dir, work)
    manifest=json.loads((pin.parent/'assets_phase3c1.json').read_text())
    check_model_paths(model_paths,manifest['assets'])
    aliases = work/'models.json'
    import yaml
    definitions = yaml.safe_load((source / 'conf/models.yaml').read_text(encoding='utf8'))
    for name, path in model_paths.items():
        resolved = str(Path(path).resolve())
        definitions[name] = {'kernel': resolved, 'local': resolved}
    aliases.write_text(json.dumps(definitions))
    os.environ.update(DEFAULT_EXTRA_DIR=str((work/"extra").resolve()), DEFAULT_MODEL_LIST_PATH=str(aliases.resolve()), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0,str(source.resolve()))
    from config import PipelineConfig
    from data import IMC2025TestData
    from pipeline import create_pipeline
    rows = [dict(image_id=str(i), dataset='ETs', scene='cluster0', image=n,
                 rotation_matrix='', translation_vector='') for i,n in enumerate(mapping)]
    frame = pd.DataFrame(rows)
    schema = IMC2025TestData(frame, data_root_dir=work)
    pipeline = create_pipeline(PipelineConfig.load_config(config_path), device=torch.device('cuda:0'))
    import pipelines.imc2025_pipeline as implementation
    import contextlib
    with contextlib.ExitStack() as stack:
        from unittest.mock import patch
        for owner, name in [(pipeline,'shortlist_generator'), (implementation,'run_point_tracking_matching_stage'),
                            (implementation,'verify_matches'), (implementation.pycolmap,'incremental_mapping')]:
            if hasattr(owner,name):
                stack.enter_context(patch.object(owner,name, measured(getattr(owner,name), name, work)))
        predictions = pipeline.run(frame, schema)
    return serialize_predictions(predictions,mapping,work)


def serialize_predictions(predictions, mapping, work):
    if list(predictions['dataset'].unique()) != ['ETs'] or sorted(predictions.image) != sorted(mapping):
        raise ValueError('Output membership mismatch')
    predictions['image'] = predictions.image.map(mapping)
    predictions.sort_values('image')[['dataset','scene','image','rotation_matrix','translation_vector']].to_csv(work/'predictions.partial.csv', index=False)
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('frozen_contract',Path(__file__).resolve().parents[3]/'snapshots/organizer_phase3a_verified/scripts/benchmark_contract.py')
    contract = module_from_spec(spec)
    spec.loader.exec_module(contract)
    for row in contract.load_records(work/'predictions.partial.csv'):
        missing = math.isnan(float(row['rotation_matrix'].split(';')[0]))
        if missing != (row['scene']=='outliers'):
            raise ValueError('Pose and scene assignment disagree')
    (work/'predictions.partial.csv').replace(work/'predictions.csv')
    return work/'predictions.csv'


def score(project, prediction, output):
    scorer = project/'src/baselines/development/evaluate.py'
    subprocess.run([sys.executable,str(scorer),'ETs',str(prediction),str(output)],check=True)


def measured(operation, name, work):
    @functools.wraps(operation)
    def wrapper(*args, **kwargs):
        import torch
        start=time.monotonic()
        row={'stage':name,'status':'FAILED'}
        try:
            result=operation(*args,**kwargs)
            if name=='shortlist_generator':
                row['ordered_pairs']=[list(map(int,pair)) for pair in result]
            if name=='run_point_tracking_matching_stage':
                row.update(match_counts(*result))
            if name=='verify_matches':
                row['verified_pairs']=result.count
                row['inliers']=sum(len(v) for group in result.inliers.values() for v in group.values())
            if name in ('shortlist_generator','incremental_mapping'):
                row['count']=len(result)
            row['status']='PASS'
            return result
        finally:
            if sys.platform=='linux':
                import resource
                row['worker_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            row.update(seconds=time.monotonic()-start,
                       peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
            with (work/'stages.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
    return wrapper


def match_counts(features, matches):
    import numpy as np
    total_points=total_matches=0
    for keys, pairs in zip(features,matches,strict=True):
        for points in keys.keypoints.values():
            if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():
                raise ValueError('Expected finite pixel xy coordinates')
            total_points+=len(points)
        for left,group in pairs.matches.items():
            for right,indices in group.items():
                if indices.ndim!=2 or indices.shape[1]!=2 or not np.issubdtype(indices.dtype,np.integer):
                    raise ValueError('Expected pairs of integer keypoint indices')
                if (indices<0).any() or (indices[:,0]>=len(keys.keypoints[left])).any() or (indices[:,1]>=len(keys.keypoints[right])).any():
                    raise ValueError('Correspondence index outside original keypoints')
                total_matches+=len(indices)
    return dict(keypoints=total_points,matches=total_matches)
