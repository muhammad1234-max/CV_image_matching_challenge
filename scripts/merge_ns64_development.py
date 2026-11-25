"""Independently score pinned worker ZIPs into a new administrative review directory."""
import argparse
import io
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from build_ns64_76 import ROOT, DATA, inventory, ordered, read, sha
from evaluate_ns64_neutral_orders import centers
import subprocess
import sys


def restore(raw, dataset, mapping):
    frame=pd.read_csv(io.BytesIO(raw))
    if sorted(frame.image)!=sorted(mapping) or set(frame.dataset)!={dataset}:
        raise ValueError('Invalid identity coverage')
    centers(frame)
    frame.image=frame.image.map(mapping)
    return frame.sort_values('image')


def aggregate(scores, partitions):
    result={}
    for label,names in partitions.items():
        missing=sorted(set(names)-scores.keys())
        means=None
        if not missing:
            means={variant:{k:sum(scores[n][variant][k] for n in names)/len(names)
                            for k in scores[names[0]][variant]} for variant in scores[names[0]]}
        # A partition or full aggregate requires complete coverage.
        result[label]=dict(missing=missing,complete=not missing,means=means)
    return result


def check_stages(run, rows, checkpoints=None):
    maximum=14400 if run.get('experiment_id')=='ns64_isolated_v1_imc2024_dioscuri_baalshamin_20261111' else 5400
    pins_path=ROOT/'reports/ns64_full/worker_pins.json'
    if pins_path.exists():
        for dataset,pin in read(pins_path).items():
            if run.get('experiment_id')==f'ns64_isolated_v1_{dataset}_20261111' and run.get('worker_sha256')==pin['worker_sha256']:
                maximum=14400
    if run['status']!='PASS' or not 0<run['seconds']<=run['budget_seconds']<=maximum:
        raise ValueError('Incomplete or over-budget run')
    if not rows or any(r['status']!='PASS' for r in rows):
        raise ValueError('Incomplete stages')
    if checkpoints is not None and (len(rows)!=len(checkpoints) or any(checkpoints.get(r['stage'])!=r for r in rows)):
        raise ValueError('Stage checkpoint mismatch')


def merge(index,output):
    if output.exists():
        raise FileExistsError(output)
    bench=read(ROOT/'configs/benchmark/imc2025_local_v1/benchmark.json')
    partitions={k:bench['evaluation_datasets'][k] for k in ('dev_a','dev_b')}
    partitions['full']=partitions['dev_a']+partitions['dev_b']
    prep=read(ROOT/'reports/phase3c8/preparation.json')
    assert all(bench[k]==v for k,v in prep['benchmark'].items())
    subprocess.run([sys.executable,str(ROOT/'scripts/validate_benchmark.py')],check=True)
    scores={}
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        staged=Path(tmp)/'review'
        staged.mkdir()
        entries=read(index)
        if len({v['dataset'] for v in entries})!=len(entries):
            raise ValueError('Duplicate dataset')
        for entry in entries:
            data=entry['dataset']
            if data in scores or data not in partitions['full']:
                raise ValueError('Duplicate or unauthorized dataset')
            raw=(ROOT/entry['archive']).read_bytes()
            if sha(raw)!=entry['sha256']:
                raise ValueError('Archive digest mismatch')
            with ZipFile(io.BytesIO(raw)) as z:
                assert z.testzip() is None and len(set(z.namelist()))==len(z.namelist())
                get=lambda k:json.loads(z.read(k+'.json'))
                run,i,e=get('run'),get('identity'),get('environment')
                rows=[json.loads(line) for line in z.read('stages.jsonl').splitlines()]
                checks={n[11:-5]:json.loads(z.read(n)) for n in z.namelist() if n.startswith('checkpoint_')}
                check_stages(run,rows,checks if data in (DATA,'imc2024_dioscuri_baalshamin') or checks else None)
                assert run['worker_sha256']==entry['worker_sha256']==e['probe_sha256']
                assert run['experiment_id']==i['experiment_id']==f'ns64_isolated_v1_{data}_20261111'
                assert i['seed']==run['seed']==20261111 and i['comparison_baseline_id']=='organizer_baseline_v1_dev_full' and i['method_id']=='ns64_mast3rhybrid_022c'
                assert i['source_revision']==prep['source_revision'] and i['config_sha256']==prep['config_sha256']
                if data in (DATA,'imc2024_dioscuri_baalshamin') or 'benchmark' in i:
                    assert i['benchmark']==prep['benchmark']
                assert e['assets']==prep['checkpoint_hashes'] and e['source_inventory_hash']==prep['source_inventory_hash']
                assert e['status']=='CLEAN_VM_PRECHECK_PASS' and get('boundary')['status'] in ('PASS','POST_STAGING_CHECKS_PASS')
                images=inventory(data)
                names=ordered(images,20261111)
                assert sha('\n'.join(images[n] for n in names).encode())==i['permutation_sha256']
                mapping={f'image_{j:04d}.png':n for j,n in enumerate(names)}
                csv=z.read('opaque.csv')
                assert sha(csv)==get('prediction')['sha256']
                frame=restore(csv,data,mapping)
            directory=staged/data
            directory.mkdir()
            predictions=directory/'predictions.csv'
            frame.to_csv(predictions,index=False)
            evaluation=directory/'evaluation.json'
            subprocess.run([sys.executable,str(ROOT/'src/baselines/development/evaluate.py'),data,str(predictions),str(evaluation)],check=True)
            scores[data]=read(evaluation)['scores']
        (staged/'coverage.json').write_text(json.dumps(aggregate(scores,partitions),indent=2))
        (staged/'archive_index.json').write_bytes(index.read_bytes())
        staged.rename(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('index',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    merge(args.index,args.output)

