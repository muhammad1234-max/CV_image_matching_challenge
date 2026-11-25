"""Administrative scoring after isolated inference."""
import argparse
import json
from pathlib import Path

import pandas as pd
import ns64_adapter
from isolated_staging import CFG, ROOT, digest, ordered


def identity(seed):
    if seed not in (20261111,20261112):
        raise ValueError('Unexpected condition')
    return dict(method_id='ns64_mast3rhybrid_022c', experiment_id=f'ns64_isolated_v1_ETs_{seed}',
                comparison_baseline_id='organizer_baseline_v1_dev_full')


def validate_identity(record,seed):
    if 'baseline_id' in record or any(record.get(k)!=v for k,v in identity(seed).items()):
        raise ValueError('Incorrect attribution')


def finish(private, output, report, receipt):
    admin=json.loads(private.read_text())
    plan=json.loads((CFG/'pilot.json').read_text())
    names=ordered(plan['images'],admin['seed'])
    expected={f'image_{i:04d}.png':name for i,name in enumerate(names)}
    evidence=json.loads((output/'boundary.json').read_text())
    if admin['mapping']!=expected or admin['benchmark']!=plan['benchmark'] or evidence['status']!='PASS':
        raise ValueError('Condition or isolation mismatch')
    if evidence['worker_sha256']!=admin['adapter_hash'] or digest(receipt)!=admin['dependency_receipt_hash']:
        raise ValueError('Worker or receipt changed')
    report.mkdir(parents=True,exist_ok=False)
    pred=ns64_adapter.serialize_predictions(pd.read_csv(output/'opaque.csv'),expected,report)
    evaluation=report/'evaluation.json'
    ns64_adapter.score(ROOT,pred,evaluation)
    scores=json.loads(evaluation.read_text())['scores']
    record=identity(admin['seed'])
    validate_identity(record,admin['seed'])
    registry=ROOT/'experiments/baseline/organizer_baseline_v1_dev_full.json'
    ref=json.loads(registry.read_text())['datasets']['ETs']['scores']
    record.update({k:admin[k] for k in ('benchmark','adapter_hash','config_hash','source_hash','checkpoints',
                                         'permutation_hash','dependency_receipt_hash')})
    record.update(scores=scores, prediction_hash=digest(pred), isolation_hash=digest(output/'boundary.json'),
                  deltas={v:{k:n-ref[v][k] for k,n in values.items()} for v,values in scores.items()})
    (report/'result.json').write_text(json.dumps(record,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for arg in ('private','output','report','receipt'):
        parser.add_argument(arg,type=Path)
    finish(**vars(parser.parse_args()))
