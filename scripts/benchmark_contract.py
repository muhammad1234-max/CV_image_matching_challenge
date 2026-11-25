"""Strict benchmark membership and experiment contracts; no model execution."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

TRAIN=['dataset','scene','image','rotation_matrix','translation_vector']
SUBMISSION=['image_id']+TRAIN

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def load_records(path,fields=TRAIN,validate_rotation=True):
    records=[];seen=set();ids=set()
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        r=csv.DictReader(f)
        if r.fieldnames!=fields:raise ValueError(f'Expected columns in exact order: {fields}')
        for n,row in enumerate(r,2):
            if None in row or any(v is None for v in row.values()):raise ValueError(f'Malformed CSV row {n}')
            key=(row['dataset'],row['image'])
            if not all(v.strip() for v in key) or not row['scene'].strip():raise ValueError(f'Empty identity/scene at row {n}')
            if key in seen:raise ValueError(f'Duplicate dataset/image at row {n}')
            seen.add(key)
            if fields==SUBMISSION:
                if not row['image_id'].strip() or row['image_id'] in ids:raise ValueError(f'Empty/duplicate image_id at row {n}')
                ids.add(row['image_id'])
            try:
                rvals=[float(x) for x in row['rotation_matrix'].split(';')];tvals=[float(x) for x in row['translation_vector'].split(';')]
            except ValueError as exc:raise ValueError(f'Non-numeric pose at row {n}') from exc
            if len(rvals)!=9 or len(tvals)!=3:raise ValueError(f'Pose vector dimensions at row {n}')
            vals=rvals+tvals
            if not (all(math.isfinite(x) for x in vals) or all(math.isnan(x) for x in vals)):raise ValueError(f'Mixed/nonfinite pose at row {n}')
            if validate_rotation and all(math.isfinite(x) for x in vals):
                R=[rvals[i:i+3] for i in (0,3,6)]
                det=R[0][0]*(R[1][1]*R[2][2]-R[1][2]*R[2][1])-R[0][1]*(R[1][0]*R[2][2]-R[1][2]*R[2][0])+R[0][2]*(R[1][0]*R[2][1]-R[1][1]*R[2][0])
                if abs(det-1)>1e-3 or any(abs(sum(R[k][i]*R[k][j] for k in range(3))-(i==j))>1e-3 for i in range(3) for j in range(3)):raise ValueError(f'Invalid rotation at row {n}')
            records.append(row)
    if not records:raise ValueError('Empty records')
    return records

def validate_groups(groups,expected_datasets,duplicate_edges):
    if set(groups)!=set(expected_datasets):raise ValueError('Incomplete or unexpected dataset source-group assignment')
    for a,b in duplicate_edges:
        if groups[a]!=groups[b]:raise ValueError(f'Known duplicate leakage between {a} and {b}')

def membership(records,groups,assignments,expected_counts=None,expected_scenes=None):
    if set(assignments)!=set(groups.values()):raise ValueError('Every source group needs exactly one assignment')
    if not set(assignments.values())<= {'dev_a','dev_b','confirmation'}:raise ValueError('Unknown partition')
    keys=set();counts=Counter();scene_counts=Counter();result=[]
    for r in records:
        key=r['dataset'],r['image']
        if key in keys:raise ValueError('Duplicate image identity')
        keys.add(key)
        if r['dataset'] not in groups:raise ValueError('Unknown dataset')
        counts[r['dataset']]+=1;scene_counts[r['dataset'],r['scene']]+=1
        partition=assignments[groups[r['dataset']]]
        result.append({k:r[k] for k in ['dataset','scene','image']}|{'source_group':groups[r['dataset']],'partition':partition,
            'fold_a_role':'excluded_confirmation' if partition=='confirmation' else 'evaluation' if partition=='dev_a' else 'training',
            'fold_b_role':'excluded_confirmation' if partition=='confirmation' else 'evaluation' if partition=='dev_b' else 'training'})
    if expected_counts is not None and dict(counts)!=expected_counts:raise ValueError('Image/dataset coverage differs from audited counts')
    if expected_scenes is not None and dict(scene_counts)!=expected_scenes:raise ValueError('Scene/outlier coverage differs from audited counts')
    return sorted(result,key=lambda x:(x['dataset'],x['image']))

def save_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def validate_submission_identity(sample,predicted):
    # Sample poses are arbitrary placeholders, not physical camera predictions.
    a=load_records(sample,SUBMISSION,validate_rotation=False);b=load_records(predicted,SUBMISSION)
    identity=lambda r:(r['image_id'],r['dataset'],r['image'])
    if list(map(identity,a))!=list(map(identity,b)):raise ValueError('Submission IDs/order/coverage differ from actual sample')

def validate_experiment(experiment,benchmark):
    required=['benchmark_version','benchmark_hash','git_commit','code_snapshot_sha256','dataset_inventory_hash','split_hash','evaluator_hash',
        'threshold_hash','model_config_hash','weights_hashes','environment','seeds','runtime_seconds','cpu_peak_memory_bytes','gpu_peak_memory_bytes',
        'per_dataset','submission_format_checks','role','experiment_id','baseline_reference_id','evaluation_partition']
    missing=[k for k in required if k not in experiment]
    if missing:raise ValueError(f'Missing experiment fields: {missing}')
    if benchmark['status']!='FROZEN':raise ValueError('Experiments require a frozen benchmark')
    for key in ['benchmark_hash','code_snapshot_sha256','dataset_inventory_hash','split_hash','evaluator_hash','threshold_hash','model_config_hash']:
        value=experiment[key]
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError(f'Missing/invalid SHA256: {key}')
    for key in ['git_commit','experiment_id','environment','seeds','submission_format_checks']:
        if not experiment[key]:raise ValueError(f'Empty required field: {key}')
    if not isinstance(experiment['weights_hashes'],dict) or not experiment['weights_hashes']:raise ValueError('Record weight hashes or explicit no-weights reason')
    for name,h in experiment['weights_hashes'].items():
        if name=='none_reason' and isinstance(h,str) and h:continue
        if not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise ValueError('Invalid weight SHA256')
    if experiment['evaluation_partition']=='confirmation' and not experiment.get('confirmation_release_id'):raise ValueError('Confirmation requires a recorded supervisor release after selection lock')
    for key in ['benchmark_version','benchmark_hash','dataset_inventory_hash','split_hash','evaluator_hash','threshold_hash']:
        if experiment[key]!=benchmark[key]:raise ValueError(f'Changed benchmark component: {key}; rerun all compared methods')
    if experiment['role'] not in ('baseline','improvement'):raise ValueError('Unknown experiment role')
    if experiment['role']=='improvement' and not experiment['baseline_reference_id']:raise ValueError('Improvement requires immutable baseline reference')
    if not isinstance(experiment['runtime_seconds'],(int,float)) or not math.isfinite(experiment['runtime_seconds']) or experiment['runtime_seconds']<0:raise ValueError('Invalid runtime')
    if experiment['evaluation_partition'] not in ('dev_a','dev_b','confirmation'):raise ValueError('Invalid evaluation partition')
    expected=benchmark['evaluation_datasets'][experiment['evaluation_partition']]
    if set(experiment['per_dataset'])!=set(expected):raise ValueError('Incomplete per-dataset results')
    for row in experiment['per_dataset'].values():
        for key in ('maa','clusterness','combined_score','registration_coverage','failure_count'):
            if key not in row:raise ValueError(f'Missing metric: {key}')
        for key in ('maa','clusterness','combined_score','registration_coverage'):
            v=row[key]
            if not isinstance(v,(float,int)) or not math.isfinite(v) or not 0<=v<=1:raise ValueError('Metrics must be finite fractions [0,1]')
        if not isinstance(row['failure_count'],int) or row['failure_count']<0:raise ValueError('Invalid failure count')
    for key in ('cpu_peak_memory_bytes','gpu_peak_memory_bytes'):
        v=experiment[key]
        if not isinstance(v,dict) or 'measured' not in v or 'value' not in v or 'reason' not in v:raise ValueError('Memory measurements require measured/value/reason')
        if v['measured'] and (not isinstance(v['value'],int) or v['value']<0):raise ValueError('Invalid measured memory')
    return True

def paired_comparison(baseline,improvement,benchmark):
    validate_experiment(baseline,benchmark);validate_experiment(improvement,benchmark)
    if baseline['role']!='baseline' or improvement['role']!='improvement':raise ValueError('Expected baseline and improvement')
    if improvement['baseline_reference_id']!=baseline['experiment_id']:raise ValueError('Baseline reference mismatch')
    if baseline['evaluation_partition']!=improvement['evaluation_partition']:raise ValueError('Different evaluation partitions')
    if baseline['seeds']!=improvement['seeds'] or baseline['environment']!=improvement['environment']:raise ValueError('Paired conditions differ')
    return {d:improvement['per_dataset'][d]['combined_score']-r['combined_score'] for d,r in baseline['per_dataset'].items()}
