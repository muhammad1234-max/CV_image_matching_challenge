"""Review authentic metadata and finalize the first benchmark, never replace a freeze."""
import argparse,csv,json,math,platform
from collections import Counter
from pathlib import Path
import numpy as np
from benchmark_contract import digest,canonical_hash,load_records,membership,save_new,SUBMISSION
ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'configs/benchmark/imc2025_local_v1'
M=ROOT/'docs/evidence/benchmark_metadata_received'
PINS={'train_labels.csv':'f65b20be20cb2cbfd288871c20f9863fd6f052f484a97b236090989af945ea0c','sample_submission.csv':'cce6dddfdf979f0b33e821587d98d6ed0fa97042a0cc0ae4cca1a77a3cf65aab','train_thresholds.csv':'5b92db83d510ab19e66bd1ef4c23865245d75b4a981986e6d8eb1aee06fc9fe2'}
def read_csv(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def review():
    for name,h in PINS.items():
        if digest(M/name)!=h:raise ValueError('Original metadata hash changed: '+name)
    records=load_records(M/'train_labels.csv');sample=load_records(M/'sample_submission.csv',SUBMISSION,validate_rotation=False)
    mapping=json.loads((C/'source_groups.json').read_text())['mapping'];groups={r['dataset']:r['source_group'] for r in mapping}
    proposal=json.loads((C/'split_proposal.json').read_text())
    audit=ROOT/'docs/evidence/kaggle_audit_received'
    counts={r['dataset']:int(r['unique_images']) for r in read_csv(audit/'dataset_groups.csv')}
    scenes={(r['dataset'],r['scene']):int(r['unique_images']) for r in read_csv(audit/'scene_statistics.csv')}
    members=membership(records,groups,proposal['assignments'],counts,scenes)
    inventory=json.loads((audit/'dataset_inventory.json').read_text())
    paths={r['path'] for r in inventory['records']}
    for r in records:
        if 'train/'+r['dataset']+'/'+r['image'] not in paths:raise ValueError('Label absent from image inventory')
        nan=all(math.isnan(float(v)) for k in ['rotation_matrix','translation_vector'] for v in r[k].split(';'))
        if nan != (r['scene']=='outliers'):raise ValueError('Pose/outlier contingency mismatch')
    edges=read_csv(ROOT/'docs/evidence/contract_audit_rerun_received/dataset_overlap_candidates.csv')
    by_key={(r['dataset'],r['image']):r for r in members}
    if len(edges)!=8:raise ValueError('Unexpected duplicate evidence count')
    for e in edges:
        ends=[by_key[(e['dataset_'+s],Path(e['path_'+s]).name)] for s in ['a','b']]
        if ends[0]['source_group']!=ends[1]['source_group'] or ends[0]['partition']!=ends[1]['partition']:raise ValueError('Exact duplicate crosses boundary')
    ts=read_csv(M/'train_thresholds.csv');expected=json.loads((C/'thresholds.json').read_text())['scenes']
    actual={(r['dataset'],r['scene']):r['thresholds'] for r in ts}
    if len(ts)!=30 or len(actual)!=30 or actual!={(r['dataset'],r['scene']):r['raw'] for r in expected}:raise ValueError('Original threshold representations differ')
    if set(actual)!={k for k in scenes if k[1]!='outliers'}:raise ValueError('Threshold scene coverage mismatch')
    balances={}
    for part in ['dev_a','dev_b','confirmation']:
        subset=[r for r in members if r['partition']==part]
        balances[part]={'datasets':sorted({r['dataset'] for r in subset}),'groups':len({r['source_group'] for r in subset}),'images':len(subset),'scenes':len({(r['dataset'],r['scene']) for r in subset if r['scene']!='outliers'}),'outliers':sum(r['scene']=='outliers' for r in subset)}
        if balances[part]!=proposal['balance'][part]:raise ValueError('Actual balance differs from proposal')
    summary={'status':'PASS','records':len(records),'non_outlier_scenes':len(actual),'outliers':sum(r['scene']=='outliers' for r in records),'finite_poses':sum(r['scene']!='outliers' for r in records),'sample_rows':len(sample),'sample_identity_order_hash':canonical_hash([[r['image_id'],r['dataset'],r['image']] for r in sample]),'split_hash':canonical_hash(members),'exact_duplicate_pairs_preserved':len(edges),'all_training_identities_in_returned_inventory':True,'threshold_raw_strings_match':True,'balance':balances,'original_metadata_sha256':PINS,'image_bytes_independently_reverified':False}
    return members,summary
def freeze():
    p=C/'benchmark.json';b=json.loads(p.read_text())
    if b['status']!='DRAFT':raise ValueError('Already frozen; use a new benchmark version for changes')
    from validate_benchmark import validate
    validate()
    members,summary=review()
    env={'purpose':'CPU reference evaluator verification; not the Phase 3 GPU environment','python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),'baseline_environment_status':'NOT REGISTERED'}
    save_new(C/'image_membership.json',{'status':'FROZEN','records':members,'split_hash':summary['split_hash']})
    save_new(C/'metadata_validation.json',summary);save_new(C/'evaluator_environment.json',env)
    save_new(C/'freeze_decision.json',{'decision':'Accept proposed two-fold plus protected confirmation design with documented small-group and outlier-imbalance limitations','authority':'Prompt 05 authorizes freeze after evidence and hard constraints pass','baseline_registered':False,'phase3_authorized':False,'previous_draft_hash':b['benchmark_hash'],'threshold_source_available':True,'confirmation_model_selection_allowed':False})
    save_new(M/'prefreeze_benchmark.json',b)
    b.update(status='FROZEN',blockers=[],split_hash=summary['split_hash'],baseline_status='NOT REGISTERED',phase3_status='AWAITING SUPERVISOR APPROVAL',metadata_hashes={str((M/k).relative_to(ROOT)).replace('\\','/'):v for k,v in PINS.items()})
    for name in ['image_membership.json','metadata_validation.json','evaluator_environment.json','freeze_decision.json']:b['file_hashes'][name]=digest(C/name)
    b.pop('benchmark_hash');b['benchmark_hash']=canonical_hash(b)
    # Only the checked DRAFT transitions; no frozen manifest is overwritten.
    p.write_text(json.dumps(b,indent=2)+'\n',encoding='utf-8')
    return b
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    print(json.dumps(freeze() if a.freeze else review()[1],indent=2))
