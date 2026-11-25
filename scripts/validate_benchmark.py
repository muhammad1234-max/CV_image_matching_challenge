"""Validate draft integrity; optionally materialize authentic metadata membership.
Never freezes automatically. Existing output files cannot be overwritten.
"""
import argparse,csv,json
from pathlib import Path
from benchmark_contract import digest,canonical_hash,load_records,validate_groups,membership,save_new,SUBMISSION
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/benchmark/imc2025_local_v1'
def validate(metadata=None,output=None):
    b=json.loads((CONFIG/'benchmark.json').read_text());expected=b.pop('benchmark_hash')
    if canonical_hash(b)!=expected:raise ValueError('Benchmark manifest changed')
    b['benchmark_hash']=expected
    for name,h in b['file_hashes'].items():
        if digest(CONFIG/name)!=h:raise ValueError('Changed config: '+name)
    for name,h in b['code_hashes'].items():
        if digest(ROOT/name)!=h:raise ValueError('Changed contract code: '+name)
    for r in json.loads((CONFIG/'provenance.json').read_text())['verified_local_artifacts']:
        if digest(ROOT/r['path'])!=r['sha256']:raise ValueError('Changed evidence: '+r['path'])
    mapping=json.loads((CONFIG/'source_groups.json').read_text())['mapping'];groups={r['dataset']:r['source_group'] for r in mapping}
    def read(p):
        with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
    audit=ROOT/'docs/evidence/kaggle_audit_received'
    counts={r['dataset']:int(r['unique_images']) for r in read(audit/'dataset_groups.csv')}
    edges=read(ROOT/'docs/evidence/contract_audit_rerun_received/dataset_overlap_candidates.csv')
    validate_groups(groups,counts,[(e['dataset_a'],e['dataset_b']) for e in edges])
    result={'status':b['status'],'integrity':'PASS','image_membership':'NOT GENERATED','benchmark_hash':expected}
    if b['status']=='FROZEN':
        for name,h in b['metadata_hashes'].items():
            if digest(ROOT/name)!=h:raise ValueError('Changed original metadata: '+name)
        from finalize_benchmark import review
        records,summary=review()
        saved=json.loads((CONFIG/'image_membership.json').read_text())
        if saved['records']!=records or saved['split_hash']!=b['split_hash'] or summary['split_hash']!=b['split_hash']:raise ValueError('Frozen membership mismatch')
        result.update(image_membership='PASS',split_hash=b['split_hash'])
    if metadata:
        metadata=Path(metadata)
        pins={'train_labels.csv':'f65b20be20cb2cbfd288871c20f9863fd6f052f484a97b236090989af945ea0c','sample_submission.csv':'cce6dddfdf979f0b33e821587d98d6ed0fa97042a0cc0ae4cca1a77a3cf65aab','train_thresholds.csv':'5b92db83d510ab19e66bd1ef4c23865245d75b4a981986e6d8eb1aee06fc9fe2'}
        for name,h in pins.items():
            if digest(metadata/name)!=h:raise ValueError('Original metadata hash mismatch: '+name)
        records=load_records(metadata/'train_labels.csv');load_records(metadata/'sample_submission.csv',SUBMISSION,validate_rotation=False)
        scenes={(r['dataset'],r['scene']):int(r['unique_images']) for r in read(audit/'scene_statistics.csv')}
        assignments=json.loads((CONFIG/'split_proposal.json').read_text())['assignments']
        members=membership(records,groups,assignments,counts,scenes)
        identities={(r['dataset'],r['image']) for r in members}
        for e in edges:
            for side in ['a','b']:
                if (e['dataset_'+side],Path(e['path_'+side]).name) not in identities:raise ValueError('Duplicate evidence identity missing')
        if output is None:raise ValueError('--output required for new membership artifact')
        save_new(output,{'status':'DRAFT_PENDING_FREEZE_REVIEW','records':members,'split_hash':canonical_hash(members),'original_metadata_sha256':pins})
        result.update(image_membership='PASS',split_hash=canonical_hash(members))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metadata');p.add_argument('--output');a=p.parse_args();print(json.dumps(validate(a.metadata,a.output),indent=2))
