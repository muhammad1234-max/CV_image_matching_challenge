"""One-time, exclusive creation of Phase 2D metadata-only draft."""
import csv,json
from pathlib import Path
from benchmark_contract import digest,canonical_hash,save_new
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'configs/benchmark/imc2025_local_v1'
E=ROOT/'docs/evidence'
def rows(p):
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    provenance=[]
    for directory in ['kaggle_audit_received','contract_audit_received','contract_audit_rerun_received']:
        p=E/directory
        for entry in json.loads((p/'archive_manifest.json').read_text())['included']:
            f=p/entry['file']
            assert digest(f)==entry['sha256'] and f.stat().st_size==entry['bytes'],str(f)
            provenance.append({'path':str(f.relative_to(ROOT)).replace('\\','/'),**entry})
    datasets=rows(E/'kaggle_audit_received/dataset_groups.csv')
    a={'imc2023_heritage','imc2024_dioscuri_baalshamin','amy_gardens','imc2023_haiper','stairs','imc2023_theather_imc2024_church'}
    c={'imc2024_lizard_pond','pt_piazzasanmarco_grandplace'}
    groups={r['dataset']:('heritage_dioscuri' if r['dataset'] in {'imc2023_heritage','imc2024_dioscuri_baalshamin'} else r['dataset']) for r in datasets}
    assignment={g:('dev_a' if d in a else 'confirmation' if d in c else 'dev_b') for d,g in groups.items()}
    mapping=[{'dataset':d,'source_group':g,'partition':assignment[g],'reason':'Eight reported byte-identical cross-dataset pairs' if g=='heritage_dioscuri' else 'No further relationship established; candidate independence only','evidence':'docs/evidence/contract_audit_rerun_received/source_provenance_review.md'} for d,g in groups.items()]
    balance={}
    for p in ['dev_a','dev_b','confirmation']:
        subset=[r for r in datasets if assignment[groups[r['dataset']]]==p]
        balance[p]={'datasets':[r['dataset'] for r in subset],'groups':len({groups[r['dataset']] for r in subset}),'images':sum(int(r['unique_images']) for r in subset),'scenes':sum(int(r['scenes']) for r in subset),'outliers':sum(int(r['outliers']) for r in subset)}
    ts=[{'dataset':r['dataset'],'scene':r['scene'],'raw':r['raw_thresholds'],'values':[float(x) for x in r['raw_thresholds'].split(';')]} for r in rows(E/'contract_audit_rerun_received/thresholds_raw_and_parsed.csv')]
    files={'source_groups.json':{'mapping':mapping,'candidate_independence_only':True},'split_proposal.json':{'status':'DRAFT','seed':20261111,'strategy':'fixed metadata-informed named allocation; no score inspection or random percentages','assignments':assignment,'balance':balance,'training':{'fold_a':balance['dev_b']['datasets'],'fold_b':balance['dev_a']['datasets']},'confirmation_for_model_selection':False},'thresholds.json':{'source_csv_sha256':'5b92db83d510ab19e66bd1ef4c23865245d75b4a981986e6d8eb1aee06fc9fe2','raw_source_locally_available':False,'skip_top_thresholds':2,'scenes':ts},'provenance.json':{'verified_local_artifacts':provenance,'remote_findings_independently_reverified':False},'evaluator.json':{'source_url':'https://raw.githubusercontent.com/ns-rokuyon/kaggle-image-matching-challenge-2025/main/utils/imc25/metric.py','source_revision':None,'sha256':'c7ae94306ba0ae4ce9c00e9abc7e15bd99f3091fa063b5826b0dd567a6c3f261','primary':'participant overlap-first','sensitivity':'mAA-first, purity tie-break, stable index final tie','adapter':'scripts/evaluate_local_reference.py','dependencies':['Python','NumPy'],'inl_cf':0,'strict_cf':-1,'skip_top_thresholds':2,'to_dec':3,'exact_kaggle_parity':False},'experiment_template.json':{k:None for k in ['benchmark_version','benchmark_hash','git_commit','code_snapshot_sha256','dataset_inventory_hash','split_hash','evaluator_hash','threshold_hash','model_config_hash','weights_hashes','environment','seeds','runtime_seconds','cpu_peak_memory_bytes','gpu_peak_memory_bytes','per_dataset','submission_format_checks','role','experiment_id','baseline_reference_id','evaluation_partition']}}
    for name,value in files.items():save_new(OUT/name,value)
    b={'benchmark_version':'imc2025_local_v1','status':'DRAFT','seed':20261111,'blockers':['Original train_labels.csv unavailable locally','Original sample_submission.csv and row identity mapping unavailable locally','Original train_thresholds.csv bytes unavailable locally','Full image membership, image-level checks and freeze review pending'],'dataset_inventory_hash':digest(E/'kaggle_audit_received/dataset_inventory.json'),'split_hash':None,'proposal_hash':canonical_hash(files['split_proposal.json']),'evaluator_hash':files['evaluator.json']['sha256'],'threshold_hash':digest(OUT/'thresholds.json'),'evaluation_datasets':{p:x['datasets'] for p,x in balance.items()},'file_hashes':{name:digest(OUT/name) for name in files},'code_hashes':{name:digest(ROOT/name) for name in ['scripts/benchmark_contract.py','scripts/evaluate_local_reference.py']}}
    b['benchmark_hash']=canonical_hash(b);save_new(OUT/'benchmark.json',b)
    print(json.dumps({'status':b['status'],'verified_artifacts':len(provenance),'benchmark_hash':b['benchmark_hash'],'balance':balance},indent=2))
if __name__=='__main__':main()
