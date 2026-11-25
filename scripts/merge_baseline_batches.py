"""Validate exported development batches and independently recompute every score."""
import argparse,csv,json,sys,tempfile,zipfile,math,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/baselines/development'))
from common import *
from evaluate import evaluate
from benchmark_contract import validate_experiment

def fold_records(results,identity,environments):
    cfg=read(CONFIG/'full_development.json');benchmark=read(SNAPSHOT/'configs/benchmark/imc2025_local_v1/benchmark.json');records={}
    for partition in ['dev_a','dev_b']:
        selected={d:r for d,r in results.items() if r['partition']==partition}
        record={k:benchmark[k] for k in ['benchmark_version','benchmark_hash','dataset_inventory_hash','split_hash','evaluator_hash','threshold_hash']}
        record.update(git_commit='UNVERSIONED_CONTENT_SNAPSHOT',code_snapshot_sha256=identity['implementation_hash'],model_config_hash=fingerprint(cfg['model_config']),weights_hashes=read(CONFIG/'full_checkpoint_lock.json'),environment=read(CONFIG/'full_environment_lock.json'),seeds={'baseline':cfg['model_config']['seed']},runtime_seconds=sum(r['successful_stage_seconds'] for r in selected.values()),cpu_peak_memory_bytes={'measured':True,'value':max(r['cpu_peak_rss_bytes'] for r in selected.values()),'reason':'maximum worker ru_maxrss'},gpu_peak_memory_bytes={'measured':True,'value':max(r['gpu_peak_allocated_bytes'] for r in selected.values()),'reason':'maximum PyTorch allocated bytes; system samples in session evidence'},per_dataset={d:dict(r['scores']['participant'],registration_coverage=r['registration_coverage'],failure_count=r['failure_count']) for d,r in selected.items()},submission_format_checks={'strict_prediction_coverage_and_poses':True,'independent_recomputation':True},role='baseline',experiment_id=cfg['registration_id']+'_'+partition,baseline_reference_id=None,evaluation_partition=partition)
        validate_experiment(record,benchmark);records[partition]=record
    return records

def inspect_batches(paths,identity):
    cfg=read(CONFIG/'full_development.json');results={};sources={};artifacts={};environments=[]
    envhash=validate_environment(read(CONFIG/'full_environment_lock.json')['observed_full_environment'],read(CONFIG/'full_environment_lock.json'))
    for path in paths:
        with zipfile.ZipFile(path) as z:
            manifest=verify_zip(z)
            if manifest.get('kind')!='report' or manifest['implementation_hash']!=identity['implementation_hash']:raise ValueError('Wrong archive kind or implementation')
            archivehash=sha(path)
            if archivehash in sources:raise ValueError('Duplicate batch archive')
            sources[archivehash]=str(path)
            for name in manifest['files']:
                if name.startswith('datasets/'):inputs(name.split('/')[1])
                if name.startswith('session_') and name.endswith('.json'):
                    session=json.loads(z.read(name))
                    if 'runtime' in session:
                        validate_environment(session['runtime'],read(CONFIG/'full_environment_lock.json'));environments.append(session)
            for name in sorted(n for n in manifest['files'] if n.endswith('/complete.json')):
                prefix=name.rsplit('/',1)[0]+'/';dataset=prefix.split('/')[1]
                if dataset in results:raise ValueError('Duplicate/conflicting dataset result: '+dataset)
                complete=json.loads(z.read(name))
                required={'result.json','predictions.csv','evaluation.json','identity_mapping.json','input_hashes.json'}
                if set(complete['files'])!=required or complete['implementation_hash']!=identity['implementation_hash']:raise ValueError('Incomplete result manifest')
                for n,h in complete['files'].items():
                    if manifest['files'].get(prefix+n)!=h:raise ValueError('Completed artifact hash mismatch')
                result=json.loads(z.read(prefix+'result.json'));mapping=json.loads(z.read(prefix+'identity_mapping.json'));inventory=json.loads(z.read(prefix+'input_hashes.json'))
                entry=next(p for p in cfg['plan'] if p['dataset']==dataset)
                expected={'dataset':dataset,'partition':entry['partition'],'images':entry['images'],'benchmark_hash':BENCHMARK,'split_hash':SPLIT,'implementation_hash':identity['implementation_hash'],'model_config_hash':fingerprint(cfg['model_config']),'checkpoint_hash':fingerprint(read(CONFIG/'full_checkpoint_lock.json')),'environment_hash':envhash,'membership_hash':fingerprint(inputs(dataset)),'input_inventory_hash':fingerprint(inventory)}
                if mapping!=inputs(dataset) or any(result.get(k)!=v for k,v in expected.items()):raise ValueError('Result identity/config/membership mismatch')
                if set(inventory)!={r['opaque'] for r in mapping}:raise ValueError('Input inventory coverage mismatch')
                for values in inventory.values():
                    if set(values)!={'original_sha256','input_sha256'} or any(len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in values.values()):raise ValueError('Invalid image hashes')
                with tempfile.TemporaryDirectory() as tmp:
                    pred=Path(tmp)/'predictions.csv';pred.write_bytes(z.read(prefix+'predictions.csv'));fresh=evaluate(dataset,pred)
                    with pred.open() as f:actual_rows=list(csv.DictReader(f))
                    assignments=json.loads(z.read(prefix+'H/assignments.json'))
                    if set(assignments)!={r['opaque'] for r in mapping}:raise ValueError('Inference assignment coverage mismatch')
                    serialized=serialize(mapping,assignments);serialized.sort(key=lambda r:(r['dataset'],r['image']))
                    if actual_rows!=serialized:raise ValueError('Predictions differ from recorded inference assignments')
                stored=json.loads(z.read(prefix+'evaluation.json'))
                for variant in fresh['scores']:
                    for metric,value in fresh['scores'][variant].items():
                        if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid score')
                        if any(not math.isclose(value,x['scores'][variant][metric],rel_tol=0,abs_tol=1e-12) for x in [stored,result]):raise ValueError('Independent score recomputation differs')
                if any(result.get(k)!=v or stored.get(k)!=v for k,v in fresh.items() if k!='scores'):raise ValueError('Diagnostics differ')
                registered=entry['images']-fresh['predicted_outliers']
                if result['registered_images']!=registered or result['registration_coverage']!=registered/entry['images']:raise ValueError('Registration count mismatch')
                if result['failure_count']!=int(registered==0) or result['status']!=('SCORED_ALGORITHMIC_FAILURE' if registered==0 else 'COMPLETE'):raise ValueError('Failure outcome mismatch')
                for stage in 'BCDEFGH':
                    stage_result=json.loads(z.read(prefix+stage+'/result.json'))
                    if stage_result['status']!='PASS' or stage_result['stage']!=stage:raise ValueError('Incomplete inference stages')
                    if result['runtime_by_stage'][stage]!=stage_result['runtime_seconds']:raise ValueError('Runtime mismatch')
                for stage,metric,field in [('C','selected_pairs','selected_pairs'),('E','retained_pairs','retained_match_pairs'),('F','verified_pairs','verified_pairs'),('G','models','models'),('G','registered_unique','registered_images')]:
                    if json.loads(z.read(prefix+stage+'/result.json'))['metrics'][metric]!=result[field]:raise ValueError('Inference count mismatch: '+field)
                results[dataset]=result
            # Preserve full lightweight evidence with an archive-specific prefix.
            artifacts.update({archivehash+'/'+n:z.read(n) for n in manifest['files']})
    if results and not environments:raise ValueError('Missing execution environment evidence')
    return results,sources,artifacts,environments

def diagnostics(output,results):
    fields=['dataset','images','registered_images','registration_coverage','true_scene_count','predicted_clusters','true_outliers','predicted_outliers','models','selected_pairs','retained_match_pairs','verified_pairs','successful_stage_seconds','failure_count','primary_maa','primary_clusterness','primary_combined','sensitivity_combined','score_delta','false_outliers','missed_outliers','merged_clusters','split_scenes']
    with (output/'dataset_metrics.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for dataset,r in sorted(results.items()):
            row={k:r[k] for k in fields if k in r};a=r['scores']['participant'];b=r['scores']['prose_sensitivity'];d=r['diagnostics']
            row.update(primary_maa=a['maa'],primary_clusterness=a['clusterness'],primary_combined=a['combined_score'],sensitivity_combined=b['combined_score'],score_delta=b['combined_score']-a['combined_score'],false_outliers=d['false_outliers'],missed_outliers=d['missed_outliers'],merged_clusters=len(d['merged_clusters']),split_scenes=len(d['split_scenes']));writer.writerow(row)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="'+str(90+len(results)*60)+'"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="13"><text x="20" y="24">Development baseline: primary (blue), sensitivity (orange), registration coverage (green); fractions [0,1]</text>']
    for i,(name,r) in enumerate(sorted(results.items())):
        y=65+i*60;svg.append(f'<text x="10" y="{y}">{html.escape(name)}</text>')
        for j,(value,color) in enumerate([(r['scores']['participant']['combined_score'],'#2166ac'),(r['scores']['prose_sensitivity']['combined_score'],'#b35806'),(r['registration_coverage'],'#1b7837')]):svg.append(f'<rect x="350" y="{y-12+j*14}" width="{500*value}" height="11" fill="{color}"/><text x="{855}" y="{y+j*14}">{value:.4f}</text>')
    svg.append('</g></svg>');(output/'scores_and_coverage.svg').write_text(''.join(svg),encoding='utf-8')

def merge(paths,output,register=False):
    identity=verify_implementation();cfg=read(CONFIG/'full_development.json')
    results,sources,artifacts,environments=inspect_batches(paths,identity);summary=aggregate(results,cfg['plan'])
    registry=ROOT/'experiments/baseline'/ (cfg['registration_id']+'.json')
    if register and summary['missing_datasets']:raise ValueError('Missing datasets; full baseline registration prohibited')
    if register and registry.exists():raise FileExistsError('Baseline registration is immutable: '+str(registry))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for name,raw in artifacts.items():
        p=output/'evidence'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    report={'registration_status':'NOT REGISTERED','implementation':identity,'summary':summary,'datasets':results,'source_archives':sources,'sessions':environments,'fidelity':'Exact successful smoke inference workers/settings; generalized dataset orchestration, identity-sorted scoring and strict runtime/checkpoint locks. No hidden-scorer equivalence claimed.','runtime_note':'successful_stage_seconds excludes preflight, preprocessing, evaluation and retries; session wall_seconds records all session work before archive export','reproducibility_status':'source/config/checkpoints pinned; GPU numerical repeatability not guaranteed; local scores independently recomputed'}
    diagnostics(output,results)
    if not summary['missing_datasets']:report['fold_experiment_records']=fold_records(results,identity,environments)
    if register:
        report['registration_status']='REGISTERED';report['experiment_id']=cfg['registration_id'];exclusive(registry,report)
    exclusive(output/'report.json',report)
    if not summary['missing_datasets']:
        write_archive(output/'imc_baseline_dev_full_report.zip',{p.relative_to(output).as_posix():p for p in output.rglob('*') if p.is_file()},{'kind':'full_report','implementation_hash':identity['implementation_hash']})
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archives',nargs='+');p.add_argument('--output',required=True);p.add_argument('--register',action='store_true');a=p.parse_args()
    report=merge(a.archives,a.output,a.register);print(json.dumps(report['summary'],indent=2));print(report['registration_status'])
