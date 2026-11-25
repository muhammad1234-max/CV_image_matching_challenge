"""One ETs attempt in a new Linux workspace after reviewing a dependency receipt."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import ns64_adapter as adapter

ROOT=Path(__file__).resolve().parents[3]
CFG=ROOT/'configs/baselines/top10_reproduction_v1'


def approved_paths(receipt, manifest):
    with zipfile.ZipFile(receipt) as z:
        record=json.loads(z.read('result.json'))
        if json.loads(z.read('manifest.json'))!=manifest or record['status']!='PASS':
            raise ValueError('Dependency receipt is incomplete or belongs to another manifest')
    if not all(v['status']=='PASS' for v in record['checks'].values()):
        raise ValueError('Dependency checks include failed results')
    paths={}
    for key,asset in manifest['assets'].items():
        matches=[Path(f['path']) for f in record['assets'][key]['files'] if f['sha256']==asset['sha256']]
        if len(matches)!=1 or record['checks']['load_'+key]['status']!='PASS':
            raise ValueError('Missing or ambiguous verified '+key)
        paths[asset['alias']]=matches[0].parent if key=='dinov2' else matches[0]
    adapter.check_model_paths(paths,manifest['assets'])
    return paths


def bounded(command,seconds,cwd,log):
    if seconds<=0:raise TimeoutError('ETs budget exhausted')
    with log.open('w') as stream:
        proc=subprocess.Popen(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait()
            raise TimeoutError('ETs budget exhausted')
        if code:raise RuntimeError(f'Worker returned {code}; inspect {log.name}')


def main(args):
    plan=json.loads((CFG/'pilot.json').read_text())
    assets=json.loads((CFG/'assets_phase3c1.json').read_text())
    if args.worker:
        paths=approved_paths(args.receipt,assets)
        adapter.infer(plan,args.images,args.work/'inference',args.work/'published_source',paths)
        return
    args.work.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    result={'status':'BLOCKED','dataset':'ETs','baseline_id':'organizer_baseline_v1_dev_full',
            'benchmark':plan['benchmark'],'source_revision':plan['revision'],'configuration_sha256':plan['config_sha256'],
            'dependency_receipt_sha256':adapter.digest(args.receipt),'budget_seconds':1800}
    stage='dependencies'
    try:
        approved_paths(args.receipt,assets)
        stage='inference'
        bounded([sys.executable,__file__,'--worker','--images',str(args.images),'--receipt',str(args.receipt),'--work',str(args.work)],1800-(time.monotonic()-start),args.work,args.work/'inference.log')
        stage='scoring'
        pred=args.work/'inference/predictions.csv'
        out=args.work/'evaluation.json'
        bounded([sys.executable,str(ROOT/'src/baselines/development/evaluate.py'),'ETs',str(pred),str(out)],1800-(time.monotonic()-start),args.work,args.work/'scoring.log')
        score=json.loads(out.read_text())
        registry=json.loads((ROOT/'experiments/baseline/organizer_baseline_v1_dev_full.json').read_text())
        ref=registry['datasets']['ETs']
        rows=[json.loads(line) for line in (args.work/'inference/stages.jsonl').read_text().splitlines()]
        registered=22-score['predicted_outliers']
        rss=max((x.get('worker_peak_rss_kib',0)*1024 for x in rows),default=0)
        gpu=max((x.get('peak_gpu_allocated_bytes',0) for x in rows),default=0)
        result.update(registered_images=registered, coverage_delta=registered/22-ref['registration_coverage'],
                      peak_worker_rss_bytes=rss, cpu_peak_delta_bytes=rss-ref['cpu_peak_rss_bytes'],
                      peak_torch_bytes=gpu,gpu_peak_delta_bytes=gpu-ref['gpu_peak_allocated_bytes'],
                      measured_stage_seconds=sum(x['seconds'] for x in rows))
        result.update(status='PASS',score=score['scores'],coverage=registered/22,
                      cluster_count=score['predicted_clusters'],deltas={v:{k:n-ref['scores'][v][k] for k,n in metrics.items()} for v,metrics in score['scores'].items()})
    except Exception as exc:result.update(failed_stage=stage,error=str(exc))
    finally:
        result['total_seconds']=time.monotonic()-start
        result['fidelity']='Published YAML and source with opaque ETs names, deterministic staging and separately compiled extensions'
        result['runtime_note']='Total includes verification, inference and scoring; stage sum excludes initialization. Compare corresponding runtime scopes.'
        result['environment_freeze']=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
        result['adapter_sha256']=adapter.digest(Path(adapter.__file__))
        result['launcher_sha256']=adapter.digest(Path(__file__))
        with zipfile.ZipFile(args.work/'imc_ns64_ets_pilot.zip','x',zipfile.ZIP_DEFLATED) as z:
            z.writestr('result.json',json.dumps(result,indent=2))
            z.write(args.receipt,'dependency_receipt.zip')
            z.write(CFG/'pilot.json','pilot.json')
            z.write(CFG/'assets_phase3c1.json','assets.json')
            z.write(Path(adapter.__file__),'source/ns64_adapter.py')
            z.write(Path(__file__),'source/ets_worker.py')
            for name in ['inference/stages.jsonl','inference/predictions.csv','evaluation.json','inference.log','scoring.log']:
                if (args.work/name).exists():z.write(args.work/name,name)
    print(result['status'])

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('images','work','receipt'):parser.add_argument('--'+key,required=True,type=lambda p:Path(p).resolve())
    parser.add_argument('--worker',action='store_true')
    main(parser.parse_args())
