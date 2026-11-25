"""Bounded sequential A-J smoke run. Failures always produce an honest receipt."""
import argparse,csv,json,os,shutil,subprocess,sys,time,traceback,zipfile,re
from pathlib import Path
from contracts import *
def stage_process(command,log,seconds):
    if seconds<=0:raise TimeoutError('20-minute budget exhausted before next stage')
    with Path(log).open('w') as f:
        try:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=seconds)
        except subprocess.TimeoutExpired as e:raise TimeoutError('Stage exceeded remaining budget; child terminated; partial stage not reusable') from e
    if r.returncode:raise RuntimeError('Stage failed; inspect '+str(log))
def prepare_images(mapping,data,work):
    from PIL import Image
    folder=work/'images';folder.mkdir(exist_ok=True);hashes={}
    for r in mapping:
        source=(data/'train'/r['dataset']/r['image']).resolve()
        if not source.is_relative_to((data/'train/ETs').resolve()):raise ValueError('Input path escaped authorized ETs dataset')
        if not source.is_file():raise FileNotFoundError(source)
        hashes[r['opaque']]={'original_sha256':sha(source)}
        # Lossless RGB view, remove captions/scene metadata; preserve only focal prior.
        with Image.open(source) as im:
            focal=im.getexif().get(41989);exif=Image.Exif()
            if focal is not None:exif[41989]=focal
            im.convert('RGB').save(folder/r['opaque'],exif=exif)
        hashes[r['opaque']]['input_sha256']=sha(folder/r['opaque'])
    return hashes
def archive(work,receipt):
    write(work/'receipt.json',receipt)
    allowed=[p for p in work.rglob('*') if p.is_file() and p.suffix in {'.json','.log','.csv'} and 'images' not in p.parts and p.name!='ground_truth.csv']
    with zipfile.ZipFile(work.parent/'imc_baseline_smoke_report.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in allowed:
            if p.stat().st_size<10*1024*1024:z.write(p,p.relative_to(work))
def run(root,mounts,work,config):
    work.mkdir(parents=True,exist_ok=True);start=time.monotonic();remaining=lambda:config['budget_seconds']-(time.monotonic()-start)
    receipt={'kind':'smoke_test','not_registered_baseline':True,'status':'RUNNING','benchmark_hash':BENCHMARK,'split_hash':SPLIT,'dataset':'ETs','role':'dev_b; training in fold A, evaluation in fold B','config':config,'config_hash':fingerprint(config),'git_commit':None,'stages':{s:{'status':'NOT_RUN'} for s in 'ABCDEFGHIJ'}}
    stage='A'
    try:
        if config['dataset']!='ETs' or config['budget_seconds']>1200:raise ValueError('Unauthorized dataset or increased smoke budget')
        # Validate fixed numerical model settings, independently of configurable memory cap.
        expected=json.loads((root/'configs/baselines/organizer_v1/smoke.json').read_text())
        for k in expected:
            if k not in ['gpu_memory_gib','budget_seconds'] and config[k]!=expected[k]:raise ValueError('Unreviewed baseline adaptation: '+k)
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCH_HOME=str(work/'torch'),HF_HOME=str(work/'hf'),TOKENIZERS_PARALLELISM='false')
        snapshot=json.loads((root/'configs/baselines/organizer_v1/source_snapshot.json').read_text())
        for n,h in snapshot['files'].items():
            if sha(root/n)!=h:raise ValueError('Source snapshot mismatch: '+n)
        receipt['source_snapshot_hash']=fingerprint(snapshot);write(work/'source_snapshot.json',snapshot)
        b=verify_benchmark(root);receipt.update(evaluator_hash=b['evaluator_hash'],threshold_hash=b['threshold_hash'],dataset_inventory_hash=b['dataset_inventory_hash'])
        mapping=selection(root)
        from preflight import run as preflight
        runtime=preflight(root,mounts,work,remaining)
        input_hashes=prepare_images(mapping,Path(mounts['competition']),work)
        write(work/'identity_mapping.json',mapping);write(work/'input_hashes.json',input_hashes)
        state={'source_root':str(root),'work':str(work),'images':[r['opaque'] for r in mapping],'dino_dir':mounts['dinov2'],'config':config}
        write(work/'inference_state.json',state)
        key=fingerprint({'inputs':input_hashes,'config':config,'source':snapshot,'assets':runtime['checkpoint_hashes'],'runtime':{k:runtime[k] for k in ['python','torch','cuda_build','gpu','packages']}})
        run_key=key
        receipt['stages']['A']={'status':'PASS','runtime_seconds':time.monotonic()-start,'input_images':len(mapping),'runtime':runtime}
        archive(work,receipt)
        for stage in 'BCDEFGH':
            directory=work/stage;stage_key=fingerprint({'run_key':key,'stage':stage})
            if cache_valid(directory,stage_key):
                receipt['stages'][stage]={'status':'PASS','reused_verified_cache':True,'original_result':json.loads((directory/'result.json').read_text())}
                key=fingerprint({'run':run_key,'previous':json.loads((directory/'checkpoint.json').read_text())['artifacts']});continue
            if directory.exists():
                # Move only this known run-local incomplete stage; never touch mounted inputs.
                old=work/(stage+'_incomplete_'+str(time.time_ns()));directory.rename(old)
            directory.mkdir();t=time.monotonic()
            stage_process([sys.executable,str(root/'src/baselines/organizer/worker.py'),stage,str(work/'inference_state.json'),str(directory)],work/(stage+'.log'),remaining())
            result=json.loads((directory/'result.json').read_text())
            if result['status']!='PASS':raise RuntimeError('Worker did not pass')
            if stage=='E':
                pairs=json.loads((work/'C/pairs.json').read_text());names=state['images']
                raw_counts={names[a]+'/'+names[b]:0 for a,b in pairs}
                for a,b,n in re.findall(r'(image_\d+\.png)-(image_\d+\.png): (\d+) matches',(work/'E.log').read_text()):raw_counts[a+'/'+b]=int(n)
                result['metrics']['raw_match_counts']=raw_counts;write(directory/'result.json',result)
            artifacts={str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file()}
            write(directory/'checkpoint.json',{'status':'PASS','cache_key':stage_key,'artifacts':artifacts})
            key=fingerprint({'run':run_key,'previous':artifacts})
            receipt['stages'][stage]=result;archive(work,receipt)
        poses=json.loads((work/'H/assignments.json').read_text());predictions=serialize(mapping,poses)
        # Restore original training CSV order for the selected dataset after inference.
        score_gt=root/'configs/baselines/organizer_v1/scoring/ETs.csv'
        with score_gt.open(newline='') as f:order=[r['image'] for r in csv.DictReader(f)]
        lookup={r['image']:r for r in predictions};write_predictions(work/'predictions.csv',[lookup[n] for n in order])
        receipt['registered_images']=sum(p is not None for p in poses.values());receipt['registration_coverage']=receipt['registered_images']/len(mapping)
        stage='I';(work/'I').mkdir(exist_ok=True);t=time.monotonic()
        stage_process([sys.executable,str(root/'src/baselines/organizer/score.py'),str(root),str(work)],work/'I.log',remaining())
        receipt['stages']['I']={'status':'PASS','runtime_seconds':time.monotonic()-t,'metrics_file':'I/metrics.json'};receipt['status']='PASS'
    except Exception as e:
        receipt['status']='BLOCKED' if stage=='A' else 'FAILED';receipt['stages'][stage]={'status':'FAILED','error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()}
        result=work/stage/'result.json'
        if result.exists():receipt['stages'][stage]['worker_result']=json.loads(result.read_text())
    finally:
        receipt['runtime_seconds']=time.monotonic()-start;receipt['stages']['J']={'status':'PASS','meaning':'receipt/export only; does not imply successful inference'};archive(work,receipt)
    return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--mounts',required=True);p.add_argument('--work',required=True);p.add_argument('--config',required=True);a=p.parse_args()
    r=run(Path(a.root),json.loads(Path(a.mounts).read_text()),Path(a.work),json.loads(Path(a.config).read_text()));print(json.dumps({'status':r['status'],'stages':r['stages']},indent=2));sys.exit(0 if r['status']=='PASS' else 2)
