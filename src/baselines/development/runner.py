"""Session-bounded whole-dataset runner using the exact successful smoke workers."""
import argparse,os,time,subprocess,signal,shutil,traceback,csv,sys,math
from pathlib import Path
from common import *
def monitored(command,log,seconds):
    if seconds<=0:raise TimeoutError('Session reserve reached before stage start')
    start=time.monotonic();peak_rss=None;peak_gpu=None
    try:import psutil
    except ImportError:psutil=None
    with Path(log).open('w') as f:
        process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=(os.name!='nt'))
        try:
            while process.poll() is None:
                if time.monotonic()-start>=seconds:raise TimeoutError('Stage deadline reached; completed checkpoints preserved')
                if psutil:
                    try:
                        p=psutil.Process(process.pid);rss=sum(x.memory_info().rss for x in [p]+p.children(recursive=True));peak_rss=max(peak_rss or 0,rss)
                    except (psutil.Error,OSError):pass
                if int(time.monotonic()-start)%5==0:
                    try:
                        out=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','--id=0'],capture_output=True,text=True,timeout=2)
                        used=int(out.stdout.strip())*1024**2;peak_gpu=max(peak_gpu or 0,used)
                    except (OSError,ValueError,subprocess.TimeoutExpired):pass
                time.sleep(.5)
        except BaseException:
            if process.poll() is None:
                if os.name=='nt':process.kill()
                else:os.killpg(process.pid,signal.SIGKILL)
            process.wait()
            write(Path(log).with_suffix('.resources.json'),{'wall_seconds':time.monotonic()-start,'sampled_process_tree_peak_rss_bytes':peak_rss,'sampled_device_used_bytes':peak_gpu,'interrupted':True})
            raise
    measurement={'wall_seconds':time.monotonic()-start,'sampled_process_tree_peak_rss_bytes':peak_rss,'sampled_device_used_bytes':peak_gpu,'measurement_note':'sampled, may miss transient peaks; device use includes other processes'}
    write(Path(log).with_suffix('.resources.json'),measurement)
    if process.returncode:raise RuntimeError('Worker failed with exit '+str(process.returncode)+'; see '+str(log))
    return measurement
def prepare(mapping,data,work):
    from PIL import Image
    folder=work/'images';folder.mkdir(exist_ok=True);hashes={}
    for r in mapping:
        source=(data/'train'/r['dataset']/r['image']).resolve();authorized=(data/'train'/r['dataset']).resolve()
        if not source.is_relative_to(authorized) or not source.is_file():raise FileNotFoundError('Missing/unauthorized input '+str(source))
        hashes[r['opaque']]={'original_sha256':sha(source)}
        with Image.open(source) as im:
            focal=im.getexif().get(41989);exif=Image.Exif()
            if focal is not None:exif[41989]=focal
            im.convert('RGB').save(folder/r['opaque'],exif=exif)
        hashes[r['opaque']]['input_sha256']=sha(folder/r['opaque'])
    return hashes
def result_valid(folder,identity):
    p=folder/'complete.json'
    if not p.exists():return False
    record=read(p)
    if record['implementation_hash']!=identity['implementation_hash']:raise ValueError('Completed result belongs to a different implementation')
    for n,h in record['files'].items():
        if not (folder/n).is_file() or sha(folder/n)!=h:raise ValueError('Completed output changed: '+n)
    return True
def restore(archives,work,implementation):
    for path in archives:
        with zipfile.ZipFile(path) as z:
            m=verify_zip(z)
            if m['implementation_hash']!=implementation:raise ValueError('Resume source/config mismatch')
            for name in m['files']:
                if not name.startswith('datasets/'):continue
                dataset=name.split('/')[1];inputs(dataset)
                target=(work/name).resolve()
                if not target.is_relative_to(work.resolve()):raise ValueError('Unsafe resume path')
                raw=z.read(name)
                if target.exists() and sha(target)!=m['files'][name]:raise ValueError('Conflicting resume artifact: '+name)
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():target.write_bytes(raw)
def export(work,batch_id,implementation,cache=False):
    stamp=time.time_ns();files={}
    for p in work.rglob('*'):
        if not p.is_file() or 'images' in p.relative_to(work).parts:continue
        if p.suffix in {'.json','.csv','.log'}:files[p.relative_to(work).as_posix()]=p
    target=work.parent/f'imc_baseline_dev_batch_{batch_id}_{stamp}.zip'
    write_archive(target,files,{'kind':'report','batch_id':batch_id,'implementation_hash':implementation})
    if cache:
        assets={}
        for p in (work/'datasets').glob('*/[BCDEFGH]/checkpoint.json'):
            ck=read(p)
            for n,h in ck['files'].items():
                f=p.parent/n
                if f.is_file() and sha(f)==h:assets[f.relative_to(work).as_posix()]=f
            assets[p.relative_to(work).as_posix()]=p
        if sum(p.stat().st_size for p in assets.values())>3*1024**3:raise RuntimeError('Cache export exceeds 3 GiB; report saved, retain working session for review')
        write_archive(work.parent/f'imc_baseline_dev_batch_{batch_id}_{stamp}_cache.zip',assets,{'kind':'cache','implementation_hash':implementation,'batch_id':batch_id})
    print('Exported',target,flush=True);return target
def run(settings):
    identity=verify_implementation();cfg=read(CONFIG/'full_development.json');work=Path(settings['work']);work.mkdir(parents=True,exist_ok=True)
    seconds=float(settings['session_remaining_seconds']);reserve=cfg['archive_reserve_seconds']
    if not math.isfinite(seconds) or seconds<=reserve or seconds>cfg['session_seconds_default']:raise ValueError('Set remaining session seconds between reserve and six-hour default cap')
    batches=settings['batches'];known={p['batch']:p for p in cfg['plan']}
    if len(set(batches))!=len(batches) or set(batches)-set(known):raise ValueError('Invalid or duplicate batch IDs')
    batch_id=settings['export_id']
    if not batch_id or any(c not in '0123456789_abcdefghijklmnopqrstuvwxyz' for c in batch_id):raise ValueError('Invalid export ID')
    chosen=[p for p in cfg['plan'] if p['batch'] in batches]
    start=time.monotonic();remaining=lambda:seconds-reserve-(time.monotonic()-start)
    session={'status':'RUNNING','implementation_hash':identity['implementation_hash'],'requested':[p['dataset'] for p in chosen],'attempts':[],'model_config':cfg['model_config']}
    session_name='session_'+str(time.time_ns())+'.json'
    try:
        restore(settings.get('resume_archives',[]),work,identity['implementation_hash'])
        mounts=settings['mounts'];os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCH_HOME=str(work/'torch'),HF_HOME=str(work/'hf'))
        # Exact observed checkpoint/config bytes, independent of mount spelling.
        expected=read(CONFIG/'full_checkpoint_lock.json')
        paths=[p for p in Path(mounts['dinov2']).iterdir() if p.is_file()]+[Path(mounts['aliked']),Path(mounts['lightglue'])]
        if {p.name:sha(p) for p in paths}!=expected:raise ValueError('Model checkpoint/config bytes differ from the successful smoke')
        from preflight import run as preflight
        runtime=preflight(SNAPSHOT,mounts,work,remaining);env_hash=validate_environment(runtime,read(CONFIG/'full_environment_lock.json'))
        session['environment_hash']=env_hash;session['runtime']=runtime
        import resource
        disk=shutil.disk_usage(work);session['available_disk_bytes']=disk.free
        try:
            import psutil
            session['available_ram_bytes']=psutil.virtual_memory().available
        except ImportError:session['available_ram_bytes']=None
        for entry in chosen:
            dataset=entry['dataset'];folder=work/'datasets'/dataset;folder.mkdir(parents=True,exist_ok=True)
            if result_valid(folder,identity):
                session['attempts'].append({'dataset':dataset,'status':'REUSED_VERIFIED_COMPLETE'});continue
            if remaining()<120:break
            print('Starting',dataset,entry['images'],'images;',round(remaining()),'seconds before archive reserve',flush=True)
            attempt={'dataset':dataset,'status':'INCOMPLETE','failure_class':None,'stages':{},'started_ns':time.time_ns()};dataset_start=time.monotonic();stage='INPUT'
            try:
                mapping=inputs(dataset);hashes=prepare(mapping,Path(mounts['competition']),folder)
                write(folder/'identity_mapping.json',mapping);write(folder/'input_hashes.json',hashes)
                model=dict(cfg['model_config'],dataset=dataset)
                state={'source_root':str(SNAPSHOT),'work':str(folder),'images':[r['opaque'] for r in mapping],'dino_dir':mounts['dinov2'],'config':model}
                write(folder/'inference_state.json',state)
                base=fingerprint({'implementation':identity['implementation_hash'],'membership':mapping,'inputs':hashes,'environment':env_hash,'weights':expected});previous=base
                estimated_cache=entry['images']*4096*128*4*4+2*1024**3
                if shutil.disk_usage(work).free<estimated_cache:raise OSError('Insufficient disk for conservative feature/database allowance')
                for stage in 'BCDEFGH':
                    directory=folder/stage;key=fingerprint({'base':base,'previous':previous,'stage':stage})
                    if checkpoint_valid(directory,key):
                        attempt['stages'][stage]={'status':'REUSED','result':read(directory/'result.json')};previous=fingerprint(read(directory/'checkpoint.json')['files']);continue
                    if remaining()<60:raise TimeoutError('Safe checkpoint stop before next stage')
                    if directory.exists():directory.rename(folder/(stage+'_incomplete_'+str(time.time_ns())))
                    directory.mkdir()
                    # E/F in the original worker copy features from the preceding folder.
                    if stage=='F':
                        for n in ['keypoints.h5','descriptors.h5']:
                            if not (folder/'E'/n).exists():shutil.copyfile(folder/'D'/n,folder/'E'/n)
                    budget=min(remaining(),cfg['sfm_stage_limit_seconds']) if stage=='G' else remaining()
                    for old in [folder/(stage+'.log'),folder/(stage+'.resources.json')]:
                        if old.exists():old.rename(old.with_name(old.stem+'_previous_'+str(time.time_ns())+old.suffix))
                    measured=monitored([sys.executable,str(SNAPSHOT/'src/baselines/organizer/worker.py'),stage,str(folder/'inference_state.json'),str(directory)],folder/(stage+'.log'),budget)
                    outcome=read(directory/'result.json')
                    if outcome['status']!='PASS':raise RuntimeError('Stage did not return PASS')
                    previous=checkpoint(directory,stage,key);attempt['stages'][stage]={'status':'PASS','result':outcome,'resources':measured}
                    write(folder/'progress.json',attempt)
                    print(dataset,stage,'PASS',flush=True)
                poses=read(folder/'H/assignments.json');pred=serialize(mapping,poses);pred.sort(key=lambda r:(r['dataset'],r['image']))
                write_predictions(folder/'predictions.csv',pred)
                stage='EVALUATION'
                monitored([sys.executable,str(ROOT/'src/baselines/development/evaluate.py'),dataset,str(folder/'predictions.csv'),str(folder/'evaluation.json')],folder/'evaluation.log',remaining())
                evaluation=read(folder/'evaluation.json');g=read(folder/'G/result.json')['metrics'];registered=sum(v is not None for v in poses.values())
                stages={s:read(folder/s/'result.json') for s in 'BCDEFGH'}
                result={'dataset':dataset,'partition':entry['partition'],'images':len(mapping),'membership_hash':fingerprint(mapping),'input_inventory_hash':fingerprint(hashes),'implementation_hash':identity['implementation_hash'],'model_config_hash':fingerprint(cfg['model_config']),'checkpoint_hash':fingerprint(expected),'environment_hash':env_hash,'benchmark_hash':BENCHMARK,'split_hash':SPLIT,'status':'SCORED_ALGORITHMIC_FAILURE' if registered==0 else 'COMPLETE','failure_count':int(registered==0),'failure_reason':'NO_REGISTERED_CAMERAS' if registered==0 else None,'registered_images':registered,'registration_coverage':registered/len(mapping),'models':g['models'],'retained_match_pairs':stages['E']['metrics']['retained_pairs'],'verified_pairs':stages['F']['metrics']['verified_pairs'],'selected_pairs':stages['C']['metrics']['selected_pairs'],'runtime_by_stage':{s:r['runtime_seconds'] for s,r in stages.items()},'successful_stage_seconds':sum(r['runtime_seconds'] for r in stages.values()),'gpu_peak_allocated_bytes':max(r.get('gpu_peak_allocated_bytes',0) for r in stages.values()),'gpu_peak_reserved_bytes':max(r.get('gpu_peak_reserved_bytes',0) for r in stages.values()),'cpu_peak_rss_bytes':max(r.get('cpu_peak_rss_bytes',0) for r in stages.values()),**evaluation}
                # Post-inference graph diagnostics; no labels and no feedback to workers.
                import sqlite3
                with sqlite3.connect(folder/'F/colmap.db') as db:
                    vertices={row[0] for row in db.execute('SELECT image_id FROM images')};edges=list(db.execute('SELECT pair_id,rows FROM two_view_geometries'))
                adjacency={v:set() for v in vertices}
                for pair_id,inliers in edges:
                    if inliers>0:
                        a,b=divmod(pair_id,2147483647)
                        adjacency[a].add(b);adjacency[b].add(a)
                unseen=set(vertices);components=[]
                while unseen:
                    pending=[unseen.pop()];count=0
                    while pending:
                        node=pending.pop();count+=1;new=adjacency[node]&unseen;unseen-=new;pending.extend(new)
                    components.append(count)
                result['connectivity']={'verified_component_sizes':sorted(components,reverse=True),'isolated_images':sum(not v for v in adjacency.values()),'empty_verified_graph':not any(adjacency.values())}
                exclusive(folder/'result.json',result)
                exclusive(folder/'complete.json',{'implementation_hash':identity['implementation_hash'],'files':{n:sha(folder/n) for n in ['result.json','predictions.csv','evaluation.json','identity_mapping.json','input_hashes.json']}})
                attempt['status']=result['status']
            except Exception as e:
                attempt.update(error=str(e),traceback=traceback.format_exc(),failed_stage=stage,failure_class='INFRASTRUCTURE_OR_INTERRUPTION' if isinstance(e,(OSError,TimeoutError)) else 'UNCLASSIFIED_REQUIRES_REVIEW')
                log=folder/(stage+'.log')
                if log.exists() and 'out of memory' in log.read_text(errors='replace').lower():attempt['failure_class']='RESOURCE_OOM'
                print(dataset,attempt['failure_class'],str(e),flush=True)
            finally:
                attempt['wall_seconds']=time.monotonic()-dataset_start;exclusive(folder/('attempt_'+str(attempt['started_ns'])+'.json'),attempt);session['attempts'].append(attempt)
                write(work/session_name,session)
            if attempt['status']=='INCOMPLETE':break
        session['status']='BATCH_COMPLETE' if all(result_valid(work/'datasets'/p['dataset'],identity) for p in chosen) else 'PARTIAL'
    except Exception as e:session.update(status='BLOCKED',error=str(e),traceback=traceback.format_exc())
    finally:
        session['wall_seconds']=time.monotonic()-start;write(work/session_name,session);export(work,batch_id,identity['implementation_hash'],settings.get('export_cache',False))
    return session
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--settings',required=True);a=p.parse_args();result=run(read(a.settings));print(result['status']);sys.exit(0 if result['status']=='BATCH_COMPLETE' else 2)
