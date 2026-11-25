"""Build clean notebooks from the two historical and hash-checked inference workers."""
import ast,base64,hashlib,io,json,zipfile
from pathlib import Path

def digest(b):
    return hashlib.sha256(b).hexdigest()

RUN = """
from pathlib import Path
from google.colab import files
import os,sys,signal,subprocess,time,json,zipfile,hashlib,base64,io,psutil,shutil
root=Path('/content/phase4g')
root.mkdir()
kit=root/'adapter'
kit.mkdir()
raw=base64.b64decode(KIT)
assert hashlib.sha256(raw).hexdigest()==KIT_HASH
with zipfile.ZipFile(io.BytesIO(raw)) as z:z.extractall(kit)
work=root/'run'
work.mkdir()
python=Path('/content/ns64_clean/venv/bin/python')
(work/'worker.py').write_text(WORKER)
assert hashlib.sha256((work/'worker.py').read_bytes()).hexdigest()==WORKER_PIN
remaining=min(int(input('Available session seconds (maximum 5400): ')),5400-int(time.monotonic()-PHASE_START))
deadline=time.monotonic()+remaining-90
def launch(args,label):
    start=time.monotonic()
    if start>=deadline:
        raise TimeoutError('deadline before '+label)
    peak=0
    with (work/(label+'.log')).open('w') as log:
        with subprocess.Popen([str(python),'-I','-u',*args],cwd=work,stdout=log,stderr=subprocess.STDOUT,
                              env=dict(os.environ,STAIRS_DEADLINE=str(deadline)),start_new_session=True) as proc:
            try:
                while proc.poll() is None:
                    try:
                        process=psutil.Process(proc.pid)
                        rss=sum(p.memory_info().rss for p in [process]+process.children(recursive=True) if p.is_running())
                        peak=max(peak,rss)
                    except psutil.NoSuchProcess:
                        pass
                    if peak>12*1024**3 or time.monotonic()>=deadline:
                        raise RuntimeError('Resource limit in '+label)
                    time.sleep(1)
                if proc.returncode:raise RuntimeError(label+' exit '+str(proc.returncode))
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait()
                (work/(label+'_usage.json')).write_text(json.dumps(dict(seconds=time.monotonic()-start,peak_tree_rss_bytes=peak)))
record=dict(status='BLOCKED',selected=None)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
try:
    launch([str(work/'worker.py'),'20261111'],'capture')
    baseline=work/'opaque.csv'
    if sha(baseline)!=BASELINE_PIN:
        raise ValueError('Opaque baseline fingerprint differs. Preserve for independent parity diagnosis.')
    record=dict(status='COMPLETE',selected='pass_through',reason='trial unavailable',parity=True)
    try:
        command="import sys;sys.path.insert(0,"+repr(str(kit))+");from pathlib import Path;from src.novel_model.reconstruction_feedback_v2.worker import run_trial;from src.novel_model.geometry_capture_v1.database import save;save(Path('decision.json'),run_trial(Path.cwd()))"
        launch(['-c',command],'feedback')
        record.update(json.loads((work/'decision.json').read_bytes()))
    except Exception as error:
        record.update(reason=str(error))
    selected=(work/'feedback/trial.csv') if record['selected']=='trial' else baseline
    shutil.copyfile(selected,work/'final.csv')
    record.update(final_sha256=sha(work/'final.csv'),pass_sha256=sha(baseline))
except Exception as error:
    record.update(status='BLOCKED',error=str(error))
finally:
    record.update(protocol=KIT_HASH,worker=WORKER_PIN,wall_seconds=time.monotonic()-PHASE_START)
    (work/'selection.json').write_text(json.dumps(record,indent=2))
    archive=root/('phase4g_'+DATASET+'_return.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for f in [*work.glob('*.log'),*work.glob('*.json'),*work.glob('*.jsonl'),*work.glob('*.csv'),*(work/'feedback').glob('*.json'),*(work/'feedback').glob('*.csv')]:
            if f.name!='models.json':z.write(f,f.relative_to(work))
    print(record['status'],record['selected'],record.get('reason',record.get('error','')),flush=True)
    files.download(str(archive))
    parent=work/'feedback/parent.db'
    if parent.exists():
        print('Save the numerical parent for reuse',flush=True)
        files.download(str(parent))
"""
def main():
    package=[Path('src/novel_model')/p for p in ('geometry_graph_v1/schemas.py','geometry_graph_v1/spectral_proposal.py',
        'geometry_capture_v1/database.py','reconstruction_feedback_v1/__init__.py','reconstruction_feedback_v2/__init__.py','reconstruction_feedback_v2/worker.py')]
    config=Path('configs/novel_model/reconstruction_feedback_v2.yaml')
    package.append(config)
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as kit:
        for path in package:
            info=zipfile.ZipInfo(path.as_posix())
            kit.writestr(info,path.read_bytes(),compress_type=zipfile.ZIP_DEFLATED)
    raw=buffer.getvalue()
    manifest=dict(files={p.as_posix():digest(p.read_bytes()) for p in package},model_bundle=digest(raw),datasets={})
    refs=json.loads(Path('reports/phase3c8/four_archive_index.json').read_bytes())
    for ds,nb in [('stairs','step03c6_stairs_colab'),('imc2023_theather_imc2024_church','step03c8_76_colab')]:
        ref=next(r for r in refs if r['dataset']==ds)
        archive=Path(ref['archive'])
        assert digest(archive.read_bytes())==ref['sha256']
        with zipfile.ZipFile(archive) as z:
            baseline_pin=digest(z.read('opaque.csv'))
            cache_members=[n for n in z.namelist() if n.endswith(('.db','.h5','.npz','.sqlite'))]
        d=json.loads(Path('notebooks',nb+'.ipynb').read_bytes())
        last=d['cells'][-1]
        worker=ast.literal_eval(ast.parse(''.join(last['source'])).body[0].value)
        assert digest(worker.encode())==ref['worker_sha256']
        old='            result=fn(*args,**kwargs)'
        assert worker.count(old)==1
        worker=worker.replace(old,"""            if name=='incremental_mapping':
                sys.path.insert(0,'/content/phase4g/adapter')
                from src.novel_model.reconstruction_feedback_v2.worker import capture
                result=capture(fn,work,DATASET,**kwargs)
            else:
                result=fn(*args,**kwargs)""".replace('DATASET',repr(ds)))
        pin=digest(worker.encode())
        head=dict(WORKER=worker,WORKER_PIN=pin,KIT=base64.b64encode(raw).decode(),KIT_HASH=digest(raw),BASELINE_PIN=baseline_pin,DATASET=ds)
        last['source']=([k+'='+repr(v)+'\n' for k,v in head.items()]+RUN.splitlines(True))
        d['cells'][0]['source']=['import time\n','PHASE_START=time.monotonic()\n']+d['cells'][0]['source']
        for c in d['cells']:
            ast.parse(''.join(c['source']))
            c.update(outputs=[],execution_count=None)
        path=Path('notebooks','step04g_'+ds+'_colab.ipynb')
        path.write_text(json.dumps(d,indent=1))
        manifest['datasets'][ds]=dict(worker=pin,notebook=digest(path.read_bytes()),opaque_baseline=baseline_pin,
            historical_archive=ref['sha256'],historical_cache_members=cache_members)
    Path('reports/phase4g/protocol_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
