"""Independent offline dependency checks. No benchmark images are read."""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def probe(code, cwd, args=(), timeout=60):
    start=time.monotonic()
    measurement = "\nimport sys\nif sys.platform=='linux':\n import resource,json; print('RESOURCE',json.dumps(dict(cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gpu_peak_allocated_bytes=__import__('torch').cuda.max_memory_allocated() if 'torch' in __import__('sys').modules and __import__('torch').cuda.is_available() else 0)))"
    try:
        result = subprocess.run([sys.executable,'-c',code+measurement,*map(str,args)],cwd=cwd,
                                capture_output=True,text=True,timeout=timeout)
        return {'status':'PASS' if result.returncode==0 else 'BLOCKED',
                'seconds':time.monotonic()-start, 'details':(result.stdout+result.stderr)[-3000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'status':'BLOCKED','details':str(exc)}


def input_files(roots, filename):
    for root in roots:
        for directory, names, files in os.walk(root):
            names[:] = [n for n in names if n not in ('competitions','train','test') and not n.startswith('image-matching-challenge-')]
            if filename in files:
                yield Path(directory)/filename


def locate(roots, asset):
    paths = sorted(set(input_files(roots,asset['filename'])))
    found = [{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in paths]
    matches = [p for p in found if p['sha256']==asset['sha256']]
    return found, Path(matches[0]['path']) if matches else None


def verify(manifest, roots, bundle, receipt):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    start=time.monotonic()
    checks={}
    record={'platform':platform.platform(),'python':sys.version,'checks':checks,'assets':{},
            'environment_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)}
    work=Path(os.environ.get('NS64_WORK_ROOT','/kaggle/working'))
    record['build_evidence']={}
    for name in ['ns64_wheelhouse/asmk_build_adaptation.json','ns64_wheelhouse/SHA256SUMS.json','ns64_toolchain_explicit.txt']:
        p=work/name
        if p.is_file():record['build_evidence'][name]=p.read_text()
    record['installed_extension_origins']={}
    import importlib.metadata
    for name in ['asmk','curope']:
        try:record['installed_extension_origins'][name]=importlib.metadata.distribution(name).read_text('direct_url.json')
        except importlib.metadata.PackageNotFoundError:record['installed_extension_origins'][name]=None
    def save():
        record['seconds']=time.monotonic()-start
        temp=receipt.with_suffix('.tmp')
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('result.json',json.dumps(record,indent=2))
            z.writestr('manifest.json',json.dumps(manifest,indent=2))
        temp.replace(receipt)
    record['status']='PARTIAL'
    save()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)
            if sha(bundle)!=manifest['archive_sha256']:
                raise ValueError('Source bundle SHA256 mismatch')
            with zipfile.ZipFile(bundle) as z:z.extractall(source)
            if sha(source/manifest['config'])!=manifest['config_sha256']:
                raise ValueError('Configuration SHA256 mismatch')
            if sys.version_info[:2] == (3,11) and sys.platform=='linux':
                wheels=sorted((source/'bundle/oss').rglob('*.whl'))
                expected=manifest['bundled_wheels']
                if any(sha(w)!=expected[w.name] for w in wheels):
                    raise ValueError('Source wheel integrity error')
                extra=next((p for p in input_files(roots,manifest['pycolmap_filename']) if sha(p)==manifest['pycolmap_sha256']), None)
                if extra:wheels.append(extra)
                target=source/'offline_site'
                install=subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(target),*map(str,wheels)],capture_output=True,text=True,timeout=180)
                checks['bundled_wheel_install']={'status':'PASS' if install.returncode==0 else 'BLOCKED','details':install.stderr[-1600:]}
                os.environ['PYTHONPATH']=os.pathsep.join([str(target),os.environ.get('PYTHONPATH','')])
            for name,code in manifest['probes'].items():
                checks[name]=probe(code,source)
                save()
            verified={}
            for key,asset in manifest['assets'].items():
                found,path=locate(roots,asset)
                if path and key=='dinov2':
                    if any(not (path.parent/n).is_file() or sha(path.parent/n)!=v for n,v in asset['sidecars'].items()):path=None
                record['assets'][key]={'files':found,'status':'PASS' if path else 'BLOCKED'}
                if path:verified[key]=path
                save()
            for key,code in manifest['loaders'].items():
                if key=='codebook':
                    checks['load_codebook']=dict(checks['load_retrieval'])
                    save()
                    continue
                required={'retrieval':['mast3r','retrieval','codebook'],'codebook':['mast3r','retrieval','codebook']}.get(key,[key])
                if all(k in verified for k in required):
                    paths=[verified[k] for k in required]
                    if key in ('retrieval','codebook') and paths[-1].parent!=paths[-2].parent:
                        result={'status':'BLOCKED','details':'Retriever requires colocated head and codebook'}
                    else:result=probe(code,source,paths,120)
                else:result={'status':'BLOCKED','details':'Missing verified prerequisite files'}
                checks['load_'+key]=result
                save()
            record['status']='PASS' if all(x['status']=='PASS' for x in list(checks.values())+list(record['assets'].values())) else 'BLOCKED'
    except Exception as exc:
        record.update(status='BLOCKED',error=str(exc))
    finally:save()
    return record
