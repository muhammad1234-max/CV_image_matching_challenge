"""Pin the new orchestration and package the unchanged verified smoke implementation."""
import base64,io,json,sys,zipfile,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/baselines/development'))
from common import *

def build():
    files=[]
    for directory in [SNAPSHOT,ROOT/'src/baselines/development']:
        files.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.extend(CONFIG/n for n in ['full_development.json','full_environment_lock.json','full_checkpoint_lock.json','development_inputs.json'])
    files.extend((CONFIG/'development_scoring').rglob('*.csv'))
    files.extend(ROOT/n for n in ['scripts/merge_baseline_batches.py','scripts/build_phase3b_notebook.py'])
    identity={'implementation_id':'organizer_smoke_verified_full_development_v1','algorithm_snapshot_hash':read(CONFIG/'full_development.json')['algorithm_snapshot_hash'],'benchmark_hash':BENCHMARK,'split_hash':SPLIT,'git_commit':None,'files':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}}
    identity['implementation_hash']=fingerprint(identity)
    lock=CONFIG/'full_implementation.json'
    if lock.exists() and read(lock)!=identity:raise RuntimeError('Implementation already pinned; review any source changes explicitly before creating a new preparation snapshot')
    if not lock.exists():exclusive(lock,identity)
    verify_implementation();files.append(lock);buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            info=zipfile.ZipInfo(p.relative_to(ROOT).as_posix(),date_time=(2026,11,12,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
    raw=buf.getvalue();payload=base64.b64encode(raw).decode()
    setup=f'''import base64, hashlib, io, json, sys, zipfile
from pathlib import Path
raw=base64.b64decode({payload!r})
assert hashlib.sha256(raw).hexdigest()=={hashlib.sha256(raw).hexdigest()!r}
ROOT=Path('/kaggle/working/phase3b_source_{identity['implementation_hash'][:12]}')
ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for name in z.namelist():
        target=(ROOT/name).resolve()
        assert target.is_relative_to(ROOT.resolve())
        data=z.read(name)
        if target.exists() and target.read_bytes()!=data:raise RuntimeError('Source differs; use a fresh session')
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(data)
print('Phase 3B source verified:', {identity['implementation_hash']!r})
'''
    # Mount resolution is copied from the reviewed Phase 3A notebook, not rebuilt.
    old=read(ROOT/'notebooks/step03a_baseline_gpu_smoke.ipynb')
    oldsettings=''.join(old['cells'][2]['source'])
    mountcode=oldsettings[oldsettings.index('def mounted('):oldsettings.index('config = json.loads')]
    settings='''# Edit only these session controls. Model settings and split are immutable.
# Use a fork pinned to the original Python 3.10.12 environment, T4, Internet OFF.
BATCH_IDS = ['01']  # 01..11 in docs/step03b_execution_plan.md
SESSION_REMAINING_SECONDS = 3600  # actual time remaining, <=21600; reserves 600 s
EXPORT_CACHE = True  # optional features/database cache; no source images/weights
RESUME_ARCHIVES = []  # absolute mounted report AND cache .zip paths for same dataset
EXPORT_ID = '_'.join(BATCH_IDS)
WORK = Path('/kaggle/working/phase3b_'+EXPORT_ID)
'''+mountcode+'''
settings={'work':str(WORK),'batches':BATCH_IDS,'export_id':EXPORT_ID,
 'session_remaining_seconds':SESSION_REMAINING_SECONDS,'export_cache':EXPORT_CACHE,
 'resume_archives':RESUME_ARCHIVES,'mounts':mounts}
SETTINGS=Path('/kaggle/working/phase3b_settings.json')
SETTINGS.write_text(json.dumps(settings))
print('Development only; batches:', BATCH_IDS, 'remaining seconds:',SESSION_REMAINING_SECONDS)
'''
    launch='''import subprocess
result=subprocess.run([sys.executable,str(ROOT/'src/baselines/development/runner.py'),'--settings',str(SETTINGS)])
print('Runner exit:',result.returncode)
print('Download all imc_baseline_dev_batch_*.zip files from /kaggle/working, including partial reports and cache archives.')
print('Return archives for independent validation. No full baseline is registered by this notebook.')
if result.returncode:print('STOP: inspect failure receipt. Preserve model settings; resume only verified checkpoints.')
'''
    cells=[{'cell_type':'markdown','metadata':{},'source':['# Phase 3B: frozen full-development baseline\n','Run one whole dataset per batch in the documented order. Default 01 is a NEW ETs execution, not reuse of the smoke score. 1,563 development images; no confirmation or hidden-test inference. Labels are used only by the separate post-prediction evaluator. Read docs/step03b_supervisor_handoff.md before execution.']}]
    for code in [setup,settings,launch]:cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)})
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
    (ROOT/'notebooks/step03b_full_development_baseline.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
    print(json.dumps({'implementation_hash':identity['implementation_hash'],'payload_bytes':len(raw),'files':len(files)}))
if __name__=='__main__':build()
