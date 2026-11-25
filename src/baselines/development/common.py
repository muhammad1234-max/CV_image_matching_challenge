"""Frozen Phase 3B identities, validated stage checkpoints and result aggregation."""
import json,sys,random,zipfile,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
SNAPSHOT=ROOT/'snapshots/organizer_phase3a_verified'
sys.path.insert(0,str(SNAPSHOT/'src/baselines/organizer'))
from contracts import sha,fingerprint,write,serialize,write_predictions,BENCHMARK,SPLIT
CONFIG=ROOT/'configs/baselines/organizer_v1'
def read(path):return json.loads(Path(path).read_text())
def exclusive(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
def verify_implementation():
    identity=read(CONFIG/'full_implementation.json');body={k:v for k,v in identity.items() if k!='implementation_hash'}
    if fingerprint(body)!=identity['implementation_hash']:raise ValueError('Implementation manifest changed')
    for n,h in identity['files'].items():
        if sha(ROOT/n)!=h:raise ValueError('Pinned implementation changed: '+n)
    frozen=read(SNAPSHOT/'configs/benchmark/imc2025_local_v1/benchmark.json')
    body={k:v for k,v in frozen.items() if k!='benchmark_hash'}
    if fingerprint(body)!=BENCHMARK or frozen['split_hash']!=SPLIT:raise ValueError('Frozen benchmark mismatch')
    original=read(SNAPSHOT/'configs/baselines/organizer_v1/source_snapshot.json')
    if fingerprint(original)!=read(CONFIG/'full_development.json')['algorithm_snapshot_hash']:raise ValueError('Smoke source snapshot mismatch')
    for n,h in original['files'].items():
        if sha(SNAPSHOT/n)!=h:raise ValueError('Executed algorithm source changed: '+n)
    return identity
def inputs(dataset):
    rows=read(CONFIG/'development_inputs.json');selected=[r for r in rows if r['dataset']==dataset]
    plan=read(CONFIG/'full_development.json')['plan'];entry=next((r for r in plan if r['dataset']==dataset),None)
    if entry is None or not selected or any(r['partition'] not in ('dev_a','dev_b') for r in selected):raise ValueError('Unauthorized/confirmation dataset')
    if len(selected)!=entry['images'] or len({r['image'] for r in selected})!=len(selected):raise ValueError('Incomplete dataset membership')
    selected=[{'dataset':r['dataset'],'image':r['image']} for r in selected]
    random.Random(20261111).shuffle(selected)
    return [dict(r,opaque=f'image_{i:04d}.png') for i,r in enumerate(selected)]
def validate_environment(runtime,lock):
    for key in ['python','torch','cuda_build','gpu']:
        if runtime.get(key)!=lock[key]:raise ValueError('Environment differs from successful smoke: '+key)
    actual={k.lower().replace('_','-'):v for k,v in runtime['packages'].items()}
    for name,v in lock['packages'].items():
        if actual.get(name)!=v:raise ValueError('Package differs from successful smoke: '+name)
    return fingerprint({k:runtime[k] for k in ['python','torch','cuda_build','gpu']}|{'packages':{k:actual[k] for k in lock['packages']}})
NEEDED={'B':['embeddings.pt'],'C':['pairs.json'],'D':['keypoints.h5','descriptors.h5'],'E':['matches.h5'],'F':['colmap.db'],'G':['poses.json'],'H':['assignments.json']}
def checkpoint_valid(folder,key):
    p=Path(folder)/'checkpoint.json'
    if not p.is_file():return False
    ck=read(p)
    return ck.get('key')==key and ck.get('status')=='PASS' and bool(ck.get('files')) and all((p.parent/n).is_file() and sha(p.parent/n)==h for n,h in ck['files'].items())
def checkpoint(folder,stage,key):
    files={n:sha(Path(folder)/n) for n in NEEDED[stage]+['result.json']}
    write(Path(folder)/'checkpoint.json',{'status':'PASS','key':key,'files':files});return fingerprint(files)
def safe_zip_files(z,max_bytes=4*1024**3):
    names=z.namelist()
    if len(names)!=len(set(names)) or sum(i.file_size for i in z.infolist())>max_bytes:raise ValueError('Duplicate archive paths or oversized archive')
    for n in names:
        p=Path(n)
        if p.is_absolute() or '..' in p.parts or '\\' in n or ':' in n:raise ValueError('Unsafe archive path')
    return names
def verify_zip(z):
    names=safe_zip_files(z);manifest=json.loads(z.read('archive_manifest.json'))
    if set(names)-{'archive_manifest.json'}!=set(manifest['files']):raise ValueError('Archive inventory differs')
    for n,h in manifest['files'].items():
        if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('Archive content hash mismatch: '+n)
    return manifest
def write_archive(path,files,metadata):
    # Never replace a completed export silently.
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
        hashes={}
        for name,p in sorted(files.items()):z.write(p,name);hashes[name]=sha(p)
        z.writestr('archive_manifest.json',json.dumps(metadata|{'files':hashes},indent=2))
def aggregate(results,plan):
    expected={p['dataset'] for p in plan}
    if set(results)-expected:raise ValueError('Unknown dataset result')
    complete=set(results)==expected
    def mean(names,variant):return sum(results[n]['scores'][variant]['combined_score'] for n in names)/len(names) if names else None
    fold={}
    for partition in ['dev_a','dev_b']:
        names=[p['dataset'] for p in plan if p['partition']==partition]
        fold[partition]={v:mean(names,v) if all(n in results for n in names) else None for v in ['participant','prose_sensitivity']}
    return {'status':'COMPLETE_UNREGISTERED' if complete else 'PARTIAL','completed_datasets':len(results),'missing_datasets':sorted(expected-set(results)),'scale':'fraction [0,1]','aggregation':'derived equal mean of pinned scorer dataset combined scores','folds':fold,'combined':{v:mean(sorted(expected),v) if complete else None for v in ['participant','prose_sensitivity']}}
