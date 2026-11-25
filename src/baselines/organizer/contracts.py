"""CPU-only isolation, serialization and provenance contracts for Phase 3A."""
import csv,hashlib,json,math,random,enum
from pathlib import Path
BENCHMARK='a78f0542413b8ac2ec8c262988ed5303c934490a2b31d2a45d3939dd6d3ab2a4'
SPLIT='170546743202f21b9bcc98376ff09a6eae2da16ce833115478a976c4f46b2057'
FIELDS=['dataset','scene','image','rotation_matrix','translation_vector']
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def json_default(value):
    """PyCOLMAP todict() includes sets such as image_names; preserve their values.

    Sort by canonical JSON so unordered options have reproducible report/cache hashes.
    Keep unknown types strict rather than hiding reporting errors with default=str.
    """
    if isinstance(value,(set,frozenset)):
        return sorted(value,key=lambda item:json.dumps(item,sort_keys=True,allow_nan=False,default=json_default))
    if isinstance(value,enum.Enum):
        return {'enum_type':type(value).__module__+'.'+type(value).__name__,'name':value.name,'value':value.value}
    # Pybind11 enums in the pinned PyCOLMAP predate Python's enum.Enum.
    if type(value).__module__.split('.')[0]=='pycolmap':
        if isinstance(getattr(value,'name',None),str):
            return {'enum_type':type(value).__module__+'.'+type(value).__name__,'name':value.name,'value':int(value)}
        if callable(getattr(value,'todict',None)):
            return value.todict()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False,default=json_default).encode()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False,default=json_default));temp.replace(path)
def verify_benchmark(root):
    root=Path(root);c=root/'configs/benchmark/imc2025_local_v1';b=json.loads((c/'benchmark.json').read_text());h=b.pop('benchmark_hash')
    if h!=BENCHMARK or fingerprint(b)!=BENCHMARK or b['split_hash']!=SPLIT or b['status']!='FROZEN':raise ValueError('Frozen benchmark mismatch')
    for n,h in b['file_hashes'].items():
        if sha(c/n)!=h:raise ValueError('Frozen file changed: '+n)
    for n,h in b['code_hashes'].items():
        if sha(root/n)!=h:raise ValueError('Frozen code changed: '+n)
    b['benchmark_hash']=BENCHMARK;return b
def selection(root,dataset='ETs'):
    verify_benchmark(root)
    if dataset!='ETs':raise ValueError('Phase 3A authorizes only preselected intact ETs')
    rows=json.loads((Path(root)/'configs/benchmark/imc2025_local_v1/image_membership.json').read_text())['records'];selected=[r for r in rows if r['dataset']==dataset]
    if len(selected)!=22 or any(r['partition']!='dev_b' or r['fold_a_role']!='training' or r['fold_b_role']!='evaluation' for r in selected):raise ValueError('Smoke selection or role mismatch')
    result=[{'dataset':r['dataset'],'image':r['image']} for r in selected];random.Random(20261111).shuffle(result)
    return [dict(r,opaque=f'image_{i:04d}.png') for i,r in enumerate(result)]
def serialize(mapping,poses):
    names={r['opaque'] for r in mapping}
    if len(names)!=len(mapping) or set(poses)-names:raise ValueError('Duplicate/mismatched inference identities')
    output=[]
    for r in mapping:
        p=poses.get(r['opaque']);row={'dataset':r['dataset'],'image':r['image']}
        if p is None:row.update(scene='outliers',rotation_matrix=';'.join(['nan']*9),translation_vector=';'.join(['nan']*3))
        else:
            R=p['rotation'];t=p['translation']
            if len(R)!=9 or len(t)!=3 or not all(math.isfinite(v) for v in R+t):raise ValueError('Malformed predicted pose')
            row.update(scene='cluster'+str(p['cluster']),rotation_matrix=';'.join(f'{v:.09f}' for v in R),translation_vector=';'.join(f'{v:.09f}' for v in t))
        output.append(row)
    return output
def write_predictions(path,rows):
    if len({(r['dataset'],r['image']) for r in rows})!=len(rows):raise ValueError('Duplicate prediction')
    with Path(path).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
def cache_valid(directory,key):
    p=Path(directory)/'checkpoint.json'
    if not p.exists():return False
    r=json.loads(p.read_text())
    return r.get('status')=='PASS' and r.get('cache_key')==key and all((Path(directory)/n).is_file() and sha(Path(directory)/n)==h for n,h in r['artifacts'].items())
def require_files(paths):
    for p in paths:
        if not Path(p).is_file():raise FileNotFoundError('Required mounted asset missing: '+str(p))
    return {str(p):sha(p) for p in paths}
