"""Build a self-contained source payload; no images, model weights or GPU installs."""
import base64,hashlib,io,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/baselines/organizer'))
from contracts import sha,write
def build():
    c=ROOT/'configs/baselines/organizer_v1';files=[]
    for directory in ['src/baselines/organizer','configs/benchmark/imc2025_local_v1','configs/baselines/organizer_v1']:
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in {'.py','.json','.csv'} and p.name!='source_snapshot.json')
    frozen=json.loads((ROOT/'configs/benchmark/imc2025_local_v1/benchmark.json').read_text())
    files.extend(ROOT/n for n in frozen['code_hashes'])
    files.extend(ROOT/n for n in ['scripts/evaluate_local_reference.py','scripts/benchmark_contract.py','docs/evidence/public_sources/first_metric.py.txt','docs/evidence/phase3a_sources/organizer.ipynb','docs/evidence/phase3a_sources/database.py','docs/evidence/phase3a_sources/h5_to_db.py','docs/evidence/phase3a_sources/lightglue_LICENSE'])
    files=sorted(set(files));snapshot={'git_commit':None,'reason':'No Git repository/commit created; complete executable source and configuration snapshot instead','files':{p.relative_to(ROOT).as_posix():sha(p) for p in files}}
    write(c/'source_snapshot.json',snapshot);files.append(c/'source_snapshot.json')
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            info=zipfile.ZipInfo(p.relative_to(ROOT).as_posix(),date_time=(2026,11,11,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
    raw=buf.getvalue();payload=base64.b64encode(raw).decode();pin=hashlib.sha256(raw).hexdigest()
    setup=f'''import base64, hashlib, io, json, sys, zipfile
from pathlib import Path
PAYLOAD = {payload!r}
raw = base64.b64decode(PAYLOAD)
assert hashlib.sha256(raw).hexdigest() == {pin!r}, 'Embedded source payload changed'
ROOT = Path('/kaggle/working/phase3a_source')
ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for info in z.infolist():
        target = (ROOT / info.filename).resolve()
        assert target.is_relative_to(ROOT.resolve()), 'Unsafe source path'
        target.parent.mkdir(parents=True, exist_ok=True)
        data = z.read(info.filename)
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError('Existing source differs: use a fresh Kaggle session')
        target.write_bytes(data)
print('Source snapshot verified; frozen benchmark preserved. Reporting fix: sets-and-enums-v2.')
'''
    settings='''# Select GPU T4 (one GPU is used), Internet OFF. Original wheels require Python 3.10.
# If a reviewed Python 3.10 environment exists, set PYTHON to its executable.
# Do not silently change packages or algorithms to bypass a compatibility failure.
PYTHON = sys.executable
WORK = Path('/kaggle/working/organizer_smoke')
GPU_MEMORY_GIB = 12
BUDGET_SECONDS = 1200
def mounted(*candidates):
    found = [p for p in map(Path, candidates) if p.exists()]
    # Keep the expected path if missing so Stage A records a useful failure archive.
    if len(found) > 1: raise RuntimeError('Ambiguous mounted assets; specify one exact input')
    return str(found[0] if found else Path(candidates[0]))
mounts = {
 'competition': mounted('/kaggle/input/competitions/image-matching-challenge-2025-ongoing', '/kaggle/input/image-matching-challenge-2025-ongoing'),
 'packages': mounted('/kaggle/input/imc2024-packages-lightglue-rerun-kornia', '/kaggle/input/datasets/oldufo/imc2024-packages-lightglue-rerun-kornia'),
 'dinov2': mounted('/kaggle/input/dinov2/pytorch/base/1', '/kaggle/input/models/metaresearch/dinov2/pytorch/base/1'),
 'aliked': mounted('/kaggle/input/aliked/pytorch/aliked-n16/1/aliked-n16.pth', '/kaggle/input/models/oldufo/aliked/pytorch/aliked-n16/1/aliked-n16.pth'),
 'lightglue': mounted('/kaggle/input/lightglue/pytorch/aliked/1/aliked_lightglue.pth', '/kaggle/input/models/oldufo/lightglue/pytorch/aliked/1/aliked_lightglue.pth')}
config = json.loads((ROOT/'configs/baselines/organizer_v1/smoke.json').read_text())
config.update(gpu_memory_gib=GPU_MEMORY_GIB, budget_seconds=BUDGET_SECONDS)
Path('/kaggle/working/smoke_mounts.json').write_text(json.dumps(mounts))
Path('/kaggle/working/smoke_config.json').write_text(json.dumps(config))
print('ETs: 22 images. dev_b diagnostic only. No confirmation or test images.')
'''
    launch='''import subprocess
command = [PYTHON, str(ROOT/'src/baselines/organizer/run.py'), '--root', str(ROOT),
 '--mounts', '/kaggle/working/smoke_mounts.json', '--work', str(WORK),
 '--config', '/kaggle/working/smoke_config.json']
result = subprocess.run(command)
if (WORK/'receipt.json').exists():
    receipt = json.loads((WORK/'receipt.json').read_text())
    print('Run status:', receipt['status'])
    for stage, record in receipt['stages'].items(): print(stage, record['status'], record.get('error', ''))
    print('Download /kaggle/working/imc_baseline_smoke_report.zip even when blocked or failed.')
else:
    raise RuntimeError('Runner could not start; return the notebook error. No GPU result exists.')
if result.returncode:
    print('STOP: review the failure receipt. Do not change model settings or run full folds.')
'''
    cells=[{'cell_type':'markdown','metadata':{},'source':['# Phase 3A — organizer baseline GPU smoke test\n','Attach ongoing competition data, original package bundle, DINOv2 base/1, ALIKED n16/1 and LightGlue aliked/1. Internet OFF. Single GPU, 20-minute limit. Current Python compatibility is unverified: a non-3.10 runtime deliberately reports BLOCKED. No model performance is claimed by this notebook.\n','Stages: A preflight; B DINOv2; C retrieval; D ALIKED; E LightGlue; F geometry; G SfM; H inferred clusters/outliers; I pinned diagnostic scoring; J receipt. The source payload contains only ETs ground truth for post-inference scoring, never passed to inference workers.']}]
    for code in [setup,settings,launch]:cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)})
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
    (ROOT/'notebooks/step03a_baseline_gpu_smoke.ipynb').write_text(json.dumps(nb,indent=2))
    print(json.dumps({'payload_bytes':len(raw),'payload_sha256':pin,'files':len(files)}))
if __name__=='__main__':build()
