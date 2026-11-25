"""Package frozen evaluator and one ETs inference launch without model weights."""
import base64
import io
import json
import hashlib
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
identity=json.loads((ROOT/'configs/baselines/organizer_v1/full_implementation.json').read_text())
names=set(identity['files'])
names.add('configs/baselines/organizer_v1/full_implementation.json')
names.add('experiments/baseline/organizer_baseline_v1_dev_full.json')
for p in (ROOT/'src/baselines/top10_reproduction').glob('*.py'):names.add(p.relative_to(ROOT).as_posix())
for name in ['pilot.json','assets_phase3c1.json']:names.add('configs/baselines/top10_reproduction_v1/'+name)
for name,h in identity['files'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
    for name in sorted(names):z.write(ROOT/name,name)
raw=buffer.getvalue()
cell=f"""from pathlib import Path
import base64,hashlib,io,zipfile,os,subprocess,shutil
from IPython.display import display,FileLink
work=Path('/kaggle/working')
project=work/'ns64_ets_project'
if project.exists():raise FileExistsError('Preserve this attempt. Use a new session or reviewed new workspace for retry.')
project.mkdir()
raw=base64.b64decode({base64.b64encode(raw).decode()!r})
assert hashlib.sha256(raw).hexdigest()=={hashlib.sha256(raw).hexdigest()!r}
with zipfile.ZipFile(io.BytesIO(raw)) as z:z.extractall(project)
source_matches=[]
for directory,folders,files in os.walk('/kaggle/input'):
    folders[:]=[n for n in folders if n not in ('competitions','train','test') and not n.startswith('image-matching-challenge-')]
    if 'ns64_pilot_source.zip.bin' in files:source_matches.append(Path(directory)/'ns64_pilot_source.zip.bin')
assert len(source_matches)==1,source_matches
target=project/'configs/baselines/top10_reproduction_v1/ns64_pilot_source.zip.bin'
assert hashlib.sha256(source_matches[0].read_bytes()).hexdigest()=='c6035c596a484331ebfaf8d3b6368b4a32acb00b3bebe670a8215c78b6c12549'
shutil.copyfile(source_matches[0],target)
collections=[p for p in [
    Path('/kaggle/input/competitions/image-matching-challenge-2025-ongoing/train/ETs'),
    Path('/kaggle/input/image-matching-challenge-2025-ongoing/train/ETs'),
] if p.is_dir()]
assert len(collections)==1,collections
python=work/'ns64_env/bin/python'
receipt=work/'imc_ns64_dependency_receipt.zip'
assert python.is_file() and receipt.is_file(),'Retain the validated session and receipt'
assert hashlib.sha256(receipt.read_bytes()).hexdigest()=={hashlib.sha256((ROOT/'reports/phase3c1_dependency/imc_ns64_dependency_receipt.zip').read_bytes()).hexdigest()!r}
env=os.environ.copy()
env.update(MPLBACKEND='Agg',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SCENE_SPACE_DIR_PERSISTENT='yes')
attempt=work/'ns64_real_ets_pilot'
print('Starting published ns64 ETs pipeline. Budget 1800 seconds.',flush=True)
result=subprocess.run([str(python),str(project/'src/baselines/top10_reproduction/ets_worker.py'),
    '--work',str(attempt),'--images',str(collections[0]),'--receipt',str(receipt)],env=env)
print('Launcher exit',result.returncode)
archive=attempt/'imc_ns64_ets_pilot.zip'
if archive.is_file():
    with zipfile.ZipFile(archive) as z:
        import json
        print(json.loads(z.read('result.json'))['status'])
    display(FileLink(str(archive)))
else:print('Inspect the traceback and preserve this working session.')
"""
compile(cell,'pilot','exec')
notebook=dict(nbformat=4,nbformat_minor=4,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
 cells=[dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=cell.splitlines(True))])
out=ROOT/'notebooks/step03c_ns64_real_ets_pilot.ipynb'
out.write_text(json.dumps(notebook,indent=2),encoding='utf8')
print(out, len(raw))
