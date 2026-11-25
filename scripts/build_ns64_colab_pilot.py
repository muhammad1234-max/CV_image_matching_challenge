"""Build portable Colab setup using the verified Kaggle dependency inventory."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
receipt=ROOT/'reports/phase3c1_dependency/imc_ns64_dependency_receipt.zip'
with zipfile.ZipFile(receipt) as z:record=json.loads(z.read('result.json'))
identity=json.loads((ROOT/'configs/baselines/organizer_v1/full_implementation.json').read_text())
names=set(identity['files'])|{'configs/baselines/organizer_v1/full_implementation.json','experiments/baseline/organizer_baseline_v1_dev_full.json'}
names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'src/baselines/top10_reproduction').glob('*.py'))
names.update('configs/baselines/top10_reproduction_v1/'+n for n in ['assets_phase3c1.json','pilot.json','ns64_pilot_source.zip.bin'])
names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'kaggle_upload/ns64_linux_cp311').glob('*.whl'))
files={n:(ROOT/n).read_bytes() for n in names}
files['colab/kaggle_environment.txt']=record['environment_freeze'].encode()
files['colab/source_dependency_receipt.zip']=receipt.read_bytes()
hashes={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
files['colab/package_manifest.json']=json.dumps(hashes,indent=2).encode()
out=ROOT/'imc_ns64_colab_migration.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for n,b in sorted(files.items()):z.writestr(n,b)
bundlehash=hashlib.sha256(out.read_bytes()).hexdigest()
setup=f"""from pathlib import Path
import os,sys,subprocess,zipfile,hashlib,json
from google.colab import files,drive
base=Path('/content/ns64')
base.mkdir(exist_ok=True)
bundle=Path('/content/imc_ns64_colab_migration.zip')
if not bundle.exists(): files.upload()  # Select only the migration ZIP.
assert bundle.is_file(),'Upload imc_ns64_colab_migration.zip'
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=={bundlehash!r}
project=base/'project'
with zipfile.ZipFile(bundle) as z:
    manifest=json.loads(z.read('colab/package_manifest.json'))
    assert set(z.namelist())==set(manifest)|{{'colab/package_manifest.json'}}
    for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    z.extractall(project)
subprocess.run(['nvidia-smi'],check=True)
drive.mount('/content/drive')
persistent=Path('/content/drive/MyDrive/IMC_ns64')
persistent.mkdir(exist_ok=True)
bootstrap=base/'bootstrap'
subprocess.run([sys.executable,'-m','pip','install','--target',str(bootstrap),'uv'],check=True)
setup_env=os.environ.copy()
setup_env.update(PYTHONPATH=str(bootstrap),UV_PYTHON_INSTALL_DIR=str(base/'python'),UV_CACHE_DIR=str(base/'uv_cache'))
uv=[sys.executable,'-m','uv']
subprocess.run(uv+['python','install','3.11.11'],env=setup_env,check=True)
python=base/'venv/bin/python'
if not python.exists():subprocess.run(uv+['venv','--python','3.11.11','--seed',str(base/'venv')],env=setup_env,check=True)
subprocess.run([str(python),'-m','pip','install','setuptools==75.8.0','wheel==0.45.1'],check=True)
published=base/'published'
with zipfile.ZipFile(project/'configs/baselines/top10_reproduction_v1/ns64_pilot_source.zip.bin') as z:z.extractall(published)
# Resolve the original file:// wheel requirements to these verified local wheels.
wheel_paths=list((published/'bundle/oss').rglob('*.whl'))+list((project/'kaggle_upload/ns64_linux_cp311').glob('*.whl'))
import email
wheel_names={{}}
for w in wheel_paths:
    with zipfile.ZipFile(w) as z:
        metadata=email.message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))))
        wheel_names[metadata['Name'].lower().replace('_','-')]=str(w)
requirements=[]
for line in (project/'colab/kaggle_environment.txt').read_text().splitlines():
    if ' @ ' in line:
        name=line.split(' @ ')[0].lower().replace('_','-')
        assert name in wheel_names,('Unresolved original local package',name)
        requirements.append(wheel_names[name])
    elif line.strip():requirements.append(line)
requirement_file=base/'requirements.txt'
requirement_file.write_text('\\n'.join(requirements)+'\\n')
subprocess.run([str(python),'-m','pip','install','--no-deps','--extra-index-url','https://download.pytorch.org/whl/cu121','-r',str(requirement_file)],check=True)
env=os.environ.copy()
env.update(MPLBACKEND='Agg',NS64_WORK_ROOT=str(base))
subprocess.run([str(python),'-c',"import torch,asmk.hamming,curope;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0));assert torch.cuda.is_available()"],env=env,check=True)
(persistent/'colab_environment.txt').write_text(subprocess.check_output([str(python),'-m','pip','freeze'],text=True))
print('Initial environment imports passed. Actual Colab model verification is next.')
"""
assets="""import urllib.request,shutil
manifest=json.loads((project/'configs/baselines/top10_reproduction_v1/assets_phase3c1.json').read_text())
asset_dir=persistent/'weights'
asset_dir.mkdir(exist_ok=True)
def download(url,path,expected):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with path.open('rb') as handle: actual=hashlib.file_digest(handle,'sha256').hexdigest()
        assert actual==expected,path
        return
    partial=path.with_suffix(path.suffix+'.partial')
    print('Downloading',path.name,flush=True)
    digest=hashlib.sha256()
    with urllib.request.urlopen(url,timeout=120) as response,partial.open('wb') as output:
        while chunk:=response.read(4*1024**2):
            digest.update(chunk);output.write(chunk)
    assert digest.hexdigest()==expected,path
    partial.replace(path)
for key,a in manifest['assets'].items():
    if key=='dinov2':
        path=asset_dir/'dinov2'/a['filename']
        url='https://huggingface.co/facebook/dinov2-base/resolve/main/pytorch_model.bin'
        for name,h in a['sidecars'].items():
            download('https://huggingface.co/facebook/dinov2-base/resolve/main/'+name,path.parent/name,h)
    else:path=asset_dir/a['filename'];url=a['url']
    download(url,path,a['sha256'])
print('All seven asset hashes verified.')
"""
verify=r"""import textwrap
worker=base/'verify.py'
code=(project/'src/baselines/top10_reproduction/verify_dependencies.py').read_text()
receipt=base/'imc_ns64_dependency_receipt.zip'
assert not receipt.exists(),'Preserve previous receipt before a reviewed retry'
code+='\nmanifest='+repr(manifest)+'\n'
code+='print(verify(manifest,[Path('+repr(str(asset_dir))+')],Path('+repr(str(project/'configs/baselines/top10_reproduction_v1/ns64_pilot_source.zip.bin'))+'),Path('+repr(str(receipt))+'))["status"])\n'
worker.write_text(code)
env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import signal
child=subprocess.Popen([str(python),str(worker)],env=env,start_new_session=True)
try:
    child.wait(timeout=1800)
except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGKILL);child.wait()
finally:
    if receipt.exists():shutil.copyfile(receipt,persistent/'imc_ns64_colab_dependency_receipt.zip')
assert receipt.exists(),'Inspect the verifier error'
with zipfile.ZipFile(receipt) as z:status=json.loads(z.read('result.json'))['status']
print('Colab dependencies:',status)
assert status=='PASS','Download the Colab dependency receipt for review before inference'
"""
pilot="""# Upload only the 22-image ETs ZIP from the supplied Kaggle CPU export instructions.
etszip=Path('/content/imc_ets_images.zip')
if not etszip.exists():files.upload()
assert etszip.exists()
plan=json.loads((project/'configs/baselines/top10_reproduction_v1/pilot.json').read_text())
image_dir=base/'data/train/ETs'
image_dir.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(etszip) as z:
    assert set(z.namelist())==set(plan['images']),'Unexpected image membership'
    for name,h in plan['images'].items():
        raw=z.read(name)
        assert hashlib.sha256(raw).hexdigest()==h,name
        (image_dir/name).write_bytes(raw)
attempt=base/'ets_attempt'
assert not attempt.exists(),'Preserve existing attempt for review'
env['SCENE_SPACE_DIR_PERSISTENT']='yes'
try:
    subprocess.run([str(python),str(project/'src/baselines/top10_reproduction/ets_worker.py'),
        '--work',str(attempt),'--images',str(image_dir),'--receipt',str(receipt)],env=env,check=True)
finally:
    archive=attempt/'imc_ns64_ets_pilot.zip'
    if archive.exists():
        shutil.copyfile(archive,persistent/archive.name)
        files.download(str(archive))
"""
codes=[setup,assets,verify,pilot]
for c in codes:compile(c,'colab-cell','exec')
nb=dict(nbformat=4,nbformat_minor=4,metadata={'colab':{'name':'ns64 ETs reproduction'},'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}},
cells=[dict(cell_type='markdown',metadata={},source=['# ns64 ETs pilot on Colab\nSelect an NVIDIA GPU runtime. Run each cell sequentially and inspect its result. Preparation is separate from the 1800-second ETs budget. Keep Drive weights private under upstream terms. Migration validation remains pending.'])]+[dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=c.splitlines(True)) for c in codes])
(ROOT/'notebooks/step03c_ns64_colab_ets_pilot.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf8')
print(out.stat().st_size,bundlehash)
