"""Build isolated stairs notebook and separate trusted Kaggle input exporter."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/baselines/top10_reproduction'))
from isolated_staging import ordered

def read(path):
    return json.loads(Path(path).read_bytes())

def cell(code):
    ast.parse(code)
    return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))

def build():
    evidence=next((ROOT/'reports/phase3b_full').rglob('stairs/identity_mapping.json'))
    hashes=read(evidence.parent/'input_hashes.json')
    images={r['image']:hashes[r['opaque']]['original_sha256'] for r in read(evidence)}
    authorized={r['image'] for r in read(ROOT/'configs/baselines/organizer_v1/development_inputs.json') if r['dataset']=='stairs'}
    assert len(images)==51 and set(images)==authorized
    names=ordered(images,20261111)
    permutation=hashlib.sha256('\n'.join(images[n] for n in names).encode()).hexdigest()
    admin=ROOT/'reports/phase3c6'
    (admin/'private.json').write_text(json.dumps({f'image_{i:04d}.png':n for i,n in enumerate(names)},indent=2))
    notebook=read(ROOT/'notebooks/step03c5_ns64_neutral_orders.ipynb')
    source=''.join(notebook['cells'][1]['source'])
    worker=ast.literal_eval(ast.parse(source).body[0].value)
    assert hashlib.sha256(worker.encode()).hexdigest()=='e7b9ba71cdd42e06a58d3331ce03ba54f3777a4ea351271c3223536d129e1b80'
    pre=worker.splitlines()[20]
    assert pre.startswith('exec(')
    worker=worker[:worker.index('PINS=')]+f'PINS={{20261111:{permutation!r}}}\n'+worker.split('ASSET_MANIFEST=',1)[1].splitlines()[0].join(['ASSET_MANIFEST=','\n'])+pre+'\n'+"""environment=json.loads((work/'environment.json').read_text())
require(environment['status']=='CLEAN_VM_PRECHECK_PASS','Access or integrity check failed')
require(seed==20261111,'Unauthorized seed')
for path in ('/content/ns64_neutral_runs','/content/opaque_inputs.zip','/content/ns64/project','/proc/1/root/content/drive','/var/run/docker.sock','/content/stairs','/content/imc_ets_images.zip'):
    require(not os.path.lexists(path),'Unexpected administrative or prior run path '+path)
save('boundary.json',dict(status='PASS', path_checks=environment['administrative_paths'], separate_clean_runtime=True))
"""+worker[worker.index('import pandas as pd'):]
    worker=worker.replace("range(22)","range(51)").replace("test/ETs","test/stairs").replace("dataset='ETs'","dataset='stairs'").replace("{'ETs'}","{'stairs'}").replace('v1_ETs_','v1_stairs_').replace("sorted(Path('/content').glob('opaque_inputs*.zip'))[0]","Path('/content/stairs_opaque.zip')").replace("    for name in names:","    require(sorted(archive.namelist())==names,'Wrong archive inventory')\n    for name in names:").replace("archive.read(f'{seed}/{name}')","archive.read(name)")
    worker=worker.replace('result=fn(*args,**kwargs)',"""if name=='incremental_mapping':
                remaining=float(os.environ['STAIRS_DEADLINE'])-time.monotonic()
                row['remaining_before_sfm']=remaining
                require(remaining>210,'Insufficient time for an ETs-sized SfM attempt plus export')
            result=fn(*args,**kwargs)""")
    worker=worker.replace("save('resources.json',peaks())","""save('resources.json',dict(**peaks(),work_bytes=sum(p.stat().st_size for p in work.rglob('*') if p.is_file()),mem_available=next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))))""")
    ast.parse(worker)
    driver=source[source.index('from pathlib import Path\nfrom google.colab'):]
    driver=driver.replace(" and (content/'imc_ns64_post_staging.json').is_file()","").replace('ns64_neutral_runs','ns64_stairs_run').replace('imc_ns64_order_seed_','imc_ns64_stairs_worker_').replace('v1_ETs_','v1_stairs_').replace('1800-(time.monotonic()-start)','3570-(time.monotonic()-start)').replace('budget_seconds=1800','budget_seconds=3600').replace('start_new_session=True','start_new_session=True,env=dict(os.environ,STAIRS_DEADLINE=str(start+3600))')
    setup=read(ROOT/'notebooks/step03c4_clean_vm_dependencies.ipynb')
    gpu=setup.copy()
    gpu['cells']=setup['cells'][:3]+[cell("from google.colab import files\nassert not Path('/content/ns64_neutral_runs').exists(), 'Use a fresh runtime'\nif not Path('/content/stairs_opaque.zip').is_file():\n    files.upload()\nassert Path('/content/stairs_opaque.zip').is_file()"),cell('WORKER='+repr(worker)+'\n'+driver+"\nassert run_seed(20261111)=='PASS'")]
    (ROOT/'notebooks/step03c6_stairs_colab.ipynb').write_text(json.dumps(gpu))
    exporter="""from pathlib import Path
import hashlib,zipfile
from IPython.display import FileLink,display
prefixes=[Path('/kaggle/input/competitions'),Path('/kaggle/input')]
candidates=[prefix/'image-matching-challenge-2025-ongoing/train/stairs' for prefix in prefixes]
folder=None
for candidate in candidates:
    if not candidate.is_dir():
        print('Missing stairs folder:',candidate)
        continue
    missing=[name for name in names if not (candidate/name).is_file()]
    print('Frozen stairs files found:',len(names)-len(missing),'of',len(names),'at',candidate)
    if not missing:
        folder=candidate
        break
    print('Missing frozen filenames (first five):',missing[:5])
if folder is None:
    for prefix in prefixes:
        if prefix.is_dir():
            print('Mounted input names at',prefix,sorted(p.name for p in prefix.iterdir())[:30])
    raise RuntimeError('No complete frozen stairs inventory. Check the ongoing competition attachment.')
out=Path('/kaggle/working/stairs_opaque.zip')
with zipfile.ZipFile(out.with_suffix('.partial'),'w') as z:
    for i,name in enumerate(names):
        data=(folder/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=images[name]:
            raise ValueError('Frozen content hash mismatch: '+name)
        z.writestr(f'image_{i:04d}.png',data)
out.with_suffix('.partial').replace(out)
print('Verified 51 stairs images')
display(FileLink('stairs_opaque.zip'))
"""
    notebook['cells']=[cell('names='+repr(names)+'\nimages='+repr(images)+'\n'+exporter)]
    (ROOT/'notebooks/step03c6_stairs_kaggle_export.ipynb').write_text(json.dumps(notebook))
    print(permutation)

if __name__=='__main__':
    build()
