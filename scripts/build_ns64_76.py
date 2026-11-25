"""Build one authorized 76-image run from the executed haiper notebook."""
import ast
import json
import hashlib
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/baselines/top10_reproduction'))
from isolated_staging import ordered
DATA='imc2023_theather_imc2024_church'
OLD='imc2023_haiper'
def read(p):
    return json.loads(p.read_bytes())
def sha(b):
    return hashlib.sha256(b).hexdigest()

def inventory(dataset):
    rows=[x for x in read(ROOT/'configs/baselines/organizer_v1/development_inputs.json') if x['dataset']==dataset]
    if not rows or any(x['partition'] not in ('dev_a','dev_b') for x in rows):
        raise ValueError('Development datasets only')
    evidence=next((ROOT/'reports/phase3b_full').rglob(dataset+'/identity_mapping.json'))
    hashes=read(evidence.parent/'input_hashes.json')
    images={r['image']:hashes[r['opaque']]['original_sha256'] for r in read(evidence)}
    assert set(images)=={x['image'] for x in rows} and len(images)==len(rows)
    return images

def main():
    images=inventory(DATA)
    names=ordered(images,20261111)
    assert len(names)==76
    seq=[images[n] for n in names]
    pin=sha('\n'.join(seq).encode())
    dest=ROOT/'reports/phase3c8'
    dest.mkdir(exist_ok=True)
    pins=read(ROOT/'reports/phase3c7/preparation.json')
    old_pin=pins['permutation_hash']
    org=read(ROOT/'experiments/baseline/organizer_baseline_v1_dev_full.json')['datasets'][DATA]
    assert org['partition']=='dev_a'
    pins.update(dataset=DATA,experiment_id=f'ns64_isolated_v1_{DATA}_20261111',permutation_hash=pin,ordered_content_hashes=seq,organizer=org)
    (dest/'private.json').write_text(json.dumps({f'image_{i:04d}.png':n for i,n in enumerate(names)},indent=2))
    notebooks=ROOT/'notebooks'
    exp=read(notebooks/'step03c7_medium_kaggle_export.ipynb')
    code=''.join(exp['cells'][0]['source']).split('from pathlib import Path',1)[1].replace(OLD,DATA).replace('Verified 54','Verified 76')
    code=code.replace("print('Verified 76","print('Package SHA256:',hashlib.sha256(out.read_bytes()).hexdigest())\nprint('Verified 76")
    exp['cells'][0]['source']=('names='+repr(names)+'\nimages='+repr(images)+'\nfrom pathlib import Path'+code).splitlines(True)
    (notebooks/'step03c8_76_kaggle_export.ipynb').write_text(json.dumps(exp))
    nb=read(notebooks/'step03c7_medium_colab.ipynb')
    code=''.join(nb['cells'][-1]['source'])
    w=ast.literal_eval(ast.parse(code).body[0].value)
    assert sha(w.encode())=='e3487c87a4dc9649736156632cb6ad440169e3b242b0ffe42e019de2ba940113'
    w=w.replace(OLD,DATA).replace(old_pin,pin).replace('range(54)','range(76)').replace('candidate_pairs=1431','candidate_pairs=2850')
    w=w.replace("'/content/ns64_stairs_run',",f"'/content/ns64_{OLD}_run','/content/{OLD}_opaque.zip','/content/ns64_stairs_run',")
    with ZipFile(ROOT/'reports/phase3c7/worker.zip') as previous:
        env=json.loads(previous.read('environment.json'))
    freeze=sha(env['environment_freeze'].encode())
    w=w.replace("require(seed==20261111",f"require(hashlib.sha256(environment['environment_freeze'].encode()).hexdigest()=={freeze!r} and environment['gpu']=={env['gpu']!r},'Dependency or GPU drift since approved run')\nrequire(seed==20261111")
    w=w.replace("save('identity.json',dict(method_id=",f"save('identity.json',dict(benchmark={pins['benchmark']!r},method_id=")
    w=w.replace("    require(sorted(archive.namelist())",f"    save('input_receipt.json',dict(archive_sha256=digest(Path('/content/{DATA}_opaque.zip')),permutation_sha256=PINS[seed]))\n    require(sorted(archive.namelist())")
    w=w.replace("            with (work/'stages.jsonl')","            checkpoint=work/f'checkpoint_{name}.json'\n            with checkpoint.open('x') as stream:\n                json.dump(row,stream)\n            with (work/'stages.jsonl')")
    driver=code[code.index('from pathlib import Path\nfrom google.colab'):].replace(OLD,DATA)
    driver=driver.replace("            for name in ('weight_load.json'", "            for checkpoint in sorted(work.glob('checkpoint_*.json')):\n                z.write(checkpoint,checkpoint.name)\n            for name in ('input_receipt.json','weight_load.json'")
    worker_hash=sha(w.encode())
    driver=driver.replace("    start=time.monotonic()",f"    assert hashlib.sha256(script.read_bytes()).hexdigest()=={worker_hash!r}, 'Worker integrity mismatch'\n    start=time.monotonic()")
    pins['worker_sha256']=worker_hash
    (dest/'preparation.json').write_text(json.dumps(pins,indent=2))
    nb['cells'][-2]['source']=''.join(nb['cells'][-2]['source']).replace(OLD,DATA).splitlines(True)
    nb['cells'][-1]['source']=('WORKER='+repr(w)+'\n'+driver).splitlines(True)
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    ast.parse(w)
    with ZipFile(ROOT/'imc_ns64_clean_runtime.zip') as z:
        for entry in z.namelist():
            if entry.startswith('source/'):
                b=z.read(entry)
                assert all(n.encode() not in b for n in names)
    (notebooks/'step03c8_76_colab.ipynb').write_text(json.dumps(nb))
    print(pin,worker_hash)
if __name__=='__main__':
    main()
