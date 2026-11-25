"""Build a configurable clean-runtime batch notebook from the approved worker."""
import ast,json
from pathlib import Path
from build_ns64_76 import ROOT,read,sha,inventory,ordered
def main():
    out=ROOT/'reports/ns64_full';out.mkdir(exist_ok=True)
    completed=read(ROOT/'reports/phase3c8/four_archive_index.json')
    from zipfile import ZipFile
    for entry in completed:
        raw=(ROOT/entry['archive']).read_bytes()
        assert sha(raw)==entry['sha256']
        with ZipFile(ROOT/entry['archive']) as z:
            assert z.testzip() is None
            assert json.loads(z.read('run.json'))['worker_sha256']==entry['worker_sha256']
    prior=read(ROOT/'reports/phase3c8/four_pilot_merge/coverage.json')
    remaining=sorted(prior['full']['missing'],key=lambda k:(len(inventory(k)),k))
    source=read(ROOT/'notebooks/step03c9_138_colab.ipynb')
    last=''.join(source['cells'][-1]['source'])
    template=ast.literal_eval(ast.parse(last).body[0].value)
    original='imc2024_dioscuri_baalshamin'
    oldpin=read(ROOT/'reports/phase3c9/preparation.json')['permutation_hash']
    workers={};pins={};private={}
    export=read(ROOT/'notebooks/step03c9_138_kaggle_export.ipynb')
    body=''.join(export['cells'][0]['source']).split('from pathlib import Path',1)[1]
    export_scripts={}
    for dataset in remaining:
        images=inventory(dataset);names=ordered(images,20261111);n=len(names)
        seq=[images[name] for name in names];pin=sha('\n'.join(seq).encode())
        private[dataset]={f'image_{i:04d}.png':name for i,name in enumerate(names)}
        worker=template.replace(original,dataset).replace(oldpin,pin).replace('range(138)',f'range({n})').replace('candidate_pairs=9453',f'candidate_pairs={n*(n-1)//2}')
        # Heritage is authorized only when it is itself the selected intact dataset.
        worker=worker.replace("'/content/imc2023_heritage',","'/content/original_training_data',")
        ast.parse(worker)
        workers[dataset]=worker
        pins[dataset]=dict(images=n,permutation_sha256=pin,worker_sha256=sha(worker.encode()))
        export_scripts[dataset]='names='+repr(names)+'\nimages='+repr(images)+'\nfrom pathlib import Path'+body.replace(original,dataset).replace('Verified 138',f'Verified {n}')
    (out/'private.json').write_text(json.dumps(private,indent=2))
    (out/'worker_pins.json').write_text(json.dumps(pins,indent=2))
    runner="""from pathlib import Path
import os,signal,subprocess,time,json,hashlib,zipfile
from google.colab import files
BASE=Path('/content/ns64_full_runs')
def run_batch(batch,session_seconds):
    if len(set(batch))!=len(batch) or any(k not in WORKERS for k in batch):
        raise ValueError('Duplicate, completed or unauthorized dataset')
    session_end=time.monotonic()+session_seconds
    BASE.mkdir(exist_ok=True)
    for dataset in batch:
        work=BASE/dataset
        archive=Path('/content')/f'imc_ns64_{dataset}_worker_20261111.zip'
        if work.exists() or archive.exists():
            print('Existing attempt requires administrative review:',dataset)
            continue
        remaining=session_end-time.monotonic()
        if remaining<8400:
            print('Session time insufficient for a new attempt:',dataset)
            break
        inputs=Path('/content')/(dataset+'_opaque.zip')
        if not inputs.is_file():
            raise FileNotFoundError(inputs)
        work.mkdir()
        script=work/'worker.py';script.write_text(WORKERS[dataset])
        assert hashlib.sha256(script.read_bytes()).hexdigest()==PINS[dataset]['worker_sha256']
        started=time.monotonic();budget=min(14400,session_end-started)
        state='FAILED';error=None
        try:
            with (work/'worker.log').open('w') as log:
                process=subprocess.Popen(['/content/ns64_clean/venv/bin/python','-I','-u',str(script),'20261111'],cwd=work,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,STAIRS_DEADLINE=str(started+budget)))
                try:
                    while process.poll() is None:
                        checkpoint=work/'current_stage.json'
                        reserve=json.loads(checkpoint.read_text())['reserve_seconds'] if checkpoint.exists() else 1200
                        if started+budget-time.monotonic()<=reserve:
                            raise subprocess.TimeoutExpired('stage deadline',budget)
                        time.sleep(1)
                    state='PASS' if process.returncode==0 else 'FAILED'
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
        except subprocess.TimeoutExpired as exc:
            state='INCOMPLETE';error=str(exc)
        finally:
            result=dict(status=state,experiment_id=f'ns64_isolated_v1_{dataset}_20261111',seed=20261111,seconds=time.monotonic()-started,budget_seconds=budget,worker_sha256=PINS[dataset]['worker_sha256'],error=error)
            (work/'run.json').write_text(json.dumps(result,indent=2))
            with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
                for p in work.iterdir():
                    if p.suffix in ('.json','.jsonl','.log') or p.name=='opaque.csv':
                        z.write(p,p.name)
            print(result,flush=True)
            files.download(str(archive))
        if state!='PASS':
            print('Preserve partial evidence. A fresh independent dataset can continue within this session budget.')
"""
    ast.parse(runner)
    def cell(s):return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True))
    nb=dict(source)
    nb['cells']=source['cells'][:3]+[
        cell("BATCH=['imc2024_dioscuri_baalshamin']\nSESSION_SECONDS=14400\nfrom google.colab import files\nfiles.upload()"),
        cell('WORKERS='+repr(workers)+'\nPINS='+repr(pins)+'\n'+runner+'\nrun_batch(BATCH,SESSION_SECONDS)')]
    (ROOT/'notebooks/step03c_full_ns64_reproduction.ipynb').write_text(json.dumps(nb))
    export['cells']=[cell("BATCH=['imc2024_dioscuri_baalshamin']\nEXPORTERS="+repr(export_scripts)+"\nassert len(BATCH)==len(set(BATCH))\nfor dataset in BATCH:\n    exec(compile(EXPORTERS[dataset],dataset,'exec'))")]
    (ROOT/'notebooks/step03c_full_ns64_kaggle_export.ipynb').write_text(json.dumps(export))
    (out/'completed_archive_index.json').write_text(json.dumps(completed,indent=2))
    print([(k,v['images']) for k,v in pins.items()])
if __name__=='__main__':main()
