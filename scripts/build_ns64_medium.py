"""Prepare the trusted exporter and isolated medium-pilot worker from the executed stairs notebook."""
import ast
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/baselines/top10_reproduction'))
from isolated_staging import ordered

def read(path):
    return json.loads(path.read_bytes())

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    dest=ROOT/'reports/phase3c7'
    dest.mkdir(exist_ok=True)
    groups=defaultdict(list)
    for row in read(ROOT/'configs/baselines/organizer_v1/development_inputs.json'):
        groups[row['dataset']].append(row)
    registry=read(ROOT/'experiments/baseline/organizer_baseline_v1_dev_full.json')['datasets']
    bytes_by_path={v['path']:v['bytes'] for v in read(ROOT/'docs/evidence/kaggle_audit_received/dataset_inventory.json')['records']}
    prior={'ETs','stairs'}
    ranking=sorted(groups,key=lambda k:(len(groups[k]),k))
    dataset=next(k for k in ranking if k not in prior and len(groups[k])>51)
    assert dataset=='imc2023_haiper'
    table=['# Frozen development selection','','Dataset | Images | Partition | Possible pairs | Input bytes | Prior ns64 | Membership SHA256','---|---:|---|---:|---:|---|---']
    for k in ranking:
        rows=groups[k]
        n=len(rows)
        assert n==registry[k]['images'] and len(set(x['image'] for x in rows))==n
        total=sum(bytes_by_path[f"train/{k}/{x['image']}"] for x in rows)
        table.append(f"{k} | {n} | {rows[0]['partition']} | {n*(n-1)//2} | {total} | {k in prior} | {registry[k]['membership_hash']}")
    table+=['','Selected imc2023_haiper before inference. Its 54 images meet the primary smallest-untested-above-51 rule. The preferred 60–110 interval is secondary.']
    (ROOT/'docs/step03c7_dataset_selection.md').write_text('\n'.join(table),encoding='utf8')
    evidence=next((ROOT/'reports/phase3b_full').rglob(dataset+'/identity_mapping.json'))
    hashes=read(evidence.parent/'input_hashes.json')
    images={v['image']:hashes[v['opaque']]['original_sha256'] for v in read(evidence)}
    assert set(images)=={x['image'] for x in groups[dataset]}
    names=ordered(images,20261111)
    sequence=[images[k] for k in names]
    pin=sha('\n'.join(sequence).encode())
    (dest/'private.json').write_text(json.dumps({f'image_{i:04d}.png':k for i,k in enumerate(names)},indent=2))
    pins=read(ROOT/'reports/phase3c6/preparation.json')
    pins.update(dataset=dataset,experiment_id='ns64_isolated_v1_'+dataset+'_20261111',organizer=registry[dataset],permutation_hash=pin,ordered_content_hashes=sequence)
    (dest/'preparation.json').write_text(json.dumps(pins,indent=2))
    notebooks=ROOT/'notebooks'
    exporter=read(notebooks/'step03c6_stairs_kaggle_export.ipynb')
    body=''.join(exporter['cells'][0]['source']).split('from pathlib import Path',1)[1]
    exporter['cells'][0]['source']=('names='+repr(names)+'\nimages='+repr(images)+'\nfrom pathlib import Path'+body.replace('stairs',dataset).replace('Verified 51','Verified 54')).splitlines(True)
    (notebooks/'step03c7_medium_kaggle_export.ipynb').write_text(json.dumps(exporter))
    nb=read(notebooks/'step03c6_stairs_colab.ipynb')
    last=''.join(nb['cells'][-1]['source'])
    worker=ast.literal_eval(ast.parse(last).body[0].value)
    assert sha(worker.encode())=='54fb4002762e6395ae7a45191370334349f252ad1db7dd638d0f87a65651f812'
    worker=worker.replace('744bdb099dfac67431db563c63058f42765beb757ee6bf5a2a39adc97a500b26',pin).replace('stairs',dataset).replace('range(51)','range(54)')
    worker=worker.replace("'/content/ns64_neutral_runs',","'/content/ns64_stairs_run','/content/stairs_opaque.zip','/content/ns64_neutral_runs',")
    worker=worker.replace("    pipeline=create_pipeline", """    import shutil
    free_vram,total_vram=torch.cuda.mem_get_info()
    ram_kib=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
    disk_free=shutil.disk_usage(work).free
    scratch_estimate=max(1024**3,4*sum(p.stat().st_size for p in images.iterdir()))
    save('preflight.json',dict(gpu=torch.cuda.get_device_name(),free_vram=free_vram,total_vram=total_vram,available_ram_kib=ram_kib,disk_free=disk_free,planning_scratch_bytes=scratch_estimate,candidate_pairs=1431,remaining_seconds=float(os.environ['STAIRS_DEADLINE'])-time.monotonic()))
    require(free_vram>=9412018176 and ram_kib>=6343916 and disk_free>=scratch_estimate,'Below pilot-based resource planning floor')
    pipeline=create_pipeline""")
    worker=worker.replace("    pipeline=create_pipeline","""    import gc,pickle
    loaded=[]
    for key,asset in json.loads(assets.read_text()).items():
        folder=base/'weights'/('dinov2' if key=='dinov2' else '')
        path=folder/asset['filename']
        if path.suffix=='.pkl':
            with path.open('rb') as stream:
                payload=pickle.load(stream)
        else:
            payload=torch.load(path,map_location='cpu',weights_only=False)
        require(payload is not None,'Empty checkpoint '+key)
        del payload
        gc.collect()
        loaded.append(key)
        save('weight_load.json',dict(loaded=loaded))
    pipeline=create_pipeline""")
    worker=worker.replace("        row=dict(stage=name,status='FAILED')","""        row=dict(stage=name,status='FAILED')
        reserve={'shortlist_generator':1200,'run_point_tracking_matching_stage':600,'verify_matches':480}.get(name,60)
        remaining=float(os.environ['STAIRS_DEADLINE'])-start
        save('current_stage.json',dict(stage=name,remaining_seconds=remaining,reserve_seconds=reserve,**peaks()))
        require(remaining>reserve,'Stage start deadline exceeded: '+name)""")
    worker=worker.replace("row['verified_pairs']=result.count","row['verified_pairs']=result.count\n                save('verified_edges.json',[[str(i),str(j)] for i,g in result.inliers.items() for j,v in g.items() if len(v)])")
    driver=last[last.index('from pathlib import Path\nfrom google.colab'):]
    driver=driver.replace('stairs',dataset).replace('start+3600','start+budget').replace('3570-','budget-60-').replace('budget_seconds=3600','budget_seconds=budget').replace("    work=batch/str(seed)","    budget=min(5400,int(input('Available session seconds now (maximum 5400): ')))\n    if budget<=1200: raise ValueError('Insufficient session time; no attempt started')\n    work=batch/str(seed)").replace("'run.json','environment.json'","'preflight.json','current_stage.json','verified_edges.json','run.json','environment.json'")
    worker=worker.replace("    (work/name).write_text(json.dumps(value,indent=2))","    temp=work/(name+'.tmp')\n    temp.write_text(json.dumps(value,indent=2))\n    temp.replace(work/name)")
    driver=driver.replace("'preflight.json','current_stage.json'","'weight_load.json','preflight.json','current_stage.json'")
    driver=driver.replace("code=process.wait(timeout=max(0,budget-60-(time.monotonic()-start)))","""while process.poll() is None:
                    left=start+budget-time.monotonic()
                    stage_path=work/'current_stage.json'
                    reserve=json.loads(stage_path.read_text())['reserve_seconds'] if stage_path.exists() else 60
                    if left<=reserve:
                        raise subprocess.TimeoutExpired('stage deadline',budget-reserve)
                    time.sleep(min(1,left-reserve))
                code=process.returncode""")
    nb['cells'][-2]['source']=''.join(nb['cells'][-2]['source']).replace('stairs',dataset).splitlines(True)
    nb['cells'][-1]['source']=('WORKER='+repr(worker)+'\n'+driver).splitlines(True)
    ast.parse(worker)
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    (notebooks/'step03c7_medium_colab.ipynb').write_text(json.dumps(nb))
    print(pin)

if __name__=='__main__':
    main()
