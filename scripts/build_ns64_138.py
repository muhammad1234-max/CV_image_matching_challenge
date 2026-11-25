"""Prepare the one authorized 138-image reproduction from the executed 76-image worker."""
import ast,json
from pathlib import Path
from zipfile import ZipFile
from build_ns64_76 import ROOT,read,sha,inventory,ordered
DATA='imc2024_dioscuri_baalshamin'
OLD='imc2023_theather_imc2024_church'

def main():
    images=inventory(DATA)
    assert len(images)==138
    groups={v['dataset']:v for v in read(ROOT/'configs/benchmark/imc2025_local_v1/source_groups.json')['mapping']}
    assert all(groups[k]['source_group']=='heritage_dioscuri' and groups[k]['partition']=='dev_a' for k in (DATA,'imc2023_heritage'))
    names=ordered(images,20261111)
    seq=[images[n] for n in names]
    pin=sha('\n'.join(seq).encode())
    dest=ROOT/'reports/phase3c9';dest.mkdir(exist_ok=True)
    prior=read(ROOT/'reports/phase3c8/preparation.json')
    pins=dict(prior,dataset=DATA,experiment_id=f'ns64_isolated_v1_{DATA}_20261111',permutation_hash=pin,ordered_content_hashes=seq,organizer=read(ROOT/'experiments/baseline/organizer_baseline_v1_dev_full.json')['datasets'][DATA])
    (dest/'private.json').write_text(json.dumps({f'image_{i:04d}.png':n for i,n in enumerate(names)},indent=2))
    notebooks=ROOT/'notebooks'
    export=read(notebooks/'step03c8_76_kaggle_export.ipynb')
    body=''.join(export['cells'][0]['source']).split('from pathlib import Path',1)[1]
    export['cells'][0]['source']=('names='+repr(names)+'\nimages='+repr(images)+'\nfrom pathlib import Path'+body.replace(OLD,DATA).replace('Verified 76','Verified 138')).splitlines(True)
    (notebooks/'step03c9_138_kaggle_export.ipynb').write_text(json.dumps(export))
    nb=read(notebooks/'step03c8_76_colab.ipynb')
    code=''.join(nb['cells'][-1]['source'])
    worker=ast.literal_eval(ast.parse(code).body[0].value)
    assert sha(worker.encode())==prior['worker_sha256']
    worker=worker.replace(OLD,DATA).replace(prior['permutation_hash'],pin).replace('range(76)','range(138)').replace('candidate_pairs=2850','candidate_pairs=9453')
    worker=worker.replace("'/content/ns64_stairs_run',",f"'/content/ns64_{OLD}_run','/content/{OLD}_opaque.zip','/content/imc2023_heritage','/content/ns64_stairs_run',")
    worker=worker.replace("reserve={'shortlist_generator':1200,'run_point_tracking_matching_stage':600,'verify_matches':480}.get(name,60)","reserve={'shortlist_generator':6600,'run_point_tracking_matching_stage':5520,'verify_matches':5520}.get(name,1200)")
    worker=worker.replace("require(remaining>210","require(remaining>5520").replace('Insufficient time for an ETs-sized SfM attempt plus export','Insufficient time for pair-scaled SfM planning and 20-minute export reserve')
    worker=worker.replace("ram_kib>=6343916","ram_kib>=8388608")
    # Check live memory, disk and the user-declared session deadline at expensive stage boundaries.
    worker=worker.replace("        try:\n            if name=='incremental_mapping':","""        try:
            if name in ('shortlist_generator','run_point_tracking_matching_stage','incremental_mapping'):
                import shutil
                free_gpu,total_gpu=torch.cuda.mem_get_info()
                available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
                row.update(free_gpu_bytes=free_gpu,available_ram_kib=available,disk_free=shutil.disk_usage(work).free,remaining_seconds=remaining)
                require(available>1048576 and row['disk_free']>1073741824,'Resource headroom exhausted')
            if name=='incremental_mapping':""")
    driver=code[code.index('from pathlib import Path\nfrom google.colab'):].replace(OLD,DATA)
    driver=driver.replace('5400','14400').replace('budget<=1200','budget<8400').replace('else 60','else 1200')
    driver=driver.replace(prior['worker_sha256'],sha(worker.encode()))
    pins.update(worker_sha256=sha(worker.encode()),budget_seconds=14400,minimum_start_seconds=8400,export_reserve_seconds=1200)
    nb['cells'][-2]['source']=''.join(nb['cells'][-2]['source']).replace(OLD,DATA).splitlines(True)
    nb['cells'][-1]['source']=('WORKER='+repr(worker)+'\n'+driver).splitlines(True)
    ast.parse(worker)
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    with ZipFile(ROOT/'imc_ns64_clean_runtime.zip') as z:
        for n in z.namelist():
            if n.startswith('source/'):assert all(k.encode() not in z.read(n) for k in names)
    (notebooks/'step03c9_138_colab.ipynb').write_text(json.dumps(nb))
    (dest/'preparation.json').write_text(json.dumps(pins,indent=2))
    print(pin,sha(worker.encode()))
if __name__=='__main__':main()
