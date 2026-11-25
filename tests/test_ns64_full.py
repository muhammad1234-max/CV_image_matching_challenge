import ast,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_ns64_full import read,sha
from merge_ns64_development import check_stages

def test_full_schedule_is_frozen():
    pins=read(ROOT/'reports/ns64_full/worker_pins.json')
    assert len(pins)==7 and sum(v['images'] for v in pins.values())==1360
    assert list(v['images'] for v in pins.values())==sorted(v['images'] for v in pins.values())
    rows=read(ROOT/'configs/baselines/organizer_v1/development_inputs.json')
    assert set(pins)<={r['dataset'] for r in rows if r['partition'] in ('dev_a','dev_b')}

def test_full_workers_and_retry_protection():
    nb=read(ROOT/'notebooks/step03c_full_ns64_reproduction.ipynb')
    code=''.join(nb['cells'][-1]['source']);tree=ast.parse(code)
    workers=ast.literal_eval(tree.body[0].value)
    pins=read(ROOT/'reports/ns64_full/worker_pins.json')
    for dataset,worker in workers.items():
        compile(worker,dataset,'exec')
        assert sha(worker.encode())==pins[dataset]['worker_sha256']
        assert f'ns64_isolated_v1_{dataset}_' in worker
        run=dict(experiment_id=f'ns64_isolated_v1_{dataset}_20261111',worker_sha256=pins[dataset]['worker_sha256'],status='INCOMPLETE',seconds=500,budget_seconds=14400)
        with pytest.raises(ValueError):check_stages(run,[dict(status='PASS')])
    assert "work.exists() or archive.exists()" in code
    assert "files.download(str(archive))" in code
