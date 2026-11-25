import ast,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_ns64_138 import DATA,inventory,ordered,sha,read
from merge_ns64_development import check_stages

def test_138_provenance_order():
    images=inventory(DATA);heritage=inventory('imc2023_heritage')
    assert len(images)==138 and len(heritage)==209
    assert sum(a==b for a in images.values() for b in heritage.values())==8
    names=ordered(images,20261111)
    assert names==ordered(dict(reversed(list(images.items()))),20261111)
    assert sha('\n'.join(images[n] for n in names).encode())=='1b7bc97f6f54a2f5daa967a06e67211cd565f84d2d9403cd60f1faa50af543b0'
    nb=read(ROOT/'notebooks/step03c9_138_colab.ipynb')
    assert not any(n in json.dumps(nb) for n in names)
    code=''.join(nb['cells'][-1]['source']);w=ast.literal_eval(ast.parse(code).body[0].value)
    assert sha(w.encode())==read(ROOT/'reports/phase3c9/preparation.json')['worker_sha256']
    assert 'range(138)' in w and 'candidate_pairs=9453' in w
    assert 'else 1200' in code and 'budget<8400' in code
    assert f'ns64_isolated_v1_{DATA}_' in w

def test_138_budget():
    row=dict(stage='mapping',status='PASS')
    run=dict(experiment_id=f'ns64_isolated_v1_{DATA}_20261111',status='PASS',seconds=7000,budget_seconds=14400)
    check_stages(run,[row],{'mapping':row})
    for bad in [dict(run,status='TIMEOUT'),dict(run,seconds=14401),dict(run,experiment_id='other')]:
        with pytest.raises(ValueError):check_stages(bad,[row],{'mapping':row})

def test_138_restoration_and_rejection():
    import pandas as pd
    from merge_ns64_development import restore
    mapping={f'image_{i:04d}.png':f'original_{i:04d}' for i in range(138)}
    frame=pd.DataFrame(dict(dataset=DATA,image=list(mapping),scene='outliers',rotation_matrix='nan;nan;nan;nan;nan;nan;nan;nan;nan',translation_vector='nan;nan;nan'))
    assert len(restore(frame.to_csv(index=False).encode(),DATA,mapping))==138
    for bad in [frame.iloc[:-1],pd.concat([frame.iloc[:-1],frame.iloc[:1]])]:
        with pytest.raises(ValueError):restore(bad.to_csv(index=False).encode(),DATA,mapping)
