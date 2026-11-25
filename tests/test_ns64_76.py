import ast
import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_ns64_76 import DATA,inventory,ordered,sha,read
from merge_ns64_development import restore,aggregate,check_stages,merge

def test_76_staging():
    images=inventory(DATA)
    names=ordered(images,20261111)
    assert len(names)==76 and names==ordered(dict(reversed(list(images.items()))),20261111)
    assert sha('\n'.join(images[n] for n in names).encode())=='2133395075c25d66aaeb7d9663d8fb594b1289888871edcf1541bf15ae9d588b'
    worker=read(ROOT/'notebooks/step03c8_76_colab.ipynb')
    assert not any(n in json.dumps(worker) for n in names)
    code=''.join(worker['cells'][-1]['source'])
    assert sha(ast.literal_eval(ast.parse(code).body[0].value).encode())==read(ROOT/'reports/phase3c8/preparation.json')['worker_sha256']

def test_protected_excluded():
    with pytest.raises(ValueError):
        inventory('imc2024_lizard_pond')

@pytest.mark.parametrize('images,valid',[(['a','b'],True),(['a','a'],False),(['a'],False)])
def test_restoration(images,valid):
    df=pd.DataFrame(dict(dataset=DATA,image=images,scene='c',rotation_matrix='1;0;0;0;1;0;0;0;1',translation_vector='0;0;0'))
    args=df.to_csv(index=False).encode(),DATA,dict(a='first',b='second')
    if valid:
        assert restore(*args).image.tolist()==['first','second']
    else:
        with pytest.raises(ValueError):restore(*args)

def test_checkpoint_and_timeout():
    row=dict(stage='mapping',status='PASS',seconds=3)
    run=dict(status='PASS',seconds=5,budget_seconds=5400)
    check_stages(run,[row],{'mapping':row})
    for params in [(dict(run,status='TIMEOUT'),[row],{'mapping':row}),(run,[row],{}),(run,[row,row],{'mapping':row})]:
        with pytest.raises(ValueError):check_stages(*params)

def test_aggregate():
    scores={str(i):{'participant':{'combined_score':float(i==0)}} for i in range(11)}
    sets={'dev_a':list(scores)[:6],'dev_b':list(scores)[6:],'full':list(scores)}
    assert aggregate(scores,sets)['full']['means']['participant']['combined_score']==1/11
    scores.pop('10')
    a=aggregate(scores,sets)
    assert a['dev_a']['complete'] and a['dev_b']['means'] is None and a['full']['missing']==['10']

def test_merge_rejects_previous_output(tmp_path):
    with pytest.raises(FileExistsError):merge(tmp_path/'index',tmp_path)

@pytest.mark.parametrize('bad',['duplicate','hash'])
def test_merge_rejects_invalid_index(tmp_path,monkeypatch,bad):
    import merge_ns64_development as mod
    monkeypatch.setattr(mod.subprocess,'run',lambda *a,**kw:None)
    entries=read(ROOT/'reports/phase3c8/archive_index.json')[:1]
    if bad=='duplicate':
        entries*=2
    else:
        entries[0]['sha256']='0'*64
    idx=tmp_path/'index.json'
    idx.write_text(json.dumps(entries))
    with pytest.raises(ValueError):merge(idx,tmp_path/'new')
