import ast,json,hashlib,zipfile
from pathlib import Path

def cells(name):
    return [''.join(c['source']) for c in json.loads(Path('notebooks/'+name).read_text())['cells'] if c['cell_type']=='code']

def test_stairs_admin_order_and_coverage():
    code=cells('step03c6_stairs_kaggle_export.ipynb')[0]
    tree=ast.parse(code)
    names,images=[ast.literal_eval(n.value) for n in tree.body[:2]]
    assert len(set(names))==51 and set(names)==set(images)
    order=hashlib.sha256('\n'.join(images[n] for n in names).encode()).hexdigest()
    assert order=='744bdb099dfac67431db563c63058f42765beb757ee6bf5a2a39adc97a500b26'
    canonical=sorted(images,key=lambda n:(images[n],n))
    assert names==[canonical[i] for i in sorted(range(51),key=lambda i:hashlib.sha256(f'ns64-order-v1/20261111/{i}'.encode()).digest())]

def test_stairs_worker_identity_and_output_contract():
    runner=cells('step03c6_stairs_colab.ipynb')[-1]
    worker=ast.literal_eval(ast.parse(runner).body[0].value)
    tree=ast.parse(worker)
    identity=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='identity.json')
    assert 'ns64_isolated_v1_stairs_' in ast.unparse(identity)
    valid=next(n.args[0] for n in ast.walk(tree) if isinstance(n,ast.Call) and len(n.args)==2 and isinstance(n.args[1],ast.Constant) and n.args[1].value=='Invalid prediction coverage')
    import pandas as pd
    names=[f'image_{i:04d}.png' for i in range(51)]
    for rows,ok in [(names,True),(names[:-1],False),(names[:-1]+names[:1],False)]:
        result=pd.DataFrame(dict(image=rows,dataset='stairs'))
        assert eval(compile(ast.Expression(valid),'contract','eval'),{'result':result,'names':names})==ok
