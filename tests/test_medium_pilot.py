"""Check separation and the one authorized medium-dataset input contract."""
import ast
import hashlib
import json
from pathlib import Path


def test_medium_export_and_worker():
    root=Path(__file__).resolve().parents[1]
    def notebook(suffix):
        return json.loads((root/f'notebooks/step03c7_medium_{suffix}.ipynb').read_text())
    setup=ast.parse(''.join(notebook('kaggle_export')['cells'][0]['source']))
    names,images=[ast.literal_eval(line.value) for line in setup.body[:2]]
    assert len(set(names))==len(images)==54
    digest=hashlib.sha256('\n'.join(images[n] for n in names).encode()).hexdigest()
    assert digest=='f1df2126a55898bce62da9d2783aa6e14d92592fd35f4c58ebcb2473d82e0b39'
    gpu=notebook('colab')
    source=''.join(gpu['cells'][-1]['source'])
    worker=ast.literal_eval(ast.parse(source).body[0].value)
    compile(worker,'worker.py','exec')
    assert all(n not in json.dumps(gpu) for n in names)
    assert digest in worker and 'ns64_isolated_v1_imc2023_haiper_' in worker
    assert 'range(54)' in worker and source.count('assert run_seed(20261111)')==1
    assert '5400' in source and 'left<=reserve' in source

