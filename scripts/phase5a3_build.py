"""Prepare operational notebook and verify local scientific pins, without invoking inference."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import io
import zipfile

def read(p):
    return json.loads(Path(p).read_bytes())

def sha(data):
    return hashlib.sha256(data).hexdigest()

def save(p,obj):
    Path(p).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf8")

def build():
    p=read("reports/phase5a/integrity.json")
    nb=read("notebooks/step04i_final_candidate_colab.ipynb")
    src="".join(nb["cells"][3]["source"])
    embedded=base64.b64decode(ast.literal_eval(ast.parse(src).body[0].value))
    assert sha(embedded)==p["kit_sha256"]
    expected={**p["frozen_file_pins"],**read("reports/phase4i/inference_pins.json")}
    expected["reports/phase4i/inference_pins.json"]=p["protocol_sha256"]
    expected["reports/phase4i/freeze.json"]=p["freeze_sha256"]
    with zipfile.ZipFile(io.BytesIO(embedded)) as z:
        actual={k:sha(z.read(k) if k=="worker_pins.json" or k.startswith("workers/") else Path(k).read_bytes()) for k in expected}
    assert actual==expected
    save("reports/phase5a3/scientific_comparison.json",dict(expected=expected,observed=actual))
    keys="images dataset worker protocol_sha256 config_sha256 source_revision image_permutation published_config benchmark image_archive".split()
    pins={k:p[k] for k in keys}
    pins.update(options_sha256=sha(read("reports/phase4f/plan.json")["options"].encode()),
        source_inventory=read("reports/phase3c9/preparation.json")["source_inventory_hash"],
        environment="8bb71679bf5735f6e3c8225885be88c1f530639094df310a5a29e6691498455b",
        weight_pins=read("configs/baselines/ns64_isolated_v1/experiment.json")["assets"])
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,"w") as z:
        for f in sorted([*Path("src/runner_5a1").glob("*.py"),*Path("src/runner_5a3").glob("*.py")]):
            z.writestr(zipfile.ZipInfo(f.as_posix()),f.read_bytes(),zipfile.ZIP_DEFLATED)
    Path("reports/phase5a3/runner.zip").write_bytes(payload.getvalue())
    revision=sha(payload.getvalue())
    src=src[:src.index("from src.novel_model.final_candidate_v1.session import stage_run")]+src[src.index("dataset="):src.index("def run(stage):")]
    src=src[:src.index("root=Path")] + "root=Path('/content/phase5a3_runs')/dataset\n"
    src+=f"ops=base64.b64decode({base64.b64encode(payload.getvalue()).decode()!r})\nassert hashlib.sha256(ops).hexdigest()=={revision!r}\n"
    src+="with zipfile.ZipFile(io.BytesIO(ops)) as z:z.extractall(kit)\nfrom src.runner_5a3.colab import interface\n"
    src+=f"run=interface(root,deadline,{pins!r},{revision!r})\n"
    nb["cells"][3]["source"]=src.splitlines(True)
    for c in nb["cells"]:
        ast.parse("".join(c["source"]))
        c.update(outputs=[],execution_count=None)
    notebook="notebooks/step05a3_dioscuri_colab.ipynb"
    save(notebook,nb)
    save("reports/phase5a3/revision.json",dict(runner=revision,notebook=sha(Path(notebook).read_bytes()),scientific=pins))
if __name__=="__main__":
    build()
