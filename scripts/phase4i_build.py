"""Rebuild a label-free notebook from two frozen worker texts. Run from the project root."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
from src.novel_model.final_candidate_v1 import POLICY, file_sha

OUT = Path("reports/phase4i")
NB = Path("notebooks/step04i_final_candidate_colab.ipynb")

def encoded(x):
    return json.dumps(x,sort_keys=True,indent=2).encode()

def digest(b):
    return hashlib.sha256(b).hexdigest()

def cell(code):
    ast.parse(code)
    return dict(cell_type="code",execution_count=None,metadata={},outputs=[],source=code.splitlines(True))

def build():
    OUT.mkdir(exist_ok=True)
    nb=json.loads(Path("notebooks/step03c_full_ns64_reproduction.ipynb").read_bytes())
    workers=ast.literal_eval(ast.parse("".join(nb["cells"][-1]["source"])).body[0].value)
    pins=json.loads(Path("reports/ns64_full/worker_pins.json").read_bytes())
    paths=list(Path("src/novel_model/final_candidate_v1").glob("*.py"))
    paths += [Path("src/novel_model")/s for s in (
        "geometry_graph_v1/schemas.py","geometry_graph_v1/spectral_proposal.py","geometry_capture_v1/database.py",
        "reconstruction_feedback_v1/__init__.py","reconstruction_feedback_v2/__init__.py","resource_monitor_v2/__init__.py")]
    paths += [Path("configs/novel_model")/(s+".yaml") for s in
              ("final_candidate_v1","phase4g1_resource_policy","reconstruction_feedback_v2")]
    contents={p.as_posix():p.read_bytes() for p in paths}
    for ds in POLICY["development"]:
        data=workers[ds].encode()
        if digest(data)!=pins[ds]["worker_sha256"]: raise ValueError(ds)
        contents["workers/"+ds+".py"]=data
    contents["worker_pins.json"]=encoded({ds:pins[ds] for ds in POLICY["development"]})
    contents["inference_pins.json"]=encoded({k:digest(v) for k,v in sorted(contents.items())})
    kit=io.BytesIO()
    with zipfile.ZipFile(kit,"w") as z:
        for name,data in sorted(contents.items()):
            z.writestr(zipfile.ZipInfo(name),data,compress_type=zipfile.ZIP_DEFLATED)
    nb["metadata"]["colab"]["name"]="Phase 4I prospective development evaluation"
    nb["cells"]=nb["cells"][:3]
    nb["cells"][0]["source"]=["import time\n","SESSION_START=time.monotonic()\n"]+nb["cells"][0]["source"]
    install="""import base64,io,time
import hashlib,sys,zipfile
from pathlib import Path
from google.colab import files
raw=base64.b64decode(EMBED)
assert hashlib.sha256(raw).hexdigest()==KIT_PIN
kit=Path('/content/phase4i_kit')
with zipfile.ZipFile(io.BytesIO(raw)) as z:z.extractall(kit)
sys.path.insert(0,str(kit))
from src.novel_model.final_candidate_v1.session import stage_run
dataset='imc2024_dioscuri_baalshamin'  # Second independent group: fbk_vineyard
archive=Path('/content')/(dataset+'_opaque.zip')
if not archive.exists():files.upload()
assert archive.is_file(),archive
remaining=int(input('Seconds actually available in this free session now: '))
deadline=min(SESSION_START+14400,time.monotonic()+remaining)
root=Path('/content/phase4i_runs')/dataset
def run(stage):
    # To resume on a new VM, upload earlier downloads using Colab Files.
    # The original archive name and its independently saved SHA-256 are required.
    checkpoint=Path('/content')/(dataset+'_'+stage+'.zip')
    restore=None
    if checkpoint.exists() and not (root/stage).exists():
        restore=(checkpoint,input('Original saved SHA-256 for '+stage+': ').strip())
    stage_run(dataset,stage,root,deadline,restore)
"""
    nb["cells"].append(cell("EMBED="+repr(base64.b64encode(kit.getvalue()).decode())+"\nKIT_PIN="+
                             repr(digest(kit.getvalue()))+"\n"+install))
    for stage in ("capture","original","trial","decision"):
        nb["cells"].append(cell("run("+repr(stage)+")"))
    for c in nb["cells"]: c.update(outputs=[],execution_count=None)
    NB.write_bytes(encoded(nb))
    (OUT/"inference_pins.json").write_bytes(contents["inference_pins.json"])
    lockpaths=[NB,Path(__file__).relative_to(Path.cwd()) if Path(__file__).is_absolute() else Path(__file__),
       Path("scripts/phase4i_evaluate.py"),Path("experiments/baseline/organizer_baseline_v1_dev_full.json"),
       Path("configs/baselines/ns64_isolated_v1/experiment.json"),Path("reports/ns64_full/worker_pins.json"),
       Path("src/baselines/development/evaluate.py"),Path("src/baselines/development/common.py")]
    lockpaths += list(Path("configs/benchmark/imc2025_local_v1").glob("*.json"))
    lockpaths += [Path("configs/baselines/organizer_v1")/(n+".json") for n in
                  ("full_development","full_environment_lock","full_checkpoint_lock","full_implementation","development_inputs")]
    lockpaths += [Path("snapshots/organizer_phase3a_verified/scripts")/p for p in ("benchmark_contract.py","evaluate_local_reference.py")]
    group_meta=json.loads(Path("configs/benchmark/imc2025_local_v1/source_groups.json").read_bytes())["mapping"]
    freeze=dict(version=POLICY["version"],protocol=digest(contents["inference_pins.json"]),
                config=file_sha(Path("configs/novel_model/final_candidate_v1.yaml")),
                inference_kit=digest(kit.getvalue()),training=[],order=list(POLICY["development"]),
                datasets={},files={p.as_posix():file_sha(p) for p in lockpaths})
    for ds,group in POLICY["development"].items():
        meta=next(x for x in group_meta if x["dataset"]==ds)
        assert meta["source_group"]==group and meta["partition"] in ("dev_a","dev_b")
        organizer=next(Path("reports/phase3b_full/evidence").rglob(ds+"/predictions.csv"))
        freeze["datasets"][ds]=dict(**pins[ds],group=group,partition=meta["partition"],
                                    organizer_prediction=organizer.as_posix(),organizer_sha256=file_sha(organizer))
    (OUT/"freeze.json").write_bytes(encoded(freeze))

if __name__=="__main__": build()
