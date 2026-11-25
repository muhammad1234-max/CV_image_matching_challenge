"""Assemble a private Kaggle upload and self-contained, validation-only notebooks."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile
from src.runner_5a5.kaggle import sha,JOURNAL

R=Path("reports/phase5a5")
def read(p):return json.loads(Path(p).read_bytes())
def save(p,data):Path(p).write_text(json.dumps(data,indent=2,sort_keys=True)+"\n",encoding="utf8")
def archive(files):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,"w") as z:
        for name,data in sorted(files.items()):
            z.writestr(zipfile.ZipInfo(name),data,zipfile.ZIP_DEFLATED)
    return stream.getvalue()
def build():
    R.mkdir(exist_ok=True)
    old=read("reports/phase5a3/revision.json")
    pins=old["scientific"]
    prev=Path("notebooks/step05a3_dioscuri_colab.ipynb")
    assert sha(prev)==old["notebook"]
    assignments={node.targets[0].id:node.value for node in ast.parse("".join(read(prev)["cells"][3]["source"])).body
                 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)}
    files={}
    for node,pin in ((assignments["EMBED"],ast.literal_eval(assignments["KIT_PIN"])),(assignments["ops"].args[0],old["runner"])):
        raw=base64.b64decode(ast.literal_eval(node))
        assert hashlib.sha256(raw).hexdigest()==pin
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for n in z.namelist():files[n]=z.read(n)
    science=read("reports/phase5a3/scientific_comparison.json")["expected"]
    checked={}
    for name,expected in science.items():
        actual=hashlib.sha256(files[name]).hexdigest() if name.startswith("workers/") or name=="worker_pins.json" else sha(name)
        assert actual==expected,name
        checked[name]=actual
    save(R/"scientific.json",checked)
    runtime=Path("imc_ns64_clean_runtime.zip")
    opaque=Path(pins["dataset"]+"_opaque.zip")
    expected=dict(ns64_runtime=read("reports/phase5a/integrity.json")["runtime_bundle"],opaque=pins["image_archive"])
    assert sha(runtime)==expected["ns64_runtime"] and sha(opaque)==expected["opaque"]
    from src.runner_5a5.kaggle import verify_opaque
    verify_opaque(opaque,pins["image_permutation"])
    with zipfile.ZipFile("reports/phase3c1_dependency/imc_ns64_dependency_receipt.zip") as z:
        historical=json.loads(z.read("result.json"))
        manifest=json.loads(z.read("manifest.json"))
    save(R/"prior_kaggle.json",dict(scope="Historical Phase 3C.1 Kaggle execution",checks=historical["checks"],status=historical["status"]))
    with zipfile.ZipFile(runtime) as z:
        wheels={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.endswith(".whl")}
        requirements=z.read("requirements.txt").decode().splitlines()
    with zipfile.ZipFile("phase4f_colab_worker.zip") as z:freeze=z.read("expected_freeze.txt")
    assert hashlib.sha256(freeze).hexdigest()==pins["environment"]
    normalized=[(s.split(" @ ",1)[0].lower()+" @ "+s.rsplit("/",1)[-1]) if " @ file:" in s else s.lower() for s in freeze.decode().splitlines()]
    cfg=dict(pins=pins,normalized_freeze=sorted(normalized),mounts={"ns64_runtime.bin":expected["ns64_runtime"],"dioscuri_opaque.bin":expected["opaque"]},
             estimated_parent=18407424*138*137//(54*53),asset_sizes=manifest["assets"],native_probes=manifest["probes"],loaders=manifest["loaders"])
    save(R/"dependency_manifest.json",dict(python="3.11.11",uv="0.13.8",wheels=wheels,requirements=requirements,environment_sha256=pins["environment"],assets=manifest["assets"]))
    journal=Path("journal-"+JOURNAL+".json")
    assert sha(journal)==JOURNAL
    for path in [prev,journal,Path("reports/phase5a3/revision.json"),Path("src/runner_5a5/kaggle.py"),Path("src/runner_5a5/commands.py"),
                 Path("scripts/phase5a5_live.py"),Path("scripts/phase5a5_persistence.py"),
                 *map(Path,["tests/phase5a3_cpu.py","tests/test_phase5a5_children.py","tests/test_phase5a3.py","tests/test_phase5a1.py"])]:
        files[path.as_posix()]=path.read_bytes()
    files["manifest.json"]=json.dumps(cfg,sort_keys=True).encode()
    bundle=archive(files)
    revision=hashlib.sha256(bundle).hexdigest()
    (R/"validation_kit.zip").write_bytes(bundle)
    upload=Path("kaggle_upload/phase5a5_private_inputs")
    upload.mkdir(exist_ok=True,parents=True)
    for f,name in [(runtime,"ns64_runtime.bin"),(opaque,"dioscuri_opaque.bin")]:
        dest=upload/name
        if not dest.exists():shutil.copyfile(f,dest)
        assert sha(dest)==cfg["mounts"][name]
    notebooks={}
    for mode in ("smoke","restore"):
        condition="None" if mode=="smoke" else "input('Paste the SHA256 independently saved from the first session: ').strip()"
        code=f"""import base64, hashlib, io, os, sys, tempfile, zipfile
from pathlib import Path
raw=base64.b64decode({base64.b64encode(bundle).decode()!r})
revision={revision!r}
assert hashlib.sha256(raw).hexdigest()==revision
with tempfile.TemporaryDirectory(prefix='ns64_compat_',dir='/tmp') as tmp:
    repo=Path(tmp)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:z.extractall(repo)
    sys.path[:0]=[str(repo),str(repo/'tests')]
    from scripts.phase5a5_live import main
    main(repo,revision,{condition})
"""
        compile(code,mode,"exec")
        path=Path(f"notebooks/step05a5_kaggle_{mode}.ipynb")
        note="Private compatibility smoke. Attach only the two approved .bin files. Enable internet for pinned setup." if mode=="smoke" else "Fresh session. Attach only phase5a5_synthetic_checkpoint.bin and use the independently saved SHA256."
        nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name="python3",display_name="Python 3",language="python")),
                cells=[dict(cell_type="markdown",id="instructions",metadata={},source=[note]),
                       dict(cell_type="code",id="validate",metadata={},outputs=[],execution_count=None,source=code.splitlines(True))])
        save(path,nb)
        notebooks[path.name]=sha(path)
    save(R/"release.json",dict(revision=revision,notebooks=notebooks,prior_release=old["runner"],journal=JOURNAL,expected_mounts=cfg["mounts"]))
if __name__=="__main__":build()
