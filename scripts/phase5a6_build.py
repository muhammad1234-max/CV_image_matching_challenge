"""Produce a deterministic prospective release from verified compatibility bytes."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile
from scripts.phase5a5_build import archive,read,save,sha
from src.runner_5a5.kaggle import verify_opaque,JOURNAL,OLD

R=Path("reports/phase5a6")
ADAPTER="e14229b7498c1ded9e36d5366d431486474d393e69f688907888dc59593e9163"
def build():
    R.mkdir(exist_ok=True)
    prior=Path("reports/phase5a5/validation_kit.zip")
    assert sha(prior)==ADAPTER
    with zipfile.ZipFile(prior) as z:
        files={n:z.read(n) for n in z.namelist()}
    smoke=read("phase5a5_kaggle_smoke.json")
    restore=read("phase5a5_kaggle_restore.json")
    checkpoint=smoke["checks"]["synthetic_checkpoint"]
    assert sha("phase5a5_synthetic_checkpoint.bin")==checkpoint["external_sha256"]=="72efbc9ee40d2045a5705b73be42978b9af08b340f4a4651b462fcf097bbfdcc"
    assert smoke["adapter"]==restore["adapter"]==ADAPTER and restore["status"]=="SYNTHETIC_CUSTODY_RESTORED"
    assert smoke["status"]=="SMOKE_COMPLETE_AWAIT_EXTERNAL_RESTORE" and restore["checks"]["fresh_session_restore"]==checkpoint["local_restore"]
    prev=read("reports/phase5a3/revision.json")
    nb=read("notebooks/step05a3_dioscuri_colab.ipynb")
    assign={n.targets[0].id:n.value for n in ast.parse("".join(nb["cells"][3]["source"])).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    digest=lambda data:hashlib.sha256(data).hexdigest()
    assert digest(base64.b64decode(ast.literal_eval(assign["ops"].args[0])))==prev["runner"]==OLD
    cfg=json.loads(files["manifest.json"])
    for name,pin in cfg["mounts"].items():
        assert sha(Path("kaggle_upload/phase5a5_private_inputs")/name)==pin
    verify_opaque(Path("kaggle_upload/phase5a5_private_inputs/dioscuri_opaque.bin"),cfg["pins"]["image_permutation"])
    journal="journal-"+JOURNAL+".json"
    assert sha(journal)==digest(files[journal])==JOURNAL and len(read(journal))==1
    assert list(Path(".").glob("journal-*.json"))==[Path(journal)],"reconcile journal inventory"
    science=read("reports/phase5a5/scientific.json")
    for name,pin in science.items():
        if name in files: assert digest(files[name])==pin
        if Path(name).is_file(): assert sha(name)==pin
        else: assert name in files
    save(R/"scientific.json",science)
    # Keep the previously exercised transfer and exception semantics; bind transport explicitly.
    source=files["src/runner_5a3/colab.py"].decode().split("def interface(")[0]
    source=source.replace("from .provenance import","from src.runner_5a3.provenance import")
    source=source.replace('content=Path("/content")','opaque,generate_command,launch').replace('content/(DS+"_opaque.zip")','opaque').replace('session.monitored(command','launch(command')
    # An unknown mapper/export failure cannot become a timeout fallback.
    path=Path("scripts/phase5a6_transfer.py")
    path.write_bytes(source.encode())
    live=files["scripts/phase5a5_live.py"].decode()
    setup=live[live.index('        base=repo/"runtime"'):live.index('        report["status"]="SMOKE_COMPLETE')]
    setup=textwrap.indent(textwrap.dedent(setup),"    ")
    path=Path("scripts/phase5a6_setup.py")
    path.write_bytes((live[:live.index("def main(")]+"def setup(repo,revision,mounted,cfg,env,run,record,output,plan):\n"+setup+"    return base,freeze\n").encode())
    operational=[*sorted(Path("scripts").glob("phase5a6_*.py")),Path("tests/test_phase5a6_dispatch.py")]
    for p in operational:
        if p.name not in ("phase5a6_build.py","phase5a6_package.py"):
            files[p.as_posix()]=p.read_bytes()
    payload=archive(files)
    release=digest(payload)
    cells=[]
    def cell(code,kind="code"):
        record=dict(cell_type=kind,id="cell-"+str(len(cells)),metadata={},source=code.splitlines(True))
        if kind=="code":record.update(outputs=[],execution_count=None)
        cells.append(record)
    cell("Private Dioscuri/Baalshamin run. Run PREFLIGHT interactively on a fresh T4 session. Run one stage cell at a time and retain every journal and archive before continuing.","markdown")
    cell(f"""import base64, hashlib, io, sys, zipfile
from pathlib import Path
release={release!r}
payload=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest()==release
repo=Path('/tmp/phase5a6_dioscuri')
repo.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall(repo)
sys.path[:0]=[str(repo),str(repo/'tests')]
from scripts.phase5a6_production import preflight
try:
    run=preflight(repo,release)
    print("PREFLIGHT PASS. Select the next authorized stage cell.")
except Exception as error:
    import json
    Path('/kaggle/working/phase5a6_blocker.json').write_text(json.dumps(dict(release=release,reason=str(error),type=type(error).__name__)))
    raise
""")
    for stage in ("capture","original","trial","decision"):cell("run("+repr(stage.upper())+")")
    notebook=Path("notebooks/step05a6_dioscuri_kaggle.ipynb")
    save(notebook,dict(nbformat=4,nbformat_minor=5,metadata= {"kernelspec":{"name":"python3","language":"python","display_name":"Python 3"}},cells=cells))
    save(R/"release.json",dict(production_release=release,notebook=sha(notebook),adapter=ADAPTER,prior_runner=OLD,
         prior_journal=JOURNAL,operational_files={p.as_posix():sha(p) for p in operational},pins=cfg["pins"],mounts=cfg["mounts"]))
    return payload
if __name__=="__main__":build()
