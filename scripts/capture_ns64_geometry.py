"""Build the self-contained Haiper Colab notebook from the hash-pinned Phase 3C.7 worker."""

import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/phase4c"
WORKER_PIN = "e3487c87a4dc9649736156632cb6ad440169e3b242b0ffe42e019de2ba940113"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("pinned worker anchor mismatch: " + old[:70])
    return text.replace(old, new)


def build():
    nb = json.loads((ROOT / "notebooks/step03c7_medium_colab.ipynb").read_bytes())
    last = "".join(nb["cells"][-1]["source"])
    original = ast.literal_eval(ast.parse(last).body[0].value)
    if sha(original.encode()) != WORKER_PIN:
        raise ValueError("reference worker changed")
    driver = last[last.index("from pathlib import Path\nfrom google.colab") :]
    package = io.BytesIO()
    paths = sorted(
        (ROOT / "src/novel_model/geometry_capture_v1").glob("*.py")
    ) + sorted((ROOT / "src/novel_model/geometry_graph_v1").glob("*.py"))
    paths += [
        ROOT / "configs/novel_model" / name
        for name in ("architecture_proposal_v1.yaml", "geometry_graph_v1.yaml")
    ]
    with zipfile.ZipFile(package, "w") as z:
        for p in paths:
            zi = zipfile.ZipInfo(p.relative_to(ROOT).as_posix())
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, p.read_bytes())
    raw = package.getvalue()
    adapter = sha(raw)
    worker = replace_once(
        original,
        "    with contextlib.ExitStack() as stack:",
        """    sys.path.insert(0,'/content/ns64_phase4c_adapter')
    from src.novel_model.geometry_capture_v1.hook import install
    with contextlib.ExitStack() as stack:
        install(stack,impl,pipeline,work,source,config)""",
    )
    worker = replace_once(
        worker,
        "require(seed==20261111,",
        """require(not any(Path('/content').glob('*haiper_worker*.zip')) and not Path('/content/ns64_imc2023_haiper_run').exists(),'Prior Haiper outputs visible')
require(seed==20261111,""",
    )
    # The parent worker has already committed any partial capture before the outer driver returns.
    driver = driver.replace("ns64_imc2023_haiper_run", "phase4c_haiper_run").replace(
        "imc_ns64_imc2023_haiper_worker_", "phase4c_haiper_receipts_"
    )
    driver = replace_once(
        driver,
        "budget=min(5400,",
        "budget=min(int(5400-(time.monotonic()-PHASE_START)),",
    )
    driver = replace_once(
        driver,
        "env=dict(os.environ,STAIRS_DEADLINE=",
        f"env=dict(os.environ,CAPTURE_BUNDLE_SHA256='{adapter}',STAIRS_DEADLINE=",
    )
    driver = replace_once(
        driver,
        "    script.write_text(WORKER)",
        "    script.write_text(WORKER)\n    assert hashlib.sha256(script.read_bytes()).hexdigest()=="
        + repr(sha(worker.encode())),
    )
    driver = replace_once(
        driver,
        "        print(result,flush=True)",
        """            cap=work/'capture_v1'
            for p in sorted(cap.glob('*')) if cap.exists() else []:
                if p.is_file() and p.suffix in ('.csv','.json'):
                    z.write(p,'capture/'+p.name)
        print(result,flush=True)
        print('Save the receipt ZIP and the numerical parent .db from the Files pane if a download is interrupted.',flush=True)
        cap=work/'capture_v1'
        if (cap/'manifest.json').is_file():
            db=cap/(json.loads((cap/'manifest.json').read_text())['backup_sha256']+'.db')
            if db.is_file(): files.download(str(db))""",
    )
    pre = """import base64,io
from pathlib import Path
adapt=Path('/content/ns64_phase4c_adapter')
if adapt.exists(): raise RuntimeError('An adapter has already been staged in this session.')
raw=base64.b64decode(ADAPTER_DATA)
assert hashlib.sha256(raw).hexdigest()==ADAPTER_PIN
adapt.mkdir()
with zipfile.ZipFile(io.BytesIO(raw)) as z: z.extractall(adapt)
"""
    nb["cells"][0]["source"] = (
        "import time\nPHASE_START=time.monotonic()\n"
        + "".join(nb["cells"][0]["source"])
    ).splitlines(True)
    nb["cells"][-1]["source"] = (
        "WORKER="
        + repr(worker)
        + "\nADAPTER_DATA="
        + repr(base64.b64encode(raw).decode())
        + "\nADAPTER_PIN="
        + repr(adapter)
        + "\n"
        + pre
        + driver
    ).splitlines(True)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            compile("".join(cell["source"]), "<colab>", "exec")
    compile(worker, "<worker>", "exec")
    dst = ROOT / "notebooks/step04c_haiper_capture_colab.ipynb"
    dst.write_bytes(json.dumps(nb, ensure_ascii=True).encode())
    REPORT.mkdir(exist_ok=True)
    pins = dict(
        notebook_sha256=sha(dst.read_bytes()),
        historical_worker_sha256=WORKER_PIN,
        capture_worker_sha256=sha(worker.encode()),
        adapter_bundle_sha256=adapter,
        adapter_files={
            p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in paths
        },
    )
    (REPORT / "workflow_pins.json").write_text(
        json.dumps(pins, indent=2), encoding="utf8"
    )
    return dst


if __name__ == "__main__":
    print(build())
