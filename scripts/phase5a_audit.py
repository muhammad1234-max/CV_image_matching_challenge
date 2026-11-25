"""Administrative CPU preflight. Never run a mapper or a scorer in this audit."""
import ast
import base64
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import shutil
import sys
import subprocess
import tempfile
import time
from types import ModuleType
from unittest.mock import patch
import zipfile

import psutil
from scripts.build_ns64_76 import inventory, ordered
from src.novel_model.final_candidate_v1 import session
from src.novel_model.final_candidate_v1.workflow import file_sha, read, save, seal, select, verify

DS = "imc2024_dioscuri_baalshamin"
OUT = Path("reports/phase5a")
CFG = "8134dd74471fe6bd25bf8ce3a891454f461e772d156715409f62977eb4c95b26"
PROTO = "ec1af922c449ed0f68cc321cf972af772e491154ddaa36700a75632a24c4a690"
REVIEW = "caca1334b3e20bb5f3749263ec4192c15026d941139ccfd8b66d4b721d94e1de"

def sha(data):
    return hashlib.sha256(data).hexdigest()

def integrity():
    freeze=read(Path("reports/phase4i/freeze.json"))
    assert freeze["config"]==CFG==file_sha("configs/novel_model/final_candidate_v1.yaml")
    assert freeze["protocol"]==PROTO==file_sha("reports/phase4i/inference_pins.json")
    assert all(file_sha(k)==v for k,v in freeze["files"].items())
    archive=Path("phase4i_final_candidate_review.zip")
    assert file_sha(archive)==REVIEW
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read("SHA256.json"))
        assert set(z.namelist())==set(manifest)|{"SHA256.json"}
        assert all(sha(z.read(n))==h==file_sha(n) for n,h in manifest.items())
    nb=read(Path("notebooks/step04i_final_candidate_colab.ipynb"))
    cells=["".join(c["source"]) for c in nb["cells"]]
    for c in cells: ast.parse(c)
    kit=base64.b64decode(ast.literal_eval(ast.parse(cells[3]).body[0].value))
    assert sha(kit)==freeze["inference_kit"]
    with zipfile.ZipFile(io.BytesIO(kit)) as z:
        pins=read(Path("reports/phase4i/inference_pins.json"))
        assert set(z.namelist())==set(pins)|{"inference_pins.json"}
        assert sha(z.read("inference_pins.json"))==PROTO
        assert all(sha(z.read(n))==h for n,h in pins.items())
        worker=z.read("workers/"+DS+".py").decode()
    spec=freeze["datasets"][DS]
    assert sha(worker.encode())==spec["worker_sha256"]
    pre=read(Path("reports/phase3c9/preparation.json"))
    ns=read(Path("configs/baselines/ns64_isolated_v1/experiment.json"))
    assert all(file_sha(k)==h for k,h in ns["source_files"].items())
    assert pre["benchmark"]==ns["benchmark"]
    original=read(Path("configs/baselines/top10_reproduction_v1/source_manifest.json"))["ns64"]
    asset_manifest=read(Path("configs/baselines/top10_reproduction_v1/assets_phase3c1.json"))
    assert pre["source_revision"]==original["revision"]==asset_manifest["revision"]
    assert pre["source_revision"] in worker
    assert all(file_sha(Path("snapshots/top10_ns")/k)==h for k,h in original["files"].items())
    assert file_sha("configs/baselines/top10_reproduction_v1/ns64_pilot_source.zip.bin")==ns["source_sha256"]==asset_manifest["archive_sha256"]
    assert file_sha(Path("snapshots/top10_ns")/asset_manifest["config"])==pre["config_sha256"]==ns["configuration_sha256"]
    assert file_sha(spec["organizer_prediction"])==spec["organizer_sha256"]
    probe=next(ast.literal_eval(x.value.args[0]) for x in ast.parse(worker).body if isinstance(x,ast.Expr)
               and isinstance(x.value,ast.Call) and getattr(x.value.func,"id",None)=="exec")
    defs={x.targets[0].id:ast.literal_eval(x.value) for x in ast.parse(probe).body if isinstance(x,ast.Assign)
          and getattr(x.targets[0],"id",None) in ("SOURCE_HASHES","ASSETS")}
    runtime=Path("imc_ns64_clean_runtime.zip")
    assert file_sha(runtime) in cells[0]
    with zipfile.ZipFile(runtime) as z:
        clean={n[7:]:sha(z.read(n)) for n in z.namelist() if n.startswith("source/") and not n.endswith("/")}
        assert clean==defs["SOURCE_HASHES"]
        assert sha(json.dumps(clean,sort_keys=True).encode())==pre["source_inventory_hash"]
        aliases=json.loads(z.read("assets.json"))
        assert sha(z.read("assets.json")) in worker
    dep=Path("reports/phase3c1_dependency/imc_ns64_dependency_receipt.zip")
    assert file_sha(dep)==read(dep.parent/"review.json")["archive_sha256"]
    with zipfile.ZipFile(dep) as z:
        prior=json.loads(z.read("result.json"))["assets"]
    required={}
    for k,record in defs["ASSETS"].items():
        assert record["sha256"]==ns["assets"][k]==pre["checkpoint_hashes"][k]==aliases[k]["sha256"]
        assert aliases[k].get("sidecars",{})==record.get("sidecars",{})
        expected={record["filename"]:record["sha256"],**record.get("sidecars",{})}
        observed={Path(f["path"]).name:f["sha256"] for f in prior[k]["files"]}
        assert observed[record["filename"]]==record["sha256"]
        # This historical receipt lists the weight itself. Sidecar pins are in the clean runtime manifest.
        assert record.get("sidecars",{})==asset_manifest["assets"][k].get("sidecars",{})
        assert all(expected[n]==h for n,h in observed.items())
        required[k]=expected
    imgs=inventory(DS)
    names=ordered(imgs,20261111)
    group=next(x for x in read(Path("configs/benchmark/imc2025_local_v1/source_groups.json"))["mapping"] if x["dataset"]==DS)
    assert group["partition"]==spec["partition"]=="dev_a" and group["source_group"]==spec["group"]=="heritage_dioscuri"
    assert len(names)==len(set(names))==spec["images"]==138
    inputs=Path(DS+"_opaque.zip")
    with zipfile.ZipFile(inputs) as z:
        expected=[f"image_{i:04d}.png" for i in range(len(names))]
        assert sorted(z.namelist())==expected
        sequence=[sha(z.read(n)) for n in expected]
    assert sequence==[imgs[n] for n in names]==pre["ordered_content_hashes"]
    assert sha("\n".join(sequence).encode())==pre["permutation_hash"]==spec["permutation_sha256"]
    assert file_sha(inputs)==read(Path("reports/phase4i/inventory.json"))["independent_inputs"][DS]["sha256"]
    assert not any(n in worker for n in names)
    return dict(dataset=DS,seed=20261111,images=len(names),partition=spec["partition"],group=spec["group"],
        review_sha256=REVIEW,validated_review_files=len(manifest),config_sha256=CFG,protocol_sha256=PROTO,
        freeze_sha256=file_sha("reports/phase4i/freeze.json"),frozen_file_pins=freeze["files"],kit_sha256=sha(kit),
        source_revision=original["revision"],published_file_count=len(original["files"]),runtime_source_count=len(clean),
        published_config=pre["config_sha256"],benchmark=ns["benchmark"],assets=required,
        historical_dependency_archive=file_sha(dep),live_assets_verified=False,worker=spec["worker_sha256"],
        runtime_bundle=file_sha(runtime),image_archive=file_sha(inputs),image_archive_bytes=inputs.stat().st_size,
        image_permutation=spec["permutation_sha256"],organizer_prediction_sha256=spec["organizer_sha256"])

def runner_check():
    # Exercise the real stage wrapper and seal logic with a child that finishes without writing outputs.
    ident=dict(dataset=DS,fixture="CPU operational check")
    colab=ModuleType("google.colab")
    colab.files=type("Downloads",(),{"download":staticmethod(lambda path:None)})
    with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,{"google.colab":colab}), \
         patch.object(session,"identity",return_value=ident), patch.object(session,"stage_command",return_value=[]):
        root=Path(tmp)
        cap=root/"capture"
        cap.mkdir()
        save(cap/"time.json",dict(seconds=1))
        seal(cap,"capture",ident,{})
        parents={"capture":file_sha(cap/"seal.json")}
        def empty_child(*args):
            subprocess.run([sys.executable,"-c","raise SystemExit(0)"],check=True,cwd=args[1])
        with patch.object(session,"monitored",side_effect=empty_child), redirect_stdout(io.StringIO()):
            session.stage_run(DS,"original",root,time.monotonic()+9000)
        original=root/"original"
        verified=verify(original,"original",ident,parents)
        original_files=sorted(read(original/"seal.json")["files"])
        assert original_files==["time.json"] and (root/(DS+"_original.zip")).exists()
        # Independently inspect handling of a failed integrity precheck on a trial.
        save(original/"branch.json",dict(available=True,valid_export=True,proposal=None,groups=[]))
        (original/"prediction.csv").write_bytes(b"synthetic original bytes")
        seal(original,"original",ident,parents)
        with patch.object(session,"monitored",side_effect=RuntimeError("fixture: trial input digest mismatch")), redirect_stdout(io.StringIO()):
            session.stage_run(DS,"trial",root,time.monotonic()+9000)
        dest=root/"selection"
        dest.mkdir()
        action=select(root,dest)
        assert action["action"]=="ABSTAIN" and file_sha(dest/"final.csv")==file_sha(original/"prediction.csv")
    return dict(scope="synthetic CPU diagnostics only; no dataset stage executed",empty_success_original={
        "sealed":True,"verified_seal":verified,"payload_files":original_files,"exported":True,
        "prediction_present":False,"branch_receipt_present":False,"observations_present":False},
        injected_trial_integrity_failure=dict(action=action["action"],selected=action["selected"],reason=action["reason"],
                                              original_bytes_preserved=True))

def main():
    r=integrity()
    save(OUT/"integrity.json",r)
    save(OUT/"runner_probe.json",runner_check())
    ram=psutil.virtual_memory()
    disk=shutil.disk_usage(Path.cwd())
    save(OUT/"resources.json",dict(observed_utc=datetime.now(timezone.utc).isoformat(),scope="local workstation",
        os=platform.system(),python=platform.python_version(),nvidia_smi=shutil.which("nvidia-smi"),
        host_ram_total_bytes=ram.total,host_ram_available_bytes=ram.available,disk_total_bytes=disk.total,
        disk_free_bytes=disk.free,linux_proc_meminfo=Path("/proc/meminfo").is_file(),
        clean_gpu_runtime_measured=False,remote_vram_bytes=None,remote_ram_bytes=None,cgroup_current_bytes=None,
        cgroup_max_bytes=None,remote_disk_bytes=None,remote_session_seconds=None,remote_pressure=None,remote_oom_events=None,
        remote_python_cuda=None,checkpoint_assets_on_remote=None,experiment_seconds=None,experiment_cpu_gpu_peak=None,
        scratch_floor_bytes=max(2**30,4*sum(x.file_size for x in zip_infos())),
        scratch_scope="published initial planning floor; peak numerical-database, branch, and export bytes unknown"))
def zip_infos():
    with zipfile.ZipFile(DS+"_opaque.zip") as z:return z.infolist()
if __name__=="__main__": main()