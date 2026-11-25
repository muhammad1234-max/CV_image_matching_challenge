"""Validation-only entry point for the private Kaggle VM."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

from src.runner_5a5.kaggle import (GIB,JOURNAL,disk_plan,gpu_env,guard,inputs,monitored,sample,save,sha,transition,verify_opaque)
from scripts.phase5a5_persistence import create,restore
from src.runner_5a3.provenance import Contract
from src.runner_5a1.runner import Runner

CUDA_SMOKE="""import gc,json,sys,torch
assert sys.version_info[:3]==(3,11,11)
assert torch.__version__=='2.5.1+cu121' and torch.version.cuda=='12.1'
assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='Tesla T4'
torch.cuda.set_device(0)
torch.cuda.reset_peak_memory_stats()
v=torch.arange(8,device='cuda:0')
assert (v*v).sum().item()==140
torch.cuda.synchronize()
peak=torch.cuda.max_memory_allocated()
del v
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
assert peak>0 and torch.cuda.memory_allocated()==0
print(json.dumps(dict(python=sys.executable,version=sys.version,torch=torch.__version__,device=torch.cuda.get_device_name(0),peak_bytes=peak,remaining_bytes=torch.cuda.memory_allocated())))
"""

def fetch(url,path,expected):
    path.parent.mkdir(exist_ok=True,parents=True)
    if not path.exists():
        part=path.with_suffix(".part")
        with urllib.request.urlopen(url,timeout=90) as response,part.open("wb") as out:
            shutil.copyfileobj(response,out)
        if sha(part)!=expected:raise ValueError("Downloaded weight digest: "+path.name)
        part.rename(path)
    if sha(path)!=expected:raise ValueError("Weight digest: "+path.name)

def main(repo,revision,restore_pin=None):
    cfg=json.loads((repo/"manifest.json").read_bytes())
    output=Path("/kaggle/working")
    receipt=output/"phase5a5_kaggle_smoke.json"
    report=dict(status="STARTED",mode="restore" if restore_pin else "smoke",adapter=revision,full_experiment_started=False,checks={})
    policy=json.loads((repo/"configs/novel_model/phase4g1_resource_policy.yaml").read_bytes())
    def record(key,value):
        report["checks"][key]=value
        save(receipt,report)
    try:
        expected={"phase5a5_synthetic_checkpoint.bin":restore_pin} if restore_pin else cfg["mounts"]
        root=Path("/kaggle/input")
        mountpoints=[line.split()[4] for line in Path("/proc/self/mountinfo").read_text().splitlines()]
        record("mounts",dict(input_exists=root.exists(),input_is_dir=root.is_dir(),
               kaggle_mounts=[m for m in mountpoints if m.startswith("/kaggle")],
               working_entries=len(list(output.iterdir()))))
        if any(p.name not in (".virtual_documents","__notebook__.ipynb",receipt.name) for p in output.iterdir()):
            raise ValueError("Use a fresh Kaggle session with clean working outputs")
        for path in ("/content/drive","/content/private.json","/kaggle/working/train_labels.csv","/root/.kaggle/kaggle.json"):
            if os.path.lexists(path):raise ValueError("Administrative path present; stop before staging")
        mounted=inputs(root,expected)
        record("approved_mounts",{str(p.relative_to(root)):sha(p) for p in mounted.values()})
        if restore_pin:
            record("fresh_session_restore",restore(mounted["phase5a5_synthetic_checkpoint.bin"],restore_pin,revision))
            report["status"]="SYNTHETIC_CUSTODY_RESTORED"
            return
        verify_opaque(mounted["dioscuri_opaque.bin"],cfg["pins"]["image_permutation"])
        record("opaque",dict(images=138,permutation=cfg["pins"]["image_permutation"]))
        devices=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.total","--format=csv,noheader,nounits"],text=True)
        selection=gpu_env(devices)
        env={**os.environ,**selection}
        record("hardware",dict(inventory=devices.splitlines(),selected=selection))
        baseline=sample(os.getpid())
        guard(baseline,baseline,policy)
        record("idle",baseline)
        free=shutil.disk_usage(repo).free
        plan=disk_plan(cfg["estimated_parent"])
        record("disk",dict(free=free,planning_bytes=plan,prospective_sufficient_at_sample=free>=sum(plan.values())))
        # Smoke budgeting is independent of the prospective journal.
        end=time.monotonic()+14400-1200
        def run(label,command,cwd=repo,up=None):
            record(label,monitored(list(map(str,command)),cwd,up or env,policy,end,GIB))
        contract=Contract(cfg["pins"],revision)
        dest=repo/"custody"
        dest.mkdir()
        pin=transition(repo/("journal-"+JOURNAL+".json"),dest/"attempts.json",contract.identity,revision)
        Runner(dest,contract,lambda *args:None,pin)
        record("custody_preview",dict(original=JOURNAL,preview=pin,events=json.loads((dest/"attempts.json").read_bytes()),committed=False))
        with tempfile.TemporaryDirectory(dir=repo) as t:
            bundle=output/"phase5a5_synthetic_checkpoint.bin"
            retained=create(Path(t),bundle,revision)
            record("synthetic_checkpoint",dict(external_sha256=retained,local_restore=restore(bundle,retained,revision)))
        run("generated_children",[sys.executable,"-m","pytest","tests/test_phase5a3.py","tests/test_phase5a5_children.py","-q","--basetemp",repo/"fixture_tmp"])
        base=repo/"runtime"
        base.mkdir()
        with zipfile.ZipFile(mounted["ns64_runtime.bin"]) as z:z.extractall(base)
        requirements=(base/"requirements.txt").read_text().replace("/content/ns64_clean/wheels/",str(base/"wheels")+"/")
        lock=base/"kaggle_requirements.txt"
        lock.write_text(requirements)
        env.update(UV_PYTHON_INSTALL_DIR=str(base/"python"),UV_CACHE_DIR=str(base/"uv_cache"),PIP_NO_CACHE_DIR="1")
        tool=base/"bootstrap"
        run("bootstrap",[sys.executable,"-m","pip","install","--no-deps","--target",tool,"uv==0.13.8"])
        boot={**env,"PYTHONPATH":str(tool)}
        run("install_python",[sys.executable,"-m","uv","python","install","3.11.11"],up=boot)
        run("create_env",[sys.executable,"-m","uv","venv","--python","3.11.11","--seed",base/"venv"],up=boot)
        python=base/"venv/bin/python"
        run("install_build_tools",[python,"-m","pip","install","setuptools==75.8.0","wheel==0.45.1"])
        run("install_lock",[python,"-m","pip","install","--no-deps","--extra-index-url","https://download.pytorch.org/whl/cu121","-r",lock])
        freeze=subprocess.check_output([python,"-m","pip","freeze"],env=env,text=True)
        def normalized(line):
            if " @ file:" in line:
                return line.split(" @ ",1)[0].lower()+" @ "+line.rsplit("/",1)[-1]
            return line.lower()
        actual=sorted(normalized(line) for line in freeze.splitlines())
        if actual!=cfg["normalized_freeze"]:
            record("package_differences",dict(missing=sorted(set(cfg["normalized_freeze"])-set(actual)),extra=sorted(set(actual)-set(cfg["normalized_freeze"]))))
            raise ValueError("Installed distribution versions or local-wheel digests differ")
        record("freeze",actual)
        # Generate each target program and inspect its entrypoint. Execution remains synthetic.
        import hashlib
        from src.runner_5a5.commands import prepare
        from src.novel_model.final_candidate_v1 import session
        live_contract=Contract(dict(cfg["pins"],environment=hashlib.sha256(freeze.encode()).hexdigest()),revision)
        inventory=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],text=True).strip()
        programs={}
        for stage in ("capture","original","trial","decision"):
            work=repo/("program-"+stage)
            work.mkdir()
            argv=prepare(session,stage,work,work,repo,live_contract,base,mounted["dioscuri_opaque.bin"],inventory,cfg["mounts"])
            programs[stage]=dict(interpreter=argv[0],program_sha256=sha(argv[3]))
        record("generated_kaggle_programs",programs)
        record("cuda",json.loads(subprocess.check_output([python,"-I","-c",CUDA_SMOKE],env=env,text=True,timeout=90)))
        source=base/"source"
        inventory={p.relative_to(source).as_posix():sha(p) for p in source.rglob("*") if p.is_file() and p.suffix!=".pyc"}
        import hashlib
        if hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest()!=cfg["pins"]["source_inventory"]:
            raise ValueError("Published source inventory mismatch")
        assets=json.loads((base/"assets.json").read_bytes())
        weights={}
        env.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",MPLBACKEND="Agg",PYTHONPATH=str(source))
        for key,code in cfg["native_probes"].items():
            run("native_"+key,[python,"-c",code],source)
        minimum=sum(a["size"] for a in cfg["asset_sizes"].values())+GIB
        if shutil.disk_usage(base).free<minimum:raise RuntimeError("Insufficient space for pinned weights and reserve")
        for key,a in assets.items():
            path=(base/"weights"/("dinov2" if key=="dinov2" else "")/a["filename"])
            url=a.get("url","")
            if key=="dinov2":
                prefix="https://huggingface.co/facebook/dinov2-base/resolve/main/"
                url=prefix+a["filename"]
                for n,pin in a["sidecars"].items():fetch(prefix+n,path.parent/n,pin)
            fetch(url,path,a["sha256"])
            weights[key]=path
        record("assets",{key:sha(p) for key,p in weights.items()})
        for key,code in cfg["loaders"].items():
            if key=="codebook":continue
            args=[weights[k] for k in ("mast3r","retrieval","codebook")] if key=="retrieval" else [weights[key]]
            run("load_"+key,[python,"-c","import torch;assert torch.cuda.device_count()==1\n"+code+"\ntorch.cuda.synchronize()",*args],source)
        remaining_free=shutil.disk_usage(base).free
        export_free=shutil.disk_usage(output).free
        record("disk_after_setup",dict(scratch_free=remaining_free,export_free=export_free,required=sum(plan.values())))
        if remaining_free<sum(plan.values()) or export_free<plan["archive"]+plan["export_margin"]:
            raise RuntimeError("Storage below prospective planning allowance")
        report["status"]="SMOKE_COMPLETE_AWAIT_EXTERNAL_RESTORE"
    except Exception as exc:
        report.update(status="BLOCKED",reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(receipt,report)
        print(report["status"],receipt)
