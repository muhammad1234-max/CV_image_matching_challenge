"""Colab notebook interface. Stage archives are self-contained relative to earlier sealed stages."""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile

from . import POLICY, file_sha
from .workflow import STAGES, read, save, seal, select, verify
from ..resource_monitor_v2 import guard, sample

KIT = Path(__file__).resolve().parents[3]
PYTHON = "/content/ns64_clean/venv/bin/python"


def unpack(path, dest, expected):
    if file_sha(path) != expected:
        raise ValueError("checkpoint archive differs from independently recorded SHA-256")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or any(Path(n).is_absolute() or ".." in Path(n).parts or
            "\\" in n or ":" in n for n in names) or sum(e.file_size for e in archive.infolist()) > 16*1024**3):
            raise ValueError("invalid checkpoint inventory")
        archive.extractall(dest)


def monitored(command, cwd, dest, end, stage):
    policy = read(KIT/"configs/novel_model/phase4g1_resource_policy.yaml")
    initial = sample(os.getpid())
    guard(initial,initial,policy)
    if time.monotonic() >= end:
        raise TimeoutError("stage budget exhausted")
    with (dest/"worker.log").open("w") as log, (dest/"heartbeat.jsonl").open("w") as heart:
        proc = subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,STAIRS_DEADLINE=str(end+POLICY["export_reserve"]),PHASE4I_DEST=str(dest)))
        try:
            while proc.poll() is None:
                usage = sample(proc.pid)
                print(json.dumps(dict(stage=stage,remaining=end-time.monotonic(),**usage)),file=heart,flush=True)
                guard(usage,initial,policy)
                if time.monotonic() >= end:
                    raise TimeoutError(stage)
                time.sleep(min(policy["sample_seconds"],max(0,end-time.monotonic())))
            if proc.returncode:
                raise RuntimeError(f"{stage} worker exit {proc.returncode}")
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()


def stage_command(stage, work, ds, dest, root):
    worker = (KIT/"workers"/(ds+".py")).read_text()
    insert = "sys.path.insert(0,"+repr(str(KIT))+")\n"
    if stage == "decision":
        # No assets, images, matching or PyCOLMAP are needed to replay sealed indexed evidence.
        argv=repr(["workflow",stage,ds,str(root),str(dest),str(work/"test"/ds)])
        code=("import sys,subprocess,hashlib\n"
              "assert sys.version_info[:3]==(3,11,11)\n"
              "freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'])\n"
              "assert hashlib.sha256(freeze).hexdigest()=='8bb71679bf5735f6e3c8225885be88c1f530639094df310a5a29e6691498455b'\n"
              +insert+"from src.novel_model.final_candidate_v1.workflow import main\nsys.argv="+argv+"\nmain()\n")
    elif stage == "capture":
        marker = "            result=fn(*args,**kwargs)"
        if worker.count(marker) != 1:
            raise ValueError("incompatible baseline hook")
        hook = ("if name=='incremental_mapping':\n    "+insert.strip()+"\n    "
                "from src.novel_model.final_candidate_v1.workflow import capture\n    "
                "capture(kwargs,Path(os.environ['PHASE4I_DEST']))\n    row['status']='DEFERRED_TO_ORIGINAL'\n    raise SystemExit(0)")
        import textwrap
        code = worker.replace(marker,textwrap.indent(hook," "*12)+"\n"+marker)
    else:
        # Repeat the unchanged worker's VM, source, weights, freeze and opaque-order checks on resume.
        prefix, suffix = worker.split("from config import PipelineConfig\n",1)
        code = prefix+insert+"from src.novel_model.final_candidate_v1.workflow import main\n"
        code += "sys.argv="+repr(["workflow",stage,ds,str(root),str(dest),str(work/"test"/ds)])+"\nmain()\n"
    script = work/"run.py"
    script.write_text(code)
    return [PYTHON,"-I","-u",str(script),str(POLICY["seed"])]


def identity(ds):
    return dict(model=POLICY["version"], seed=POLICY["seed"],dataset=ds,
                protocol=file_sha(KIT/"inference_pins.json"),worker=file_sha(KIT/"workers"/(ds+".py")))


def stage_run(ds, stage, root, deadline, restore=None):
    from google.colab import files
    if ds not in POLICY["development"] or stage not in STAGES:
        raise ValueError("dataset/stage outside freeze")
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    ident = identity(ds)
    parents = {}
    for earlier in STAGES[:STAGES.index(stage)]:
        parents[earlier] = verify(root/earlier,earlier,ident,dict(parents))
    spent=sum(read(root/k/"time.json")["seconds"] for k in parents)
    # Keep a new VM from resetting the per-dataset execution budget.
    deadline=min(deadline,time.monotonic()+POLICY["session_seconds"]-spent)
    final = root/stage
    archive = root/f"{ds}_{stage}.zip"
    if not final.exists():
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            dest = Path(tmp)/"output"
            dest.mkdir()
            if restore:
                unpack(Path(restore[0]),dest,restore[1])
                verify(dest,stage,ident,parents)
            else:
                start = time.monotonic()
                remaining = deadline-start
                if remaining <= POLICY["export_reserve"] or (stage=="capture" and remaining<POLICY["min_capture_start"]):
                    raise RuntimeError("session time insufficient; preserve verified checkpoints")
                end = deadline-POLICY["export_reserve"]
                if stage=="trial":
                    end=min(end,start+POLICY["trial_seconds"])
                work = Path(tmp)/"work"
                work.mkdir()
                try:
                    monitored(stage_command(stage,work,ds,dest,root),work,dest,end,stage)
                    if stage=="capture":
                        for name in ("identity.json","prediction.json","opaque.csv","input_receipt.json","resources.json",
                                     "boundary.json","checkpoint_verify_matches.json","stages.jsonl","environment.json"):
                            if (work/name).is_file():
                                shutil.copyfile(work/name,dest/name)
                        if not (dest/"job.json").exists():
                            if not (dest/"opaque.csv").is_file():
                                raise RuntimeError("no parent or published output")
                            save(dest/"no_mapper.json",dict(reason="published pipeline returned before mapping"))
                        else:
                            job=read(dest/"job.json")
                            pin=read(KIT/"worker_pins.json")[ds]
                            hashes=job["image_hashes"]
                            ordered="\n".join(hashes[n] for n in sorted(hashes))
                            if hashlib.sha256(ordered.encode()).hexdigest()!=pin["permutation_sha256"]:
                                raise ValueError("mapper input order/bytes differ")
                except Exception as error:
                    if stage != "trial":
                        receipt=root/f"{stage}_failed_{time.time_ns()}.zip"
                        with zipfile.ZipFile(receipt,"x",zipfile.ZIP_DEFLATED) as z:
                            z.writestr("failure.json",json.dumps(dict(stage=stage,identity=ident,error=str(error))))
                            for p in [dest/"heartbeat.jsonl",dest/"worker.log",work/"current_stage.json"]:
                                if p.is_file(): z.write(p,p.name)
                        print("Failure archive",receipt.name,file_sha(receipt),flush=True)
                        files.download(str(receipt))
                        raise
                    save(dest/"branch.json",dict(available=False,reason=str(error),failure_type=type(error).__name__))
                save(dest/"time.json",dict(seconds=time.monotonic()-start))
                seal(dest,stage,ident,parents)
            dest.rename(final)
    verify(final,stage,ident,parents)
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        with zipfile.ZipFile(partial,"w",zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for p in final.rglob("*"):
                if p.is_file(): z.write(p,p.relative_to(final))
        partial.replace(archive)
    print(stage, file_sha(archive), "save this digest with the download",flush=True)
    if stage=="decision": print(read(final/"selection.json")["action"],flush=True)
    files.download(str(archive))
