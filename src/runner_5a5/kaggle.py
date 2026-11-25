"""Kaggle compatibility adapter. The only workloads dispatched here are release smoke tests."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile
from src.novel_model.resource_monitor_v2 import cgroups, guard, kb, process_row

GIB=1024**3
JOURNAL="c55696d7f66f5555474fffc693566b91973b5e172f33e2a7e7e20f8a6e9a8dc4"
OLD="dbefffec1ede534a31bad1f8cd84da7b61f484e125ef6ef1adeb5d4698ecd886"

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def save(path,value):
    path=Path(path)
    pending=path.with_suffix(".partial")
    pending.write_text(json.dumps(value,indent=2,sort_keys=True))
    pending.replace(path)

def inputs(root,expected):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Kaggle input root absent or redirected")
    found={}
    for directory,dirs,files in os.walk(root,onerror=raise_walk):
        for name in dirs+files:
            path=Path(directory)/name
            if path.is_symlink():
                raise ValueError("Unexpected input link")
        for name in files:
            path=Path(directory)/name
            if name not in expected or name in found or not path.is_file():
                raise ValueError("Unexpected or duplicate mounted file; detach extra inputs")
            if sha(path)!=expected[name]:
                raise ValueError("Input digest mismatch: "+name)
            found[name]=path
    if set(found)!=set(expected):
        raise ValueError("Missing approved input: "+str(sorted(set(expected)-set(found))))
    return found

def raise_walk(exc):
    raise exc

def transition(source,dest,identity,new_revision):
    if sha(source)!=JOURNAL:
        raise ValueError("Unrecognized prior journal; reconcile latest custody externally")
    old=json.loads(source.read_bytes())
    if old!=[dict(status="created",stage="",identity=identity,runner=OLD)]:
        raise ValueError("Prior experiment identity or state differs")
    # The base Runner authenticates the release in its first record. Preserve the
    # predecessor verbatim and append an explicit zero-cost migration event.
    events=[dict(status="created",stage="",identity=identity,runner=new_revision),
            dict(status="migrated",stage="",previous_events=old,previous_sha256=JOURNAL,
                 previous_runner=OLD,new_runner=new_revision,seconds=0,previous_checkpoints=[],
                 reason="Colab capture time gate; Kaggle compatibility validation")]
    if dest.exists():
        if json.loads(dest.read_bytes())!=events:
            raise ValueError("Existing journal requires explicit reconciliation")
    else:
        save(dest,events)
    return sha(dest)

def gpu_env(inventory):
    rows=[s.split(", ") for s in inventory.strip().splitlines()]
    row=next((r for r in rows if r[0]=="0"),None)
    if row is None or len(row)!=5 or row[2]!="Tesla T4" or not row[1].startswith("GPU-"):
        raise ValueError("Physical GPU 0 must be a Tesla T4")
    return dict(CUDA_DEVICE_ORDER="PCI_BUS_ID",CUDA_VISIBLE_DEVICES=row[1],WORLD_SIZE="1",RANK="0",LOCAL_RANK="0")

def disk_plan(parent_bytes):
    # Simultaneous retained copies plus staging allowances, in bytes.
    return dict(sealed_parent=parent_bytes,sqlite_copies=3*parent_bytes,
                two_reconstructions=2*GIB,indexed_observations=GIB,
                archive=parent_bytes+GIB,restore=2*parent_bytes,export_margin=2*GIB)

def kaggle_process_row(pid, proc=Path("/proc")):
    try:
        return process_row(pid, proc)
    except ProcessLookupError:
        # A zombie can remain visible to psutil after its memory map disappears.
        try:
            stat = (proc / str(pid) / "stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            return dict(pid=pid, exited=True)
        fields = stat.rsplit(")", 1)[-1].split()
        # proc stat field 9 contains Linux task flags. PF_EXITING is set
        # before teardown is complete and before state becomes Z.
        if fields[0] in ("Z", "X", "x") or int(fields[6]) & 0x4:
            return dict(pid=pid, exited=True)
        raise RuntimeError(f"Memory map unavailable for PID {pid}, state={fields[0]}, flags={fields[6]}")

def sample(pid):
    import psutil
    info=kb(Path("/proc/meminfo").read_text())
    try:
        tree=[pid]+[p.pid for p in psutil.Process(pid).children(recursive=True)]
    except psutil.NoSuchProcess:
        tree=[]
    rows=[kaggle_process_row(p) for p in tree]
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,memory.used","--format=csv,noheader,nounits"],text=True,timeout=5).strip().splitlines()
    pressure=Path("/proc/pressure/memory").read_text()
    full=float(next(l for l in pressure.splitlines() if l.startswith("full ")).split()[1].split("=")[1])
    return dict(groups=cgroups(),host={k:info[k] for k in ("MemTotal","MemAvailable","SwapTotal","SwapFree")},
                tree=rows,totals={k:sum(r[k] for r in rows if k in r) for k in ("pss","uss","rss")},
                full_avg10=full,gpu=[(n.strip(),int(t),int(u)) for n,t,u in (l.split(",") for l in gpu)])

def monitored(command,cwd,env,policy,deadline,disk_required):
    start=time.monotonic()
    baseline=sample(os.getpid())
    guard(baseline,baseline,policy)
    if shutil.disk_usage(cwd).free<disk_required or start>=deadline:
        raise RuntimeError("Insufficient scratch or time for smoke substage")
    import signal
    with subprocess.Popen(command,cwd=cwd,env=env,start_new_session=True) as child:
        peak=0
        try:
            while child.poll() is None:
                try:
                    row=sample(child.pid)
                except ProcessLookupError as exc:
                    path=Path(exc.filename or "")
                    if path.parent.parent!=Path("/proc") or not path.parent.name.isdecimal() or path.name!="smaps_rollup":
                        raise
                    if child.poll() is not None:
                        break
                    # A descendant exited during enumeration. Require a fresh complete sample.
                    row=sample(child.pid)
                guard(row,baseline,policy)
                if "CUDA_VISIBLE_DEVICES" in env:
                    tree_pids={p["pid"] for p in row["tree"]}
                    processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid","--format=csv,noheader,nounits"],text=True,timeout=5)
                    for entry in processes.splitlines():
                        pid,uuid=entry.split(", ")
                        if int(pid) in tree_pids and uuid!=env["CUDA_VISIBLE_DEVICES"]:
                            raise RuntimeError("Child process allocated on an unselected GPU")
                peak=max(peak,row["totals"]["pss"])
                if time.monotonic()>=deadline or shutil.disk_usage(cwd).free<disk_required:
                    raise RuntimeError("Time or scratch reserve exhausted")
                time.sleep(1)
            if child.returncode:raise RuntimeError(f"Smoke subprocess exit {child.returncode}")
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
    return dict(seconds=time.monotonic()-start,peak_tree_pss_bytes=peak)

def verify_opaque(path,pin):
    with zipfile.ZipFile(path) as z:
        names=sorted(z.namelist())
        if names!=[f"image_{i:04d}.png" for i in range(138)]:
            raise ValueError("Opaque membership")
        sequence="\n".join(hashlib.sha256(z.read(n)).hexdigest() for n in names)
        if hashlib.sha256(sequence.encode()).hexdigest()!=pin:
            raise ValueError("Opaque permutation")
