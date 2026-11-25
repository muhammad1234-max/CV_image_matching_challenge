"""Linux operational sampler. All memory counters except GPU MiB are bytes."""
import json
import os
import subprocess
from pathlib import Path

def kb(text):
    return {k:int(v.split()[0])*1024 for line in text.splitlines() if ":" in line
            for k,v in [line.split(":",1)] if v.strip().endswith(" kB")}

def process_row(pid,proc=Path("/proc")):
    try:
        fields=kb((proc/str(pid)/"smaps_rollup").read_text())
        return dict(pid=pid,rss=fields["Rss"],pss=fields["Pss"],
                    uss=fields["Private_Clean"]+fields["Private_Dirty"]+fields.get("Private_Hugetlb",0))
    except FileNotFoundError:
        return dict(pid=pid,exited=True)
    except (PermissionError,KeyError,ValueError):
        return dict(pid=pid,unavailable=True)

def cgroups(proc=Path("/proc")):
    member=next((l[3:] for l in (proc/"self/cgroup").read_text().splitlines() if l.startswith("0::")),None)
    if member is None:raise ValueError("unavailable cgroup v2 memory controller")
    for line in (proc/"self/mountinfo").read_text().splitlines():
        if " - cgroup2 " not in line:continue
        _,_,_,root,mount,*_=line.split()
        mount=mount.replace(r"\040"," ")
        relative=os.path.relpath(member,root)
        if relative.startswith(".."):
            # A private cgroup namespace can expose its own root as /.
            visible=Path(mount)/member.lstrip("/")
            try:
                if str(os.getpid()) not in (visible/"cgroup.procs").read_text().split():continue
            except OSError:
                continue
            relative=member.lstrip("/")
        mount=Path(mount)
        leaf=mount/relative
        if not leaf.is_dir():continue
        rows=[]
        for p in [leaf,*leaf.parents]:
            if (p/"memory.current").is_file():
                limit=(p/"memory.max").read_text().strip()
                rows.append(dict(path=str(p),limit=None if limit=="max" else int(limit),
                                 current=int((p/"memory.current").read_text()),
                                 events={k:int(v) for k,v in (l.split() for l in (p/"memory.events").read_text().splitlines())}))
            if p==mount:break
        if rows:return rows
    raise ValueError("cgroup mount or memory counters unavailable")

def sample(pid):
    import psutil
    info=kb(Path("/proc/meminfo").read_text())
    try:
        tree=[pid]+[p.pid for p in psutil.Process(pid).children(recursive=True)]
    except psutil.NoSuchProcess:
        tree=[]
    rows=[process_row(p) for p in tree]
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,memory.used","--format=csv,noheader,nounits"],text=True,timeout=5).strip().splitlines()
    pressure=Path("/proc/pressure/memory").read_text()
    full=float(next(l for l in pressure.splitlines() if l.startswith("full ")).split()[1].split("=")[1])
    return dict(groups=cgroups(),host={k:info[k] for k in ("MemTotal","MemAvailable","SwapTotal","SwapFree")},
                tree=rows,totals={k:sum(r[k] for r in rows if k in r) for k in ("pss","uss","rss")},
                full_avg10=full,gpu=[(n.strip(),int(t),int(u)) for n,t,u in (l.split(",") for l in gpu)])

def guard(row,initial,policy):
    host=row["host"]
    finite=[g for g in row["groups"] if g["limit"] is not None]
    if not finite:raise RuntimeError("no measured finite cgroup limit")
    for limit,available in [(host["MemTotal"],host["MemAvailable"]),*((g["limit"],g["limit"]-g["current"]) for g in finite)]:
        if available<=max(policy["minimum_margin_bytes"],policy["margin_fraction"]*limit):
            raise RuntimeError("host or cgroup headroom exhausted")
    previous={g["path"]:g["events"] for g in initial["groups"]}
    for g in row["groups"]:
        if any(g["events"][k]>previous[g["path"]][k] for k in ("oom","oom_kill")):
            raise RuntimeError("new cgroup OOM event")
    if row["full_avg10"]>=policy["pressure_full_avg10_percent"]:raise RuntimeError("host memory pressure")
    if not row["gpu"] or any(t<=0 or t-u<=t*policy["gpu_margin_fraction"] for _,t,u in row["gpu"]):
        raise RuntimeError("GPU headroom exhausted")


