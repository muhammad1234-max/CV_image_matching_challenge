"""Transport for one supervised Kaggle development run."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from src.runner_5a1.contract import DS, STAGES, check, equal, read
from src.runner_5a1.runner import Runner
from src.runner_5a3.provenance import Contract as ParentContract
from contextlib import closing
from src.novel_model.final_candidate_v1 import immutable_db
from src.runner_5a5 import kaggle
from src.runner_5a5.commands import prepare
from src.novel_model.final_candidate_v1 import session

class Contract(ParentContract):
    def branch(self, stage, d, root):
        super().branch(stage,d,root)
        check(read(d/"branch.json").get("valid_export") is not False,
              "indexed observation export failed")

    def capture(self, d):
        counts=super().capture(d)
        with closing(immutable_db(d/"parent.db")) as db:
            for table in ("matches","two_view_geometries"):
                count=db.execute(f"select count(*) from {table} where rows>0 and length(data)>0").fetchone()[0]
                check(count>0,"missing numerical records in "+table)
            check(db.execute("select count(*) from cameras").fetchone()[0]>0,"missing cameras")
        check(any(counts.values()),"empty keypoints")
        return counts

def check_gpu(env):
    inventory=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,memory.used,memory.free","--format=csv,noheader,nounits"],text=True)
    for line in inventory.splitlines():
        uuid,used,free=line.split(", ")
        if uuid==env["CUDA_VISIBLE_DEVICES"]:
            check(int(free)*1024**2>=9412018176,"selected T4 below frozen capture floor")
        else:
            check(int(used)<=64,"other GPU already allocated")
    return inventory

def space(repo, output, plan):
    scratch = shutil.disk_usage(repo).free
    export = shutil.disk_usage(output).free
    check(scratch >= sum(plan.values()), "scratch reserve exhausted")
    check(export >= plan["archive"] + plan["export_margin"], "export reserve exhausted")
    return dict(scratch=scratch, export=export, required=plan)

def monitor(argv, work, dest, end, stage, env, policy, plan, output):
    check(time.monotonic() < end,"stage deadline expired during custody")
    initial = kaggle.sample(os.getpid())
    kaggle.guard(initial, initial, policy)
    space(work, output, plan)
    child_env = dict(env, PHASE4I_DEST=str(dest), STAIRS_DEADLINE=str(end+1200))
    with (dest/"heartbeat.jsonl").open("w") as heart, (dest/"worker.log").open("w") as log:
        with subprocess.Popen(argv, cwd=work, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True) as child:
            try:
                while True:
                    row = kaggle.sample(child.pid)
                    print(json.dumps(dict(stage=stage, **row)), file=heart, flush=True)
                    kaggle.guard(row, initial, policy)
                    space(work, output, plan)
                    processes = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid","--format=csv,noheader,nounits"],text=True)
                    tree = {p["pid"] for p in row["tree"]}
                    for line in processes.splitlines():
                        pid, uuid = line.split(", ")
                        check(int(pid) not in tree or uuid == env["CUDA_VISIBLE_DEVICES"], "GPU isolation")
                    if child.poll() is not None:
                        break
                    if time.monotonic() >= end:
                        raise TimeoutError(stage)
                    time.sleep(min(policy["sample_seconds"], max(0,end-time.monotonic())))
                if child.returncode:
                    raise RuntimeError(f"{stage} exit {child.returncode}")
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()

def dispatch(runner, stage, dest, end, base, opaque, inventory, mounts, launch):
    from scripts.phase5a6_transfer import execute
    def generate(session, stage, work, dest, root, contract):
        return prepare(session, stage, work, dest, root, contract, base, opaque, inventory, mounts)
    execute(runner, stage, dest, end, opaque, generate, launch)

def retain(path, pin):
    equal(kaggle.sha(path), pin, "exported bytes")
    print(f"Download {path.name} from /kaggle/working. Verify its SHA256 on your computer: {pin}", flush=True)
    equal(input("Paste independently saved file SHA256: ").strip(), pin, "external custody")

def digest_input(prompt):
    pin=input(prompt).strip().lower()
    check(re.fullmatch("[0-9a-f]{64}",pin),"external SHA256 required")
    return pin

def preflight(repo, revision):
    from scripts.phase5a6_setup import setup
    output=Path("/kaggle/working")
    check(all(p.name in (".virtual_documents","__notebook__.ipynb") for p in output.iterdir()),"fresh Kaggle session required")
    check(not any(os.path.lexists(p) for p in ("/content/drive","/content/private.json","/root/.kaggle/kaggle.json","/content/ns64_clean")),"unexpected administrative path")
    cfg=read(repo/"manifest.json")
    remaining=int(input("Actual seconds remaining in this Kaggle session now: "))
    check(0 < remaining <= 43200,"invalid session time")
    deadline=time.monotonic()+remaining
    latest=digest_input("Reconcile external custody. Enter latest actual journal SHA256: ")
    mounts=dict(cfg["mounts"])
    if latest!=kaggle.JOURNAL:
        mounts["journal-"+latest+".json"]=latest
    archives={}
    for stage in STAGES:
        name=DS+"_"+stage+".bin"
        if any(Path("/kaggle/input").rglob(name)):
            mounts[name]=digest_input("Externally retained "+stage+" archive SHA256: ")
            archives[stage]=name
    mounted=kaggle.inputs(Path("/kaggle/input"),mounts)
    check(latest!=kaggle.JOURNAL,"attach the externally committed production journal")
    history=read(mounted["journal-"+latest+".json"])
    prior=read(repo/("journal-"+kaggle.JOURNAL+".json"))
    expected=dict(status="created",stage="",identity=prior[0]["identity"],runner=revision)
    equal(history[0],expected,"release and experiment")
    equal(history[1],dict(status="migrated",stage="",previous_events=prior,
          previous_sha256=kaggle.JOURNAL,previous_runner=kaggle.OLD,new_runner=revision,seconds=0,
          previous_checkpoints=[],reason="Phase 5A.6 approved Kaggle production migration"),"migration event")
    check(history[-1]["status"] in ("migrated","complete"),"interrupted or failed stage requires review")
    equal([e["stage"] for e in history if e["status"]=="complete"],list(archives),"required stage archives")
    kaggle.verify_opaque(mounted["dioscuri_opaque.bin"],cfg["pins"]["image_permutation"])
    plan=kaggle.disk_plan(cfg["estimated_parent"])
    policy=read(repo/"configs/novel_model/phase4g1_resource_policy.yaml")
    env=dict(os.environ,**kaggle.gpu_env(subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.total","--format=csv,noheader,nounits"],text=True)))
    receipt=output/"phase5a6_preflight.json"
    checks={}
    def record(key,value):
        checks[key]=value
        kaggle.save(receipt,dict(release=revision,checks=checks))
    record("mounts",mounts)
    record("selection",{k:env[k] for k in ("CUDA_VISIBLE_DEVICES","WORLD_SIZE")})
    idle=kaggle.sample(os.getpid())
    kaggle.guard(idle,idle,policy)
    record("idle",idle)
    record("gpu_memory",check_gpu(env))
    def run(label,command,cwd=repo,up=None):
        print(label,flush=True)
        record(label,kaggle.monitored(list(map(str,command)),cwd,up or env,policy,deadline-1200,kaggle.GIB))
    base,freeze=setup(repo,revision,mounted,cfg,env,run,record,output,plan)
    pins=dict(cfg["pins"],environment=hashlib.sha256(freeze.encode()).hexdigest())
    record("raw_freeze_sha256",pins["environment"])
    contract=Contract(pins,revision)
    for name,pin in read(repo/"inference_pins.json").items():
        equal(kaggle.sha(repo/name),pin,name)
    equal(session.identity(DS),contract.identity,"packaged worker")
    # Exercise the same dispatch and transfer before committing actual custody.
    fixture_env=dict(env,NS64_FIXTURE_PYTHON=str(base/"venv/bin/python"),NS64_TEST_RELEASE=revision)
    run("production_fixture",[sys.executable,"-m","pytest","tests/test_phase5a6_dispatch.py","-q","--basetemp",repo/"fixture_tmp"],up=fixture_env)
    inventory=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],text=True).strip()
    # Exercise the actual generated capture preamble through its runtime and
    # weight checks. The stop point precedes opening the real image archive.
    import tempfile
    with tempfile.TemporaryDirectory(dir=repo) as tmp:
        work=Path(tmp)
        cmd=prepare(session,"capture",work,work,repo,contract,base,mounted["dioscuri_opaque.bin"],inventory,mounts)
        script=Path(cmd[3])
        code=script.read_text()
        stop="names=[f'image_{i:04d}.png' for i in range(138)]"
        check(code.count(stop)==1,"capture preamble boundary")
        record("capture_program_sha256",kaggle.sha(script))
        script.write_text(code.split(stop)[0])
        run("generated_environment_preamble",cmd,work)
        probe=read(work/"environment.json")
        equal(probe["status"],"CLEAN_VM_PRECHECK_PASS","live generated environment")
        equal(probe["assets"],pins["weight_pins"],"live model inventory")
        record("generated_boundary",read(work/"boundary.json"))
    runroot=repo/"actual"
    runroot.mkdir()
    prior=repo/("journal-"+kaggle.JOURNAL+".json")
    journal=runroot/"attempts.json"
    def custody(path,pin):
        external=output/("journal-"+pin+".json")
        shutil.copyfile(path,external)
        retain(external,pin)
    check(latest!=kaggle.JOURNAL,"use the externally retained migrated prospective journal")
    shutil.copyfile(mounted["journal-"+latest+".json"],journal)
    pin=latest
    runner=Runner(runroot,contract,custody,pin)
    migration=runner.events[1]
    equal(migration["previous_sha256"],kaggle.JOURNAL,"original custody")
    equal(migration["previous_events"],read(prior),"custody origin")
    equal(migration["new_runner"],revision,"prospective release")
    complete=[e["stage"] for e in runner.events if e["status"]=="complete"]
    equal(complete,list(archives),"all completed checkpoints required on resume")
    for stage,name in archives.items():
        runner.restore(stage,mounted[name],mounts[name])
    check(runner.events[-1]["status"] in ("migrated","complete"),"failed or interrupted stage requires custody review")
    record("journal",pin)
    record("stage_budget_remaining",14400-sum(e.get("seconds",0) for e in runner.events))
    record("session_seconds_remaining",deadline-time.monotonic())
    check(deadline-time.monotonic() >= (8400 if not complete else 1200),"insufficient live session after setup")
    record("preflight","PASS")
    def stage_run(mode):
        stage=mode.lower()
        check(stage in STAGES,"select exactly one stage")
        kaggle.inputs(Path("/kaggle/input"),mounts)
        space(repo,output,plan)
        def launch(argv,work,d,end,s):
            equal(argv[0],str(base/"venv/bin/python"),"pinned stage interpreter")
            if s=="capture":
                check(end+1200-time.monotonic() >= 8400,"capture window exhausted during external custody")
            monitor(argv,work,d,end,s,env,policy,plan,output)
        def execute(s,d,end):
            dispatch(runner,s,d,end,base,mounted["dioscuri_opaque.bin"],inventory,mounts,launch)
        for name,pin in read(repo/"inference_pins.json").items():
            equal(kaggle.sha(repo/name),pin,name)
        if (runner.root/stage).exists():
            runner.verify(stage,runner.root/stage)
        else:
            if stage=="capture":
                record("capture_gpu_memory",check_gpu(env))
            try:
                runner.run(stage,execute,deadline)
            except BaseException as exc:
                record("stage_failure",dict(stage=stage,reason=str(exc),journal=runner.pin))
                diag=runner.events[-1].get("diagnostics")
                if diag:
                    import zipfile
                    with zipfile.ZipFile(output/(stage+"_diagnostics.zip"),"w",zipfile.ZIP_DEFLATED) as z:
                        for f in (runner.root/diag).iterdir():
                            if f.suffix in (".json",".jsonl",".log"):
                                z.write(f,f.name)
                raise
        for name,pin in read(repo/"inference_pins.json").items():
            equal(kaggle.sha(repo/name),pin,name)
        path=output/(DS+"_"+stage+".bin")
        digest=runner.export(stage,path,digest_input("Existing archive SHA256: ") if path.exists() else None)
        retain(path,digest)
        record(stage,dict(archive=digest,seal=runner.verify(stage,runner.root/stage),journal=runner.pin))
    return stage_run
