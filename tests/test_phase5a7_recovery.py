from pathlib import Path
import shutil
import sqlite3
import time
from contextlib import closing
import pytest
from src.novel_model.geometry_capture_v1.database import logical
from src.runner_5a1.contract import Fatal,read,save,file_sha
from scripts import phase5a7_recovery as rec
from scripts.phase5a6_production import Contract
from test_phase5a1 import Fixture

PREVIOUS=Path("journal-"+rec.LATEST+".json")
AUTH="Phase 5A.7 single recovery CAPTURE"
PROOF="a"*64

@pytest.fixture
def env(tmp_path,monkeypatch):
    fx=Fixture(tmp_path)
    ident=read(PREVIOUS)[0]["identity"]
    fx.p.update(worker=ident["worker"],protocol_sha256=ident["protocol"])
    contract=Contract(fx.p,"test-only-recovery-release")
    registry=rec.Custodian(tmp_path/"admin")
    event=rec.transition(PREVIOUS,contract,AUTH,PROOF,True)
    searches=dict.fromkeys(("local","kaggle_outputs","versions","private_datasets","other_destinations","runtime"),"checked_absent")
    monkeypatch.setattr(rec,"validate_persistence",lambda **kwargs:PROOF)
    monkeypatch.setattr(rec,"private_saved_output",lambda *args:None)
    def start(root=None):
        return rec.initiate(root or tmp_path/"recovery",PREVIOUS,contract,registry,event,{},searches)
    return fx,contract,registry,event,searches,start

def capture(fx,d):
    fx.child("capture",d,0)
    with closing(sqlite3.connect(d/"parent.db")) as db:
        db.execute("alter table matches add column rows integer")
        db.execute("alter table matches add column data blob")
        db.execute("insert into matches select pair_id,rows,data from two_view_geometries")
        db.commit()
        records=logical(db)
    job=read(d/"job.json")
    job["parent"]=dict(original_logical=records,backup_sha256=file_sha(d/"parent.db"))
    save(d/"job.json",job)

@pytest.mark.parametrize("problem",["available","unsearched","no_authorization","no_proof","missing_archive"])
def test_recovery_admission(env,problem,monkeypatch):
    fx,contract,reg,event,searches,start=env
    with pytest.raises(Fatal):
        if problem=="missing_archive":
            rec.restore_historical(fx.tmp/"invalid",PREVIOUS,contract,fx.tmp/"missing")
        elif problem=="no_authorization":
            rec.transition(PREVIOUS,contract,"",PROOF,True)
        else:
            if problem=="no_proof":monkeypatch.setattr(rec,"validate_persistence",lambda **k:"wrong")
            else:searches["private_datasets"]=problem
            start()
    assert not list(reg.folder.glob("*.next"))

def test_history_debit_and_activation(env):
    fx,contract,reg,event,searches,start=env
    r=start()
    initial=r.pin
    past=read(PREVIOUS)
    assert file_sha(PREVIOUS)==rec.LATEST and r.events[:-1]==past
    def execute(s,d,end):
        assert 0<end-time.monotonic() <=14400-rec.DEBIT-1200
        capture(fx,d)
    r.run("capture",execute,time.monotonic()+14400)
    completed=r.pin
    archive=fx.tmp/"external.bin"
    pin=r.export("capture",archive)
    with pytest.raises(Fatal):r.run("original",lambda *a:None,time.monotonic()+14400)
    with pytest.raises(Fatal):r.registered("capture",rec.LOST)
    with pytest.raises(Fatal):start(fx.tmp/"replay")
    with pytest.raises(Fatal):reg.latest(initial)
    shutil.rmtree(r.root/"capture")
    restarted=rec.RecoveryRunner(r.root,contract,reg,completed,PREVIOUS,event)
    with pytest.raises(Fatal):restarted.run("capture",execute,time.monotonic()+14400)
    restarted.activate("capture",archive,pin,reg.folder/("journal-"+completed+".json"),completed,"private-version-fixture/1")
    assert restarted.parents("original")=={"capture":r.events[-1]["seal"]}
    assert restarted.contract.binding("original",r.root)["parent"]==file_sha(r.root/"capture/parent.db")
    assert restarted.events[-1]["parent"]==file_sha(r.root/"capture/parent.db")
    assert sum(e.get("seconds",0) for e in restarted.events)==rec.DEBIT+r.events[-1]["seconds"]
    assert restarted.events[:len(past)]==past and len([e for e in restarted.current() if e["status"]=="started"])==1

@pytest.mark.parametrize("problem",["missing_output","foreign_identity","wrong_archive","synthetic","lost_seal"])
def test_fatal_and_single_use(env,problem):
    fx,contract,reg,event,searches,start=env
    r=start()
    def execute(s,d,end):
        capture(fx,d)
        if problem=="missing_output":(d/"input_receipt.json").unlink()
        if problem=="foreign_identity":
            data=read(d/"identity.json")
            data["seed"]=0
            save(d/"identity.json",data)
    if problem in ("missing_output","foreign_identity"):
        with pytest.raises(Fatal):r.run("capture",execute,time.monotonic()+14400)
        assert r.events[-1]["kind"]=="INTEGRITY_FATAL"
        with pytest.raises(Fatal):r.run("capture",execute,time.monotonic()+14400)
        with pytest.raises(Exception):r.export("capture",fx.tmp/"invalid")
        return
    r.run("capture",execute,time.monotonic()+14400)
    if problem=="lost_seal":
        with pytest.raises(Fatal):r.registered("capture",rec.LOST)
    else:
        archive=fx.tmp/"archive.bin"
        r.export("capture",archive)
        if problem=="synthetic":
            archive=Path("phase5a5_synthetic_checkpoint.bin")
            pin=file_sha(archive)
        else:pin="0"*64
        with pytest.raises(Fatal):r.activate("capture",archive,pin,r.journal,r.pin,"test/1")
    assert not any(e["status"]=="externally_restored" for e in r.events)
from test_phase5a3 import lab

@pytest.mark.parametrize("fault",["complete","timeout"])
def test_both_mappers_use_durable_parent(lab,monkeypatch,fault):
    from scripts.phase5a6_production import dispatch
    from src.runner_5a3 import colab
    import phase5a3_cpu
    fx,cpu,_=lab
    contract=Contract(fx.p,"test-generated-recovery")
    event=rec.transition(PREVIOUS,contract,AUTH,PROOF,True)
    registry=rec.Custodian(fx.tmp/"admin")
    monkeypatch.setattr(rec,"validate_persistence",lambda **kwargs:PROOF)
    monkeypatch.setattr(rec,"private_saved_output",lambda *args:None)
    inventory=dict.fromkeys(("local","kaggle_outputs","versions","private_datasets","other_destinations","runtime"),"checked_absent")
    runner=rec.initiate(fx.tmp/"recover",PREVIOUS,contract,registry,event,{},inventory)
    namespace=phase5a3_cpu.NS
    def native_ns(**values):
        if "get_device_name" in values:values["device_count"]=lambda:1
        return namespace(**values)
    monkeypatch.setattr(phase5a3_cpu,"NS",native_ns)
    with closing(sqlite3.connect(cpu.original/"parent.db")) as db:
        db.execute("alter table matches add column rows integer")
        db.execute("alter table matches add column data blob")
        db.execute("insert into matches select pair_id,rows,data from two_view_geometries")
        db.commit()
    for stage in ("capture","original","trial","decision"):
        cpu.failure=fault if stage=="trial" else ""
        def execute(s,d,end):
            dispatch(runner,s,d,end,fx.tmp/"ns64_clean",fx.tmp/(colab.DS+"_opaque.zip"),
                     "Tesla T4, 580.82.07, 15360 MiB",{},cpu.monitor)
        runner.run(stage,execute,time.monotonic()+14400)
        archive=fx.tmp/(stage+".bin")
        digest=runner.export(stage,archive)
        completed=runner.pin
        shutil.rmtree(runner.root/stage)
        runner.activate(stage,archive,digest,registry.folder/("journal-"+completed+".json"),completed,"disposable-fixture/1")
    dbpin=file_sha(runner.root/"capture/parent.db")
    for stage in ("original","trial"):
        assert read(runner.root/stage/"identity.json")["branch_execution"]["parent"]==dbpin
    assert read(runner.root/"decision/selection.json")["action"]==("ABSTAIN" if fault=="timeout" else "REJECT")

def test_unresolved_start_encumbers_budget(env):
    fx,contract,reg,event,searches,start=env
    r=start()
    r.record("started","capture",reserved=9700,parents={})
    reopened=rec.RecoveryRunner(r.root,contract,reg,r.pin,PREVIOUS,event)
    for stage in ("capture","original"):
        with pytest.raises(Fatal):reopened.run(stage,lambda *a:None,time.monotonic()+14400)
    assert reopened.events[-1]["reserved"]==9700

def test_mutable_producer_workdir_is_not_durable(tmp_path):
    a=tmp_path/"a.bin"
    j=tmp_path/"j.json"
    a.write_bytes(b"a")
    j.write_bytes(b"[]")
    with pytest.raises(Fatal):rec.private_saved_output(a,j,"private/fixture/versions/1")
