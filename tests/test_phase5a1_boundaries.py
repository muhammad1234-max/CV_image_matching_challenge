import csv
import time
from pathlib import Path
import sqlite3
import pytest
from test_phase5a1 import fx, STAGES, Fatal, Runner, read, save, file_sha
from src.runner_5a1.colab import execute
from src.runner_5a1.contract import equal

def test_timeout(fx):
    fx.through("original")
    def child(s,d,end):
        fx.child(s,d,end)
        (d/"prediction.csv").unlink()
        (d/"errors.npz").unlink()
        reason="trial deadline exceeded"
        save(d/"failure.json",dict(kind="EXECUTION_RECOVERABLE",stage=s,code="TRIAL_TIMEOUT",reason=reason))
        save(d/"branch.json",dict(available=False,reason=reason))
    fx.r.run("trial",child,time.monotonic()+14400)
    fx.r.run("decision",fx.child,time.monotonic()+14400)
    assert read(fx.r.root/"decision/selection.json")["action"]=="ABSTAIN"
    assert (fx.r.root/"decision/final.csv").read_bytes()==(fx.r.root/"original/prediction.csv").read_bytes()

def test_observation_failure(fx):
    fx.through("capture")
    def child(s,d,end):
        fx.child(s,d,end)
        b=read(d/"branch.json")
        b.update(valid_export=False,export_error="invalid depth",groups=[])
        save(d/"branch.json",b)
        save(d/"failure.json",dict(kind="EXECUTION_FATAL",stage=s,code="OBSERVATION_EXPORT",reason="invalid depth"))
        (d/"errors.npz").unlink()
    fx.r.run("original",child,time.monotonic()+14400)
    assert read(fx.r.root/"original/contract.json")["outcome"]=="OBSERVATIONS_UNAVAILABLE"
    with pytest.raises(Fatal):fx.r.run("trial",fx.child,time.monotonic()+14400)

@pytest.mark.parametrize("attribute",["duplicate","missing","membership","mask"])
def test_payload_semantics(fx,attribute):
    stage={"membership":"capture","mask":"trial"}.get(attribute,"original")
    if stage!="capture":fx.through(STAGES[STAGES.index(stage)-1])
    def child(s,d,end):
        fx.child(s,d,end)
        if attribute in ("missing","duplicate"):
            p=d/"prediction.csv"
            rows=p.read_text().splitlines()
            p.write_text("\n".join(rows[:-1]+([rows[-2]] if attribute=="duplicate" else [])))
        elif attribute=="membership":
            with sqlite3.connect(d/"parent.db") as db:
                db.execute("update images set name='unlisted.png' where image_id=1")
            job=read(d/"job.json")
            job["parent"]["backup_sha256"]=file_sha(d/"parent.db")
            save(d/"job.json",job)
        else:
            a=read(d/"input.json")
            a["forbidden"]=[]
            save(d/"input.json",a)
    with pytest.raises(Fatal):fx.r.run(stage,child,time.monotonic()+14400)

def test_interrupted_restart(fx):
    fx.r.record("started","capture",reserved=10000,parents={})
    restart=Runner(fx.r.root,fx.contract,lambda *a:None,fx.r.pin)
    with pytest.raises(Fatal):restart.run("capture",fx.child,time.monotonic()+14400)
    for pin in ("NEW","f"*64):
        with pytest.raises(Fatal):Runner(fx.r.root,fx.contract,lambda *a:None,pin)

def test_retry(fx):
    def fail(*args):raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):fx.r.run("capture",fail,time.monotonic()+14400)
    with pytest.raises(Fatal):fx.r.run("capture",fx.child,time.monotonic()+14400)
    with pytest.raises(Fatal):fx.r.run("capture",fx.child,time.monotonic()+7000,retry_pin=fx.r.pin)
    fx.r.run("capture",fx.child,time.monotonic()+14400,retry_pin=fx.r.pin)
    assert [e["status"] for e in fx.r.events]==["created","started","failed","started","complete"]

def test_wrong_archive_hash(fx):
    fx.through("capture")
    p=fx.tmp/"capture.zip"
    pin=fx.r.export("capture",p)
    with p.open("ab") as f:f.write(b"changed")
    with pytest.raises(Fatal):fx.r.restore("capture",p,pin)

@pytest.mark.parametrize("field",["dataset","seed","model","protocol"])
def test_wrong_identity(fx,field):
    fx.through("capture")
    d=fx.r.root/"capture"
    b=read(d/"contract.json")
    b["identity"][field]="other"
    save(d/"contract.json",b)
    with pytest.raises(Fatal):fx.r.verify("capture",d)

def test_scientific_invariance():
    pins=read(Path("reports/phase5a/integrity.json"))
    assert pins["config_sha256"]==file_sha("configs/novel_model/final_candidate_v1.yaml")
    assert pins["protocol_sha256"]==file_sha("reports/phase4i/inference_pins.json")
    for name,pin in {**pins["frozen_file_pins"],**read(Path("reports/phase4i/inference_pins.json"))}.items():
        # Workers are separately packed inside the pinned historical notebook.
        if name!="worker_pins.json" and not name.startswith("workers/"):
            assert file_sha(name)==pin

@pytest.mark.parametrize("failure",["timeout","digest","unknown"])
def test_colab_trial_adapter(fx,monkeypatch,failure):
    from src.runner_5a1 import colab
    from src.runner_5a1.contract import DS, Execution
    archive=fx.tmp/(DS+"_opaque.zip")
    archive.write_bytes(b"synthetic archive")
    fx.p["image_archive"]=file_sha(archive)
    fx.through("original")
    kit=fx.tmp/"kit"
    kit.mkdir()
    save(kit/"inference_pins.json",{})
    monkeypatch.setattr(colab.session,"KIT",kit)
    monkeypatch.setattr(colab.session,"identity",lambda ds:fx.contract.identity)
    monkeypatch.setattr(colab.session,"stage_command",lambda *a:["fixture","-I","-u","script"])
    def monitor(command,work,d,end,stage):
        if failure=="timeout":
            fx.child(stage,d,end)
            (d/"prediction.csv").unlink()
            (d/"errors.npz").unlink()
            raise TimeoutError(stage)
        save(d/"child_error.json",dict(type="RuntimeError",reason="fixture: trial input digest mismatch" if failure=="digest" else "unexpected"))
        raise RuntimeError("trial worker exit 1")
    monkeypatch.setattr(colab.session,"monitored",monitor)
    def child(stage,d,end):
        execute(fx.r,stage,d,end,content=fx.tmp)
    if failure=="timeout":
        fx.r.run("trial",child,time.monotonic()+14400)
        fx.r.run("decision",fx.child,time.monotonic()+14400)
        assert read(fx.r.root/"decision/selection.json")["action"]=="ABSTAIN"
    else:
        with pytest.raises(Fatal if failure=="digest" else Execution):
            fx.r.run("trial",child,time.monotonic()+14400)
        assert not (fx.r.root/"trial").exists()

def test_valid_archive_conflict(fx):
    import zipfile
    from src.novel_model.final_candidate_v1.workflow import seal
    fx.through("capture")
    d=fx.r.root/"capture"
    archive=fx.tmp/"existing.zip"
    prior=(d/"seal.json").read_bytes()
    (d/"worker.log").write_text("different valid run")
    seal(d,"capture",fx.contract.identity,{})
    with zipfile.ZipFile(archive,"w") as z:
        for p in d.iterdir():z.write(p,p.name)
    (d/"worker.log").unlink()
    (d/"seal.json").write_bytes(prior)
    pin=file_sha(archive)
    with pytest.raises(Fatal,match="conflict"):fx.r.export("capture",archive,pin)
    assert file_sha(archive)==pin
