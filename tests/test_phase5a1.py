"""Only the child is synthetic. Runner contracts, database replay, seals and selector are real."""
import csv
import json
from pathlib import Path
import shutil
import sqlite3
import time
import zipfile
import numpy as np
import pytest
from src.runner_5a1.contract import Contract, DS, Fatal, digest, read, save, file_sha, COLUMNS, STAGES
from src.runner_5a1.runner import Runner
from src.novel_model.geometry_capture_v1.database import logical, MAX_ID
from src.novel_model.reconstruction_feedback_v1 import make_copy
from src.novel_model.final_candidate_v1.workflow import partition, select, seal

class Fixture:
    def __init__(self,tmp):
        self.tmp=tmp
        self.names=[f"image_{i:04d}.png" for i in range(138)]
        hashes={n:digest(n.encode()) for n in self.names}
        self.p=dict(images=138,protocol_sha256="protocol",worker="worker",config_sha256="config",
                    source_revision="source",published_config="published",benchmark={},weight_pins={},
                    image_permutation=digest("\n".join(hashes.values()).encode()),options_sha256=digest(b"options"),
                    image_archive="archive",source_inventory="inventory",environment=digest(b"freeze"))
        self.hashes=hashes
        self.contract=Contract(self.p,"synthetic-runner")
        self.r=Runner(tmp/"run",self.contract,lambda *args:None,"NEW")
        self.geometry=True

    def child(self,stage,d,end):
        save(d/"usage.json",dict(stage=stage,samples=1,peak_pss=1,peak_rss=2,peak_uss=1))
        (d/"heartbeat.jsonl").write_text(json.dumps(dict(stage=stage,totals=dict(pss=1,rss=2,uss=1))))
        root=self.r.root
        if stage=="decision":
            return select(root,d)
        if stage=="capture":
            db=sqlite3.connect(d/"parent.db")
            for table,schema in dict(cameras="camera_id integer",images="image_id integer,name text,camera_id integer",
                                      matches="pair_id integer",keypoints="image_id integer,rows integer",
                                      two_view_geometries="pair_id integer,rows integer,cols integer,data blob").items():
                db.execute(f"create table {table} ({schema})")
            db.execute("insert into cameras values(1)")
            for i,n in enumerate(self.names,1):
                db.execute("insert into images values(?,?,1)",(i,n))
                db.execute("insert into keypoints values(?,40)",(i,))
                if self.geometry and i<=12:
                    for j in range(1,i):
                        count=40 if (i<=6)==(j<=6) else int((i,j)==(7,6))
                        if count:
                            db.execute("insert into two_view_geometries values(?,?,2,?)",(j*MAX_ID+i,count,bytes(count*8)))
            db.commit()
            records=logical(db)
            db.close()
            save(d/"job.json",dict(image_hashes=self.hashes,options="options",parent=dict(
                original_logical=records,backup_sha256=file_sha(d/"parent.db"))))
            save(d/"identity.json",dict(source_revision="source",config_sha256="published",benchmark={},seed=20261111,
                permutation_sha256=self.p["image_permutation"],experiment_id=f"ns64_isolated_v1_{DS}_20261111",
                method_id="ns64_mast3rhybrid_022c",comparison_baseline_id="organizer_baseline_v1_dev_full"))
            save(d/"input_receipt.json",dict(archive_sha256=self.p["image_archive"],permutation_sha256=self.p["image_permutation"]))
            save(d/"checkpoint_verify_matches.json",dict(status="PASS",stage="verify_matches",verified_pairs=31 if self.geometry else 0))
            save(d/"environment.json",dict(status="CLEAN_VM_PRECHECK_PASS",source_inventory_hash="inventory",assets={},environment_freeze="freeze"))
            save(d/"boundary.json",dict(status="PASS"))
            return
        for name in ("environment.json","identity.json","boundary.json","input_receipt.json"):
            shutil.copyfile(root/"capture"/name,d/name)
        parent=root/"capture/parent.db"
        pin=file_sha(parent)
        prop,mask=partition(parent,pin,self.names)
        if stage=="trial" and not prop:
            reason="no eligible fully registered component or original export"
            save(d/"branch.json",dict(available=False,reason=reason))
            save(d/"failure.json",dict(kind="EXECUTION_RECOVERABLE",stage=stage,code="NO_ELIGIBLE_GEOMETRY",reason=reason))
            return
        masked=d/"tmp.db"
        audit=make_copy(parent,masked,pin,mask if stage=="trial" else [])
        masked.unlink()
        save(d/"input.json",audit)
        groups=[self.names[:12]] if stage=="original" else list(prop["sides"])
        save(d/"branch.json",dict(available=True,valid_export=True,proposal=prop if stage=="original" else None,
                                  groups=groups,models=[dict(images=g,points=40) for g in groups]))
        with (d/"prediction.csv").open("w",newline="") as f:
            w=csv.writer(f)
            w.writerow(COLUMNS)
            for i,n in enumerate(self.names):
                scene=next((str(k) for k,g in enumerate(groups) if n in g),"outliers")
                rot="1;0;0;0;1;0;0;0;1" if i<12 else ";".join(["nan"]*9)
                w.writerow([i,DS,scene,n,rot,"0;0;0" if i<12 else "nan;nan;nan"])
        np.savez(d/"errors.npz",**{n:np.full(40,1.0 if stage=="original" else .7) for n in self.names[:12]})

    def through(self,stage):
        for s in STAGES[:STAGES.index(stage)+1]:
            self.r.run(s,self.child,time.monotonic()+14400)

@pytest.fixture
def fx(tmp_path):
    return Fixture(tmp_path)

@pytest.mark.parametrize("geometry,action",[(True,"ACCEPT"),(False,"ABSTAIN")])
def test_roundtrip(fx,geometry,action):
    fx.geometry=geometry
    fx.through("decision")
    assert read(fx.r.root/"decision/selection.json")["action"]==action
    if not geometry:
        assert (fx.r.root/"decision/final.csv").read_bytes()==(fx.r.root/"original/prediction.csv").read_bytes()
    before=list(fx.r.events)
    for s in STAGES:
        archive=fx.tmp/(s+".zip")
        pin=fx.r.export(s,archive)
        assert fx.r.export(s,archive,pin)==pin
        shutil.rmtree(fx.r.root/s)
        fx.r.restore(s,archive,pin)
    assert fx.r.events==before

@pytest.mark.parametrize("stage,missing",[
    ("capture","parent.db"),("capture","identity.json"),("capture","environment.json"),
    ("original","branch.json"),("original","prediction.csv"),("original","errors.npz"),("original","input.json"),
    ("trial","input.json"),("decision","selection.json"),("decision","final.csv")])
def test_missing(fx,stage,missing):
    if stage!="capture":
        fx.through(STAGES[STAGES.index(stage)-1])
    def child(s,d,end):
        fx.child(s,d,end)
        (d/missing).unlink()
    with pytest.raises(Fatal):
        fx.r.run(stage,child,time.monotonic()+14400)
    assert not (fx.r.root/stage).exists()

def test_time_only(fx):
    fx.through("capture")
    with pytest.raises(Fatal):
        fx.r.run("original",lambda *a:None,time.monotonic()+14400)
    d=fx.r.root/"original"
    d.mkdir()
    save(d/"time.json",dict(seconds=0))
    seal(d,"original",fx.contract.identity,fx.r.parents("original"))
    archive=fx.tmp/"bad.zip"
    with zipfile.ZipFile(archive,"w") as z:
        for f in d.iterdir():z.write(f,f.name)
    for op in (lambda:fx.r.verify("original",d),lambda:fx.r.export("original",archive),
               lambda:fx.r.restore("original",archive,file_sha(archive)),lambda:fx.r.parents("trial")):
        with pytest.raises(Fatal):op()

@pytest.mark.parametrize("cause,kind",[(RuntimeError("fixture: trial input digest mismatch"),"INTEGRITY_FATAL"),
                                      (RuntimeError("unknown"),"EXECUTION_FATAL"),(KeyboardInterrupt(),"EXECUTION_FATAL")])
def test_execution_failure(fx,cause,kind):
    fx.through("original")
    before=file_sha(fx.r.root/"original/prediction.csv")
    def fail(*args):raise cause
    with pytest.raises((RuntimeError,Fatal,KeyboardInterrupt)):
        fx.r.run("trial",fail,time.monotonic()+14400)
    assert fx.r.events[-1]["kind"]==kind and fx.r.events[-1]["seconds"]>=0
    assert file_sha(fx.r.root/"original/prediction.csv")==before
    restarted=Runner(fx.r.root,fx.contract,lambda *args:None,fx.r.pin)
    for stage in ("trial","decision"):
        with pytest.raises(Fatal):restarted.run(stage,fx.child,time.monotonic()+14400)
    assert not (fx.r.root/"decision").exists()

@pytest.mark.parametrize("field",["identity","source","stage","config","parents","runner","permutation","outcome"])
def test_false_receipt(fx,field):
    fx.through("capture")
    d=fx.r.root/"capture"
    header=read(d/"contract.json")
    header[field]="wrong"
    save(d/"contract.json",header)
    seal(d,"capture",fx.contract.identity,{})
    with pytest.raises(Fatal):fx.r.parents("original")

@pytest.mark.parametrize("name",["../bad","/bad","a/b","C:evil","dup.json"])
def test_bad_zip(fx,name):
    archive=fx.tmp/"bad.zip"
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr(name,"")
        if name=="dup.json":z.writestr(name,"")
    with pytest.raises(Fatal):fx.r.restore("capture",archive,file_sha(archive))

def test_stale(fx):
    fx.through("capture")
    archive=fx.tmp/"capture.zip"
    with zipfile.ZipFile(archive,"w") as z:z.writestr("time.json","{}")
    with pytest.raises(Fatal):fx.r.export("capture",archive,file_sha(archive))
    assert not archive.exists() and list(fx.tmp.glob("*.quarantine-*"))
