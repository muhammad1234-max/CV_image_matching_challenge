import ast
import base64
from contextlib import contextmanager
import io
import json
from pathlib import Path
import shutil
import time
import zipfile
import pytest
from test_phase5a1 import Fixture
from phase5a3_cpu import CPU
from src.runner_5a3 import colab
from src.runner_5a3.provenance import DS,Contract
from src.runner_5a1.contract import Fatal,STAGES,read,save,file_sha
from src.runner_5a1.runner import Runner

@pytest.fixture
def lab(tmp_path,monkeypatch):
    fx=Fixture(tmp_path)
    cpu=CPU(fx,tmp_path)
    kit=tmp_path/"kit"
    nb=read(Path("notebooks/step05a3_dioscuri_colab.ipynb"))
    source="".join(nb["cells"][3]["source"])
    assigns={n.targets[0].id:n.value for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    for data in (assigns["EMBED"],assigns["ops"].args[0]):
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(data)))) as z:z.extractall(kit)
    monkeypatch.setattr(colab.session,"KIT",kit)
    archive=tmp_path/(DS+"_opaque.zip")
    with zipfile.ZipFile(archive,"w") as z:
        for n in fx.names:z.writestr(n,n.encode())
    rev=read(Path("reports/phase5a3/revision.json"))
    fx.p.update(config_sha256=rev["scientific"]["config_sha256"],source_revision=rev["scientific"]["source_revision"],benchmark=rev["scientific"]["benchmark"],
                image_archive=file_sha(archive),published_config=file_sha(cpu.config),
                worker=colab.session.identity(DS)["worker"],protocol_sha256=colab.session.identity(DS)["protocol"])
    fx.r=Runner(tmp_path/"experiment",Contract(fx.p,rev["runner"]),lambda *a:None,"NEW")
    monitor=cpu.monitor
    def monitored(cmd,work,d,end,stage):
        monitor(cmd,work,d,end,stage)
        if stage!=getattr(cpu,"target",None):return
        path=work/"identity.json"
        if cpu.failure=="missing":path.unlink()
        elif cpu.failure=="crash":
            path.rename(work/"identity.json.tmp")
            raise RuntimeError("injected crash")
        elif cpu.failure=="copy":shutil.copyfile(fx.r.root/"capture/identity.json",path)
        elif cpu.failure in ("prediction.csv","errors.npz"):(d/cpu.failure).unlink()
        elif cpu.failure=="changed":
            with (d/"prediction.csv").open("ab") as f:f.write(b"\n")
        elif cpu.failure in ("parent","dataset","proposed_mask","experiment_id"):
            obj=read(path)
            (obj if cpu.failure=="experiment_id" else obj["branch_execution"])[cpu.failure]="other"
            save(path,obj)
    monkeypatch.setattr(colab.session,"monitored",monitored)
    def run(stage):
        fx.r.run(stage,lambda s,d,end:colab.execute(fx.r,s,d,end,tmp_path),time.monotonic()+14400)
    return fx,cpu,run

def test_rehearsal(lab):
    fx,cpu,run=lab
    for s in STAGES:
        run(s)
        archive=fx.tmp/(s+".zip")
        pin=fx.r.export(s,archive)
        shutil.rmtree(fx.r.root/s)
        fx.r.restore(s,archive,pin)
        assert fx.r.export(s,archive,pin)==pin
    assert (fx.r.root/"decision/final.csv").read_bytes()==(fx.r.root/"original/prediction.csv").read_bytes()
    assert read(fx.r.root/"decision/selection.json")["action"]=="REJECT"

@pytest.mark.parametrize("stage,fault",[
    ("original","missing"),("trial","missing"),("original","parent"),("original","dataset"),
    ("trial","proposed_mask"),("original","copy"),("trial","copy"),("original","experiment_id"),
    ("original","changed"),("original","prediction.csv"),("original","errors.npz"),("original","crash"),
    ("trial","digest"),("trial","unknown")])
def test_production_failure(lab,stage,fault):
    fx,cpu,run=lab
    for s in STAGES[:STAGES.index(stage)]:run(s)
    cpu.failure=fault
    cpu.target=stage
    with pytest.raises((Fatal,RuntimeError,ValueError)):run(stage)
    assert not (fx.r.root/stage).exists() and not (fx.r.root/"decision").exists()
    if fault=="digest":assert fx.r.events[-1]["kind"]=="INTEGRITY_FATAL"

@pytest.mark.parametrize("fault",["timeout","geometry"])
def test_recoverable(lab,fault):
    fx,cpu,run=lab
    if fault=="geometry":
        import sqlite3
        with sqlite3.connect(cpu.original/"parent.db") as db:db.execute("delete from two_view_geometries")
    for s in STAGES:
        if s=="trial":cpu.failure=fault
        run(s)
    assert read(fx.r.root/"decision/selection.json")["action"]=="ABSTAIN"

@pytest.mark.parametrize("mode",["restore","reuse","predecessor","transfer"])
def test_provenance_boundaries(lab,monkeypatch,mode):
    fx,cpu,run=lab
    run("capture")
    if mode=="transfer":
        copy=shutil.copyfile
        def transfer(src,dst,*args,**kw):
            if Path(src).name=="identity.json" and Path(dst).parent.name.startswith("attempt-"):
                return
            return copy(src,dst,*args,**kw)
        monkeypatch.setattr(shutil,"copyfile",transfer)
        with pytest.raises(Fatal):run("original")
        return
    run("original")
    d=fx.r.root/"original"
    if mode=="predecessor":
        (d/"identity.json").unlink()
        with pytest.raises(Fatal):run("trial")
        return
    from src.novel_model.final_candidate_v1.workflow import seal
    fake=fx.tmp/"fake"
    shutil.copytree(d,fake)
    (fake/"identity.json").unlink()
    seal(fake,"original",fx.r.contract.identity,fx.r.parents("original"))
    archive=fx.tmp/"original.zip"
    with zipfile.ZipFile(archive,"w") as z:
        for p in fake.iterdir():z.write(p,p.name)
    with pytest.raises(Fatal):
        if mode=="restore":fx.r.restore("original",archive,file_sha(archive))
        else:fx.r.export("original",archive,file_sha(archive))

@pytest.mark.parametrize("stage",["original","trial"])
def test_pre_repair_children(lab,stage):
    from src.runner_5a1 import colab as previous
    fx,cpu,run=lab
    for s in STAGES[:STAGES.index(stage)]:run(s)
    with pytest.raises(Fatal,match="identity.json"):
        fx.r.run(stage,lambda s,d,end:previous.execute(fx.r,s,d,end,fx.tmp),time.monotonic()+14400)
    attempt=fx.r.root/fx.r.events[-1]["diagnostics"]
    assert (attempt/"prediction.csv").exists() and not (attempt/"identity.json").exists()

def test_notebook_cells(lab,monkeypatch):
    import sys,types
    fx,cpu,_=lab
    notebook=read(Path("notebooks/step05a3_dioscuri_colab.ipynb"))
    base=fx.tmp
    # Colab paths, prompts, downloads and fixture scientific pins are the external test boundary.
    files=types.SimpleNamespace(download=lambda name:shutil.copyfile(name,base/("download-"+Path(name).name)))
    monkeypatch.setitem(sys.modules,"google.colab",types.SimpleNamespace(files=files))
    def answer(prompt):
        if "Seconds actually" in prompt:return "14400"
        if "NEW" in prompt:return "NEW"
        if "externally" in prompt:return prompt.split()[-2]
        raise AssertionError(prompt)
    monkeypatch.setattr("builtins.input",answer)
    from phase5a3_cpu import separate_process_monitor
    monkeypatch.setattr(colab.session,"monitored",lambda *args:separate_process_monitor(cpu,base/"kit",*args))
    actual=colab.interface
    actual_execute=colab.execute
    monkeypatch.setattr(colab,"execute",lambda runner,s,d,end:actual_execute(runner,s,d,end,base))
    monkeypatch.setattr(colab,"Path",lambda value:base if value=="/content" else Path(value))
    monkeypatch.setattr(colab,"interface",lambda root,end,pins,rev:actual(root,end,fx.p,rev))
    import time
    scope=dict(SESSION_START=time.monotonic(),subprocess=__import__("subprocess"))
    src="".join(notebook["cells"][3]["source"])
    src=src.replace("'/content/phase4i_kit'",repr(str(base/"kit"))).replace("'/content/phase5a3_runs'",repr(str(base/"nb_runs"))).replace("'/content'",repr(str(base)))
    with monkeypatch.context() as m:
        m.setattr(sys,"path",list(sys.path))
        exec(src,scope)
        for cell in notebook["cells"][4:]:exec("".join(cell["source"]),scope)
    assert (base/("download-"+DS+"_decision.zip")).is_file()
