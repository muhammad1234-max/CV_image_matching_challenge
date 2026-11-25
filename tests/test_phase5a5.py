import json
from pathlib import Path
import pytest
from src.runner_5a5.kaggle import inputs,sha,transition,gpu_env,disk_plan,JOURNAL
from src.runner_5a1.runner import Runner
from src.runner_5a3.provenance import Contract

def test_mount_inventory(tmp_path):
    p=tmp_path/"approved.bin"
    p.write_bytes(b"fixture")
    assert inputs(tmp_path,{p.name:sha(p)})=={p.name:p}
    (tmp_path/"unexpected").write_bytes(b"")
    with pytest.raises(ValueError):inputs(tmp_path,{p.name:sha(p)})

def test_absent_input(tmp_path):
    with pytest.raises(ValueError):inputs(tmp_path/"missing",{})

def test_wrong_artifact(tmp_path):
    (tmp_path/"a").write_text("changed")
    with pytest.raises(ValueError):inputs(tmp_path,{"a":"bad"})

def test_selected_device():
    env=gpu_env("1, GPU-second, Tesla T4, 580.178.04, 15360\n0, GPU-first, Tesla T4, 580.178.04, 15360")
    assert env["CUDA_VISIBLE_DEVICES"]=="GPU-first" and env["WORLD_SIZE"]=="1"

def test_wrong_device():
    with pytest.raises(ValueError):gpu_env("0, GPU-a, Tesla P100, x, 16384")

def test_disk_scaling():
    assert sum(disk_plan(200).values())-sum(disk_plan(100).values())==700

def test_journal(tmp_path):
    rev=json.loads(Path("reports/phase5a3/revision.json").read_bytes())
    contract=Contract(rev["scientific"],"synthetic-new-revision")
    source=next(Path.cwd().glob("journal-"+JOURNAL+".json"))
    dest=tmp_path/"attempts.json"
    pin=transition(source,dest,contract.identity,contract.runner)
    runner=Runner(tmp_path,contract,lambda *a:None,pin)
    assert runner.events[-1]["previous_sha256"]==JOURNAL
    assert sum(e.get("seconds",0) for e in runner.events)==0
    assert transition(source,dest,contract.identity,contract.runner)==pin
    assert all(e["status"]!="started" for e in runner.events)

def test_changed_prior(tmp_path):
    source=tmp_path/"old"
    source.write_text("[]")
    with pytest.raises(ValueError):transition(source,tmp_path/"new",{},"new")

def test_journal_collision(tmp_path):
    target=tmp_path/"attempts.json"
    target.write_text("[]")
    source=Path("journal-"+JOURNAL+".json")
    with pytest.raises(ValueError):transition(source,target,json.loads(source.read_bytes())[0]["identity"],"new")
    assert target.read_text()=="[]"
