import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sqlite3
import time
import pytest
from test_phase5a3 import lab
from scripts.phase5a6_production import dispatch, monitor, Contract
from src.runner_5a1.contract import DS,STAGES,Fatal,read
from src.runner_5a1.runner import Runner


@pytest.mark.parametrize("fault",["complete","timeout","digest"])
def test_dispatch_and_custody(lab,fault,monkeypatch):
    fx,cpu,_=lab
    kit=Path(__file__).resolve().parents[1]
    python=os.environ.get("NS64_FIXTURE_PYTHON",sys.executable)
    revision=os.environ.get("NS64_TEST_RELEASE","fixture-release")
    runner=Runner(fx.tmp/"roundtrip",Contract(fx.p,revision),lambda *args:None,"NEW")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES",os.environ.get("CUDA_VISIBLE_DEVICES","fixture-GPU"))
    with sqlite3.connect(cpu.original/"parent.db") as db:
        db.execute("alter table matches add column rows integer")
        db.execute("alter table matches add column data blob")
        db.execute("insert into matches select pair_id,rows,data from two_view_geometries")
    for stage in STAGES:
        def launch(argv,work,d,end,s):
            assert argv[0]==str(cpu.base/"ns64_clean/venv/bin/python")
            spec=dict(argv=argv,work=str(work),dest=str(d),end=end,stage=s,
                      base=str(cpu.base),pins=fx.p,names=fx.names,python=python,
                      visible=os.environ["CUDA_VISIBLE_DEVICES"],fault=fault if s=="trial" else "")
            specfile=work/"fixture.json"
            specfile.write_text(json.dumps(spec))
            args=[python,*(["-I"] if "NS64_FIXTURE_PYTHON" in os.environ else []),str(kit/"scripts/phase5a6_fixture.py"),str(specfile)]
            if "NS64_FIXTURE_PYTHON" in os.environ:
                # Native-service fixture supplies its own heartbeat. Retain real
                # subprocess-monitor telemetry for the contract instead.
                metrics=work/"fixture-monitor"
                metrics.mkdir()
                try:
                    monitor(args,work,metrics,end,s,os.environ.copy(),
                            read(kit/"configs/novel_model/phase4g1_resource_policy.yaml"),
                            {"archive":0,"export_margin":0},metrics)
                except RuntimeError as exc:
                    if str(exc)==f"{s} exit 124":
                        raise TimeoutError(s) from exc
                    raise
                finally:
                    for name in ("heartbeat.jsonl","worker.log"):
                        if (metrics/name).exists():
                            shutil.copyfile(metrics/name,d/name)
                return
            result=subprocess.run(args)
            if result.returncode==124:
                raise TimeoutError(s)
            if result.returncode:
                raise RuntimeError("fixture child exit "+str(result.returncode))
        def execute(s,d,end):
            dispatch(runner,s,d,end,cpu.base/"ns64_clean",cpu.base/(DS+"_opaque.zip"),
                     "Tesla T4, 580.82.07, 15360 MiB",{},launch)
        if stage=="trial" and fault=="digest":
            with pytest.raises(Fatal,match="digest"):
                runner.run(stage,execute,time.monotonic()+14400)
            assert not (runner.root/stage).exists()
            return
        runner.run(stage,execute,time.monotonic()+14400)
        archive=fx.tmp/(stage+".bin")
        pin=runner.export(stage,archive)
        shutil.rmtree(runner.root/stage)
        runner.restore(stage,archive,pin)
    assert read(runner.root/"decision/selection.json")["action"]==("ABSTAIN" if fault=="timeout" else "REJECT")
