from pathlib import Path
import json
import pytest
from src.novel_model.resource_monitor_v2 import guard,kb,process_row,cgroups

policy=json.loads(Path("configs/novel_model/phase4g1_resource_policy.yaml").read_bytes())
def sample():
    return dict(host=dict(MemTotal=20<<30,MemAvailable=8<<30),groups=[dict(path="/",limit=16<<30,current=8<<30,events=dict(oom=0,oom_kill=0))],gpu=[("T4",15360,9000)],full_avg10=0,totals=dict(rss=22<<30,pss=7<<30))
@pytest.mark.parametrize("failure",["safe_shared","oom","gpu","pressure","host","cgroup","unlimited"])
def test_guard(failure):
    r=sample()
    if failure=="safe_shared":
        guard(r,sample(),policy)
        return
    if failure=="oom":r["groups"][0]["events"]["oom_kill"]=1
    if failure=="gpu":r["gpu"]=[("T4",15360,15359)]
    if failure=="pressure":r["full_avg10"]=12
    if failure=="host":r["host"]["MemAvailable"]=100
    if failure=="cgroup":r["groups"][0]["current"]=16<<30
    if failure=="unlimited":r["groups"][0]["limit"]=None
    with pytest.raises(RuntimeError):guard(r,sample(),policy)

def test_rollup(tmp_path):
    assert process_row(1,tmp_path)["exited"]
    path=tmp_path/"1/smaps_rollup"
    path.parent.mkdir()
    path.write_text("Rss: 30 kB\nPss: 20 kB\nPrivate_Clean: 4 kB\nPrivate_Dirty: 6 kB\n")
    assert process_row(1,tmp_path)==dict(pid=1,rss=30720,pss=20480,uss=10240)
    for invalid in ("Rss: nan kB","Pss: 8 kB"):
        path.write_text(invalid)
        assert process_row(1,tmp_path)["unavailable"]

def test_v1(tmp_path):
    (tmp_path/"self").mkdir()
    (tmp_path/"self/cgroup").write_text("2:memory:/")
    with pytest.raises(ValueError):cgroups(tmp_path)

def test_nested_v2_limit(tmp_path):
    ctrl=tmp_path/"cg"
    child=ctrl/"job"
    child.mkdir(parents=True)
    for path,limit in [(ctrl,"8000000000"),(child,"max")]:
        (path/"memory.current").write_text("10000")
        (path/"memory.max").write_text(limit)
        (path/"memory.events").write_text("oom 0\noom_kill 0")
    (tmp_path/"self").mkdir()
    (tmp_path/"self/cgroup").write_text("0::/job")
    mount=ctrl.as_posix().replace(" ",r"\040")
    (tmp_path/"self/mountinfo").write_text(f"30 1 0:1 / {mount} rw - cgroup2 cgroup rw")
    assert [g["limit"] for g in cgroups(tmp_path)]==[None,8000000000]

def test_recovery_archive_on_guard_error():
    import ast
    from scripts.phase4g_build_notebooks import RUN
    path=Path("notebooks/step04g1_stairs_colab.ipynb")
    src="".join(json.loads(path.read_bytes())["cells"][-1]["source"])
    ast.parse(src)
    assert src.index("guard(initial,initial,POLICY)")<src.index("launch([str(work/'worker.py')")
    assert "os.killpg(proc.pid,signal.SIGKILL)" in src
    assert "with zipfile.ZipFile(archive) as check: assert check.testzip() is None" in src

