from pathlib import Path
import json,zipfile,io
import numpy as np
import pytest
from scripts.phase4h_replay import sha,verify,branch,MODEL
from src.novel_model.reconstruction_feedback_v2 import decide

def test_seal_and_whitelist():
    p=Path("reports/phase4h")
    verify((p/"decision.json").read_bytes(),(p/"decision.sha256").read_text())
    r=json.loads((p/"decision.json").read_bytes())
    verify((p/"selected.csv").read_bytes(),r["selected_sha256"])
    verify((p/"inference_only.zip").read_bytes(),r["package"])
    with zipfile.ZipFile(p/"inference_only.zip") as z:
        assert set(z.namelist())=={"model.zip","inputs.json"}
        verify(z.read("model.zip"),MODEL)
        d=json.loads(z.read("inputs.json"))
    assert set(d)=={"nodes","edges","sides","pass_through","active"}
    assert "errors" not in d["pass_through"]
    for _ in range(2):
        assert decide(d["sides"],d["pass_through"],d["active"])==r["decision"]

def test_wrong_branch():
    plan=Path("reports/phase4f/plan.json").read_bytes()
    with zipfile.ZipFile("phase4f_active_return.zip") as z,pytest.raises(AssertionError):
        branch(z,"pass_through",json.loads(plan),sha(plan))

def test_observation_intersection():
    from test_reconstruction_feedback_v2 import fixture
    sides,b,t=fixture()
    b["errors"]["a0"][0]=np.nan
    t["errors"]["a0"][1]=np.nan
    r=decide(sides,b,t)
    o=r["observations"]["a0"]
    assert (o["common"],o["lost"],o["gained"])==(98,1,1)
    del b["errors"]
    with pytest.raises(KeyError):decide(sides,b,t)

def test_projection_gauge():
    xyz=np.array([1.,2.,4.]); center=np.array([.5,0,1])
    def pixel(p,c):
        x=p-c
        return x[:2]/x[2]
    np.testing.assert_allclose(pixel(xyz,center),pixel(3*xyz+5,3*center+5))
