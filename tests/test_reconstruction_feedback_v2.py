import copy,json,zipfile
from pathlib import Path
import numpy as np
import pytest
from src.novel_model.reconstruction_feedback_v2 import decide

def fixture():
    left=[f"a{i}" for i in range(6)]
    right=[f"b{i}" for i in range(6)]
    names=left+right
    base=dict(models=[dict(images=names,points=100)],errors={n:np.ones(100) for n in names})
    trial=dict(models=[dict(images=left,points=50),dict(images=right,points=50)],
               errors={n:np.full(100,0.8) for n in names})
    return [left,right],base,trial

def test_positive_shared_observations():
    assert decide(*fixture())["selected"]=="trial"

@pytest.mark.parametrize("case",["null_case","repeated_texture","legitimate_bridge","missing","lost_camera",
                                 "small_side","timeout","point_loss","selective_loss","roundoff","outlier","fragment"])
def test_abstain(case):
    sides,base,trial=fixture()
    if case in ("null_case","repeated_texture","legitimate_bridge"):
        trial["errors"]=copy.deepcopy(base["errors"])
    elif case=="missing":trial["errors"]["a0"][:]=np.nan
    elif case=="lost_camera":trial["models"][0]["images"]=sides[0][1:]
    elif case=="small_side":sides[0]=sides[0][:5]
    elif case=="timeout":trial=None
    elif case=="point_loss":trial["models"][1]["points"]=44
    elif case=="selective_loss":trial["errors"]["a1"][90:]=np.nan
    elif case=="roundoff":trial["errors"]={n:a-1e-8 for n,a in base["errors"].items()}
    elif case=="outlier":base["models"][0]["images"]=sides[1]
    elif case=="fragment":trial["models"].append(dict(images=["extra"],points=1))
    assert decide(sides,base,trial)["selected"]=="pass_through"

def test_real_receipts_lack_common_observations():
    with zipfile.ZipFile("phase4f_active_return.zip") as z:
        models=json.loads(z.read("result.json"))["models"]
    assert sorted(m["cameras"] for m in models)==[15,16,23]
    assert all("errors" not in m for m in models)
    assert decide(None,{},None)["selected"]=="pass_through"
