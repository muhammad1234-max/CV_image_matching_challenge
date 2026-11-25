import ast
import base64
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from src.novel_model.final_candidate_v1 import decide,file_sha,partition
from src.novel_model.final_candidate_v1.workflow import read,save,seal,select,verify
from src.novel_model.final_candidate_v1.submission import COLUMNS,validate,write_submission
from src.novel_model.final_candidate_v1.session import unpack,stage_command
from scripts.phase4i_evaluate import aggregate

NAMES=[f"image_{i:04d}.png" for i in range(12)]
SIDES=[NAMES[:6],NAMES[6:]]


def branches():
    a={n:np.linspace(.5,2.5,100) for n in NAMES}
    b={n:x-.2 for n,x in a.items()}
    return ([NAMES],a),(SIDES,b)


def test_accept_and_determinism():
    a,b=branches()
    first=decide(SIDES,a,b)
    assert first==decide(SIDES,a,b) and first["action"]=="ACCEPT"


@pytest.mark.parametrize("at,reason",[(slice(-3,None),"lost observation coverage"),
                                       (slice(0,6),"lost observation coverage")])
def test_selective_deletion(at,reason):
    a,b=branches()
    b[1][NAMES[0]][at]=np.nan
    assert decide(SIDES,a,b)["reason"]==reason


def test_lost_gained_and_cost():
    a,b=branches()
    a[1][NAMES[0]][0]=np.nan
    b[1][NAMES[0]][1]=np.nan
    ev=decide(SIDES,a,b)["per_image"][NAMES[0]]
    assert ev["lost"]==ev["gained"]==1 and ev["common"]==98
    assert ev["cost_delta_px"]==pytest.approx((4-98*.2)/99)


@pytest.mark.parametrize("mode,action",[("unchanged","REJECT"),("wrong_groups","REJECT"),
                         ("missing","ABSTAIN"),("small","ABSTAIN"),("insufficient","ABSTAIN")])
def test_fallbacks(mode,action):
    a,b=branches()
    if mode=="unchanged": b=(SIDES,a[1])
    if mode=="wrong_groups": b=([NAMES],b[1])
    if mode=="missing": b=None
    if mode=="small": a=(SIDES,a[1])
    if mode=="insufficient": b[1][NAMES[0]][20:]=np.nan
    assert decide(SIDES,a,b)["action"]==action


@pytest.mark.parametrize("bad",[np.inf,-2.0])
def test_invalid_residual(bad):
    a,b=branches()
    b[1][NAMES[0]][5]=bad
    with pytest.raises(ValueError): decide(SIDES,a,b)


def test_mixed_shapes_and_duplicate_membership():
    a,b=branches()
    b[1][NAMES[0]]=b[1][NAMES[0]][:-1]
    with pytest.raises(ValueError):decide(SIDES,a,b)
    with pytest.raises(ValueError):decide(SIDES,a,([*SIDES,SIDES[0]],b[1]))


@pytest.mark.parametrize("available",[True,False])
def test_sealed_selection(tmp_path,available):
    a,b=branches()
    for stage,(groups,err) in zip(("original","trial"),(a,b)):
        folder=tmp_path/stage
        folder.mkdir()
        save(folder/"branch.json",dict(available=stage=="original" or available,valid_export=True,groups=groups,
                                      proposal={"sides":SIDES}))
        np.savez(folder/"errors.npz",**err)
        (folder/"prediction.csv").write_bytes(stage.encode())
    dest=tmp_path/"decision"
    dest.mkdir()
    sel=select(tmp_path,dest)
    expected="trial" if available else "original"
    assert sel["selected"]==expected
    assert (dest/"final.csv").read_bytes()==expected.encode()
    seal(dest,"decision",{"model":"test"},{"original":"original","trial":"trial"})
    assert verify(dest,"decision",{"model":"test"},{"original":"original","trial":"trial"})
    (dest/"final.csv").write_bytes(b"tamper")
    with pytest.raises(ValueError): verify(dest,"decision",{"model":"test"},{"original":"original","trial":"trial"})


def test_real_parent_reproduces_only_frozen_proposal():
    p=read(Path("reports/phase4f/plan.json"))
    path=Path(p["parent"]+".db")
    prop,mask=partition(path,p["parent"],sorted(p["image_hashes"]))
    assert list(map(list,prop["sides"]))==p["sides"] and mask==sorted(p["forbidden_ids"])
    with pytest.raises(ValueError):partition(path,"0"*64,NAMES)


def test_embedded_kit_and_stage_programs(tmp_path,monkeypatch):
    import src.novel_model.final_candidate_v1.session as session
    nb=read(Path("notebooks/step04i_final_candidate_colab.ipynb"))
    for c in nb["cells"]: ast.parse("".join(c["source"]))
    encoded=ast.literal_eval(ast.parse("".join(nb["cells"][3]["source"])).body[0].value)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as z:z.extractall(tmp_path)
    for name,pin in read(tmp_path/"inference_pins.json").items():assert file_sha(tmp_path/name)==pin
    monkeypatch.setattr(session,"KIT",tmp_path)
    for ds in session.POLICY["development"]:
        for stage in ("capture","original","trial","decision"):
            cmd=stage_command(stage,tmp_path,ds,tmp_path,tmp_path)
            code=Path(cmd[3]).read_text()
            ast.parse(code)
            if stage=="decision":
                assert "pip','freeze" in code and "assets" not in code
            else:
                assert "clean" in code and "CLEAN_VM_PRECHECK_PASS" in code
            if stage!="capture": assert "pipeline.run(frame" not in code
    assert aggregate(tmp_path)["completed"]==[]


def test_untrusted_resume(tmp_path):
    p=tmp_path/"bad.zip"
    with zipfile.ZipFile(p,"w") as z:z.writestr("../bad","bad")
    for pin in ("0"*64,file_sha(p)):
        with pytest.raises(ValueError):unpack(p,tmp_path/"out",pin)


def test_sample_restore(tmp_path):
    row=dict(zip(COLUMNS,["0007","d","x","a","1;0;0;0;1;0;0;0;1","0;0;0"]))
    p=tmp_path/"sample.csv"
    with p.open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=COLUMNS)
        w.writeheader();w.writerow(row)
    out=tmp_path/"submission.csv"
    write_submission(p,[dict(row,image_id="8")],out)
    with out.open() as stream:assert list(csv.DictReader(stream))==[row]
    with pytest.raises(ValueError):validate([dict(row,scene="outliers")])
    with pytest.raises(ValueError):validate([dict(row,rotation_matrix="0;0;0;0;0;0;0;0;0")])

def test_graph_to_indexed_exports_to_sealed_choice(tmp_path):
    """Synthetic measurement protocol: same keypoint rows and two different local landmark ID spaces."""
    import sqlite3
    from types import SimpleNamespace as NS
    from src.novel_model.final_candidate_v1 import export_observations
    from src.novel_model.geometry_capture_v1.database import MAX_ID
    db=tmp_path/"synthetic.db"
    xy=np.array([[x,10.] for x in range(40)],dtype="<f4")
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE images(image_id integer,name text,camera_id integer)")
        con.execute("CREATE TABLE keypoints(image_id integer,rows integer,cols integer,data blob)")
        con.execute("CREATE TABLE two_view_geometries(pair_id integer,rows integer,cols integer,data blob)")
        for i,name in enumerate(NAMES,1):
            con.execute("INSERT INTO images VALUES(?,?,1)",(i,name))
            con.execute("INSERT INTO keypoints VALUES(?,40,2,?)",(i,xy.tobytes()))
            for j in range(1,i):
                inliers=40 if (i<=6)==(j<=6) else int((i,j)==(7,6))
                if inliers:con.execute("INSERT INTO two_view_geometries VALUES(?,?,2,?)",
                                       (j*MAX_ID+i,inliers,bytes(inliers*8)))
    proposal,mask=partition(db,file_sha(db),NAMES)
    assert list(map(list,proposal["sides"]))==SIDES and len(mask)==36
    class Identity:
        def __mul__(self,v):return v
    def model(group,offset,shift):
        feats={}
        landmarks={}
        for k,pt in enumerate(xy):
            refs=[NS(image_id=NAMES.index(n)+1,point2D_idx=k) for n in group]
            landmarks[k+offset]=NS(xyz=np.array([pt[0]+shift,pt[1],1.]),track=NS(elements=refs))
        for n in group:
            feats[NAMES.index(n)+1]=NS(name=n,camera_id=1,cam_from_world=Identity(),
                points2D=[NS(xy=p,point3D_id=k+offset,has_point3D=lambda:True) for k,p in enumerate(xy)])
        return NS(reg_image_ids=lambda:list(feats),images=feats,points3D=landmarks,
                  cameras={1:NS(img_from_cam=lambda p:p)})
    for stage,models in (("original",{0:model(NAMES,100,1.)}),
                         ("trial",{0:model(SIDES[0],500,0.),1:model(SIDES[1],1000,0.)})):
        folder=tmp_path/stage
        folder.mkdir()
        groups,err=export_observations(models,db)
        np.savez(folder/"errors.npz",**err)
        save(folder/"branch.json",dict(available=True,groups=groups,valid_export=True,proposal=proposal))
        (folder/"prediction.csv").write_bytes(stage.encode())
    dst=tmp_path/"decision";dst.mkdir()
    assert select(tmp_path,dst)["action"]=="ACCEPT"
    assert (dst/"final.csv").read_bytes()==b"trial"
    broken=model(NAMES,7,0.)
    broken.points3D[7].track.elements.pop()
    with pytest.raises(ValueError,match="backlink"):export_observations({0:broken},db)


def test_administrative_failure_kept_out_of_score_mean(tmp_path):
    from scripts.phase4i_evaluate import record_failure,FREEZE
    f=read(FREEZE)
    ds=f["order"][0]
    ident=dict(dataset=ds,model=f["version"],worker=f["datasets"][ds]["worker_sha256"],
               seed=20261111,protocol=f["protocol"])
    archive=tmp_path/"failure.zip"
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("failure.json",json.dumps(dict(stage="capture",identity=ident,error="resource limit")))
    record_failure(ds,archive,file_sha(archive),tmp_path)
    summary=aggregate(tmp_path)
    assert summary["equal_group_mean"]=={} and len(summary["failures"])==1 and ds in summary["missing"]
    with pytest.raises(ValueError):record_failure(f["order"][1],archive,file_sha(archive),tmp_path)
