import sqlite3
from contextlib import closing
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from src.novel_model.geometry_capture_v1.database import (
    snapshot,
    inventory,
    MAX_ID,
    sha,
    file_sha,
    pair_ids,
)
from src.novel_model.geometry_capture_v1.hook import replay_copy
from src.novel_model.geometry_capture_v1.rotations import recover
from src.novel_model.geometry_capture_v1.analysis import convert
from src.novel_model.geometry_graph_v1.schemas import policy


@pytest.fixture
def parent(tmp_path):
    p = tmp_path / "live.db"
    names = [f"image_{i:04}.png" for i in range(54)]
    with closing(sqlite3.connect(p)) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""CREATE TABLE cameras(camera_id,model,width,height,params,prior_focal_length);
        CREATE TABLE images(image_id,name,camera_id);CREATE TABLE keypoints(image_id,rows,cols,data);
        CREATE TABLE matches(pair_id,rows,cols,data);CREATE TABLE two_view_geometries(pair_id,rows,cols,data,config);""")
        db.execute(
            "INSERT INTO cameras VALUES(7,0,800,600,?,1)",
            (np.array([500.0, 400.0, 300.0]).tobytes(),),
        )
        xy = np.array([[10.5, 11.5]], "<f4").tobytes()
        for n, name in enumerate(reversed(names), 1):
            db.execute("INSERT INTO images VALUES(?,?,7)", (n, name))
            db.execute("INSERT INTO keypoints VALUES(?,1,2,?)", (n, xy))
        db.execute("INSERT INTO matches VALUES(?,1,2,?)", (MAX_ID + 2, bytes(8)))
        db.execute(
            "INSERT INTO two_view_geometries VALUES(?,1,2,?,2)", (MAX_ID + 2, bytes(8))
        )
        db.commit()
        dst = tmp_path / "snap.db"
        m = snapshot(p, dst)
        assert m["source_wal_file_sha256"]
    m.update(
        offset_sha256={i: sha(xy) for i in range(1, 55)},
        source_inventory_hash="a" * 64,
        config_sha256="b" * 64,
        image_sha256=dict.fromkeys(names, "c" * 64),
        focal_quality=dict.fromkeys(names, False),
    )
    m["arrays"] = inventory(dst, names)[-1]
    return dst, m, names


def test_capture_replay(parent, tmp_path):
    db, m, names = parent
    ims, cams, keys, pairs, arrays = inventory(db, names, m["offset_sha256"])
    assert (
        ims[1] == dict(name=names[-1], camera_id=7)
        and pairs[1, 2]["verified"].dtype == np.uint32
    )
    s = convert(db, m, tmp_path / "reports")
    assert s["images"] == 54 and s["verified_pairs"] == 1 and s["usable_rotations"] == 0
    assert (
        replay_copy(
            db, tmp_path / "copy.db", m, dict(method="pass_through_v1", keep=[True])
        )
        == m["original_logical"]
    )
    with pytest.raises(ValueError):
        replay_copy(
            db, tmp_path / "bad.db", m, dict(method="candidate_a_v1", keep=[False])
        )


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE keypoints SET rows=2",
        "UPDATE images SET camera_id=8",
        "UPDATE two_view_geometries SET data=x'0100000000000000'",
        "UPDATE two_view_geometries SET rows=9",
        "DELETE FROM images WHERE image_id=54",
    ],
)
def test_corrupt(parent, sql):
    db, m, names = parent
    with closing(sqlite3.connect(db)) as conn:
        conn.execute(sql)
        conn.commit()
    with pytest.raises(ValueError):
        inventory(db, names)
    assert file_sha(db) != m["backup_sha256"]


def test_offset(parent):
    db, m, names = parent
    with pytest.raises(ValueError):
        inventory(db, names, {})
    with pytest.raises(ValueError):
        pair_ids(MAX_ID + 1)


@pytest.mark.parametrize("reverse", [True, False])
def test_rotation(reverse):
    x = np.random.default_rng(6).uniform(-1, 1, (100, 3))
    x[:, 2] += 4
    r = Rotation.from_euler("y", 15, degrees=True).as_matrix()
    y = x @ r.T + [0.7, 0.1, 0]
    cam = dict(
        model="PINHOLE", params=[500.0, 500.0, 400.0, 300.0], width=800, height=600
    )
    proj = lambda v: 500 * v[:, :2] / v[:, 2:] + [400, 300]
    a, b = proj(x), proj(y)
    rot, detail = recover(
        dict(config=2),
        b if reverse else a,
        a if reverse else b,
        cam,
        cam,
        True,
        True,
        policy()["rotation"],
    )
    assert np.allclose(rot, r.T if reverse else r, atol=1e-4), detail
    assert (
        recover(dict(config=4), a, b, cam, cam, True, True, policy()["rotation"])[0]
        is None
    )
    assert (
        recover(
            dict(config=2),
            a,
            b,
            dict(cam, model="OTHER"),
            cam,
            True,
            True,
            policy()["rotation"],
        )[0]
        is None
    )


def test_trusted_parent_required(parent, tmp_path):
    db, m, names = parent
    m["backup_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        convert(db, m, tmp_path / "out")

@pytest.mark.parametrize("model,params",[("SIMPLE_PINHOLE",[500,400,300]),("PINHOLE",[500,510,400,300]),("SIMPLE_RADIAL",[500,400,300,0.01]),("RADIAL",[500,400,300,0.01,0.001]),("OPENCV",[500,510,400,300,0.01,0,0.001,0])])
def test_camera_conventions(model,params):
    from src.novel_model.geometry_capture_v1.rotations import normalized
    rays,_=normalized(np.array([[400.,300.],[410.,305.]]),dict(model=model,params=params,width=800,height=600))
    assert np.allclose(rays[0],0) and np.all(rays[1]>0)

def test_failure_receipt_and_worker_isolation(tmp_path):
    import subprocess,sys,json,zipfile,ast,base64,io
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    out=tmp_path/'result'
    p=subprocess.run([sys.executable,str(root/'scripts/validate_ns64_geometry.py'),str(tmp_path/'missing.zip'),str(tmp_path/'missing.db'),str(out)],capture_output=True)
    assert p.returncode and json.loads((out/'failure.json').read_bytes())['stage']=='validate_capture'
    nb=json.loads((root/'notebooks/step04c_haiper_capture_colab.ipynb').read_bytes())
    code=ast.parse(''.join(nb['cells'][-1]['source']))
    z=zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(code.body[1].value))))
    assert not any(n.endswith(('.csv','.db')) or 'private.json' in n or 'compare_ns64_mapper' in n for n in z.namelist())

def test_pose_parity_detects_camera_drift():
    from src.novel_model.geometry_capture_v1.parity import compare
    import csv,io
    def payload(drift):
        out=io.StringIO()
        writer=csv.writer(out);writer.writerow(['dataset','scene','image','rotation_matrix','translation_vector'])
        for i in range(54):
            writer.writerow(['imc2023_haiper','one',f'image_{i:04}.png','1;0;0;0;1;0;0;0;1',f'{i};{drift if i==7 else 0};0'])
        return out.getvalue().encode()
    assert compare(payload(0),payload(0))['pose_equivalent']
    assert not compare(payload(0),payload(1))['pose_equivalent']
