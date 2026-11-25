import sqlite3
from contextlib import closing
import pytest
from src.novel_model.reconstruction_feedback_v1 import make_copy, file_sha, MAX_ID

@pytest.mark.parametrize("active",[False,True])
def test_transactional_mask(tmp_path,active):
    parent=tmp_path/"parent.db"
    pair=MAX_ID+2
    with closing(sqlite3.connect(parent)) as db:
        for t in ("images","cameras","keypoints"):
            db.execute(f"create table {t}(id integer primary key,data blob)")
            db.execute(f"insert into {t} values(1,X'01')")
        for t in ("matches","two_view_geometries"):
            db.execute(f"create table {t}(pair_id integer primary key,data blob)")
            db.executemany(f"insert into {t} values(?,?)",[(pair,b"x"),(pair+1,b"y")])
        db.commit()
    pin=file_sha(parent)
    target=tmp_path/"branch.db"
    with pytest.raises(ValueError):
        make_copy(parent,target,"bad")
    audit=make_copy(parent,target,pin,[pair] if active else [])
    assert file_sha(parent)==pin
    with closing(sqlite3.connect(target)) as db:
        for t in ("matches","two_view_geometries"):
            assert list(db.execute(f"select * from {t} order by pair_id"))==([(pair+1,b"y")] if active else [(pair,b"x"),(pair+1,b"y")])
    assert audit["before"]["records"]["tables"]["images"]==audit["after"]["records"]["tables"]["images"]
    with pytest.raises(FileExistsError):
        make_copy(parent,target,pin)
