"""Restore sample identities and validate competition poses before atomic output."""
import csv
import os
from pathlib import Path
import tempfile
import numpy as np

COLUMNS = ["image_id","dataset","scene","image","rotation_matrix","translation_vector"]


def validate(rows):
    keys,ids = set(),set()
    for r in rows:
        key = (r["dataset"],r["image"])
        if (set(r)!=set(COLUMNS) or key in keys or not all(key) or not r["scene"] or
            not r["image_id"] or r["image_id"] in ids):
            raise ValueError("duplicate or incomplete submission identity")
        keys.add(key)
        ids.add(r["image_id"])
        rot, trans = (np.array([float(x) for x in r[c].split(";")]) for c in COLUMNS[-2:])
        if rot.size != 9 or trans.size != 3:
            raise ValueError("pose dimensions")
        if r["scene"] == "outliers":
            if not (np.isnan(rot).all() and np.isnan(trans).all()):
                raise ValueError("outlier pose")
        else:
            R = rot.reshape(3,3)
            if (not np.isfinite(R).all() or not np.isfinite(trans).all()
                or not np.allclose(R@R.T,np.eye(3),atol=1e-4) or abs(np.linalg.det(R)-1)>1e-4):
                raise ValueError("invalid registered pose")
    if not keys:
        raise ValueError("empty submission")


def write_submission(sample, predictions, target):
    with Path(sample).open(newline="",encoding="utf-8-sig") as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise ValueError("sample header")
        expected=list(reader)
    by_key={(r["dataset"],r["image"]):r for r in predictions}
    if len(by_key)!=len(predictions) or len(expected)!=len(by_key):
        raise ValueError("prediction coverage")
    rows=[]
    for row in expected:
        pred=by_key[row["dataset"],row["image"]]
        rows.append({k:(row[k] if k in ("image_id","dataset","image") else pred[k]) for k in COLUMNS})
    validate(rows)
    with tempfile.NamedTemporaryFile(mode="w",newline="",encoding="utf8",dir=Path(target).parent,
                                     delete=False) as stream:
        writer=csv.DictWriter(stream,fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(stream.name,target)
