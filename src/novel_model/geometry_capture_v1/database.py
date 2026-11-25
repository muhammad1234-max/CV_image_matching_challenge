"""Read and fingerprint a COLMAP database in one pinned SQLite read transaction."""

import base64
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
import numpy as np

MAX_ID = 2**31 - 1
MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
}
OFFSET = "colmap_xy_at_pixel_centers_plus_0.5_once"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, obj):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, sort_keys=True, indent=2, allow_nan=False), encoding="utf8"
    )
    tmp.replace(p)


def read_only(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    return db


def pair_ids(value):
    a, b = divmod(value, MAX_ID)
    if not 0 < a < b < MAX_ID:
        raise ValueError("invalid encoded pair")
    return a, b


def logical(db):
    """Include schema, every table/row/column and blob, independent of page layout."""
    master = list(
        db.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
        )
    )
    tables = {}
    for kind, name, _, _ in master:
        if kind != "table":
            continue
        quoted = '"' + name.replace('"', '""') + '"'
        h = hashlib.sha256()
        for row in db.execute(f"SELECT * FROM {quoted} ORDER BY rowid"):
            values = [
                {"blob_sha256": sha(x), "bytes": len(x)} if isinstance(x, bytes) else x
                for x in row
            ]
            h.update(json.dumps(values, separators=(",", ":")).encode() + b"\n")
        tables[name] = h.hexdigest()
    return {
        "schema": sha(json.dumps([list(r) for r in master]).encode()),
        "tables": tables,
    }


def snapshot(source, target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    partial = target.with_suffix(".partial")
    if partial.exists():
        raise FileExistsError(partial)
    with closing(read_only(source)) as db, closing(sqlite3.connect(partial)) as dst:
        db.execute("BEGIN")
        original = logical(db)
        db.backup(dst)
        if dst.execute("PRAGMA integrity_check").fetchone()[
            0
        ] != "ok" or original != logical(dst):
            raise ValueError("SQLite backup record mismatch")
    partial.rename(target)
    return {
        "original_logical": original,
        "backup_sha256": file_sha(target),
        "source_main_file_sha256": file_sha(source),
        "source_wal_file_sha256": file_sha(str(source) + "-wal")
        if Path(str(source) + "-wal").is_file()
        else None,
    }


def array(blob, dtype, shape):
    blob = b"" if blob is None else blob
    if (
        any(v < 0 for v in shape)
        or len(blob) != int(np.prod(shape)) * np.dtype(dtype).itemsize
    ):
        raise ValueError("array byte count mismatch")
    v = np.frombuffer(blob, dtype=dtype).reshape(shape)
    if not np.isfinite(v).all():
        raise ValueError("nonfinite database array")
    return v


def inventory(path, names, offset_pins=None):
    """Yield logical numerical arrays. The SQLite snapshot is their lossless storage."""
    names = list(names)
    if len(set(names)) != len(names):
        raise ValueError("duplicate expected image")
    arrays = {}

    def take(key, blob, dtype, shape):
        v = array(blob, dtype, shape)
        arrays[key] = {"sha256": sha(v.tobytes()), "shape": list(shape), "dtype": dtype}
        return v

    with closing(read_only(path)) as db:
        cameras = {}
        for c in db.execute("SELECT * FROM cameras"):
            cid = c["camera_id"]
            model, size = MODELS.get(
                c["model"], (f"unsupported_{c['model']}", len(c["params"]) // 8)
            )
            params = take(f"cameras/{cid}/params", c["params"], "<f8", (size,))
            if min(c["width"], c["height"]) <= 0:
                raise ValueError("invalid dimensions")
            cameras[cid] = dict(
                model=model,
                width=c["width"],
                height=c["height"],
                params=params.tolist(),
                prior_focal_length=c["prior_focal_length"],
            )
        images = {
            r["image_id"]: dict(name=r["name"], camera_id=r["camera_id"])
            for r in db.execute("SELECT * FROM images")
        }
        if sorted(v["name"] for v in images.values()) != sorted(names):
            raise ValueError("database membership mismatch")
        if any(
            not 0 < i < MAX_ID or v["camera_id"] not in cameras
            for i, v in images.items()
        ):
            raise ValueError("image/camera association")
        keys = {}
        for r in db.execute("SELECT * FROM keypoints"):
            i = r["image_id"]
            if i not in images or r["cols"] not in (2, 4, 6):
                raise ValueError("keypoint identity or layout")
            keys[i] = take(f"keypoints/{i}", r["data"], "<f4", (r["rows"], r["cols"]))
            if offset_pins is not None and sha(
                keys[i][:, :2].tobytes()
            ) != offset_pins.get(i):
                raise ValueError("pixel-center offset mismatch")
        if set(keys) != set(images):
            raise ValueError("missing keypoint row")
        pairs = {}
        for table in ("matches", "two_view_geometries"):
            for r in db.execute(f"SELECT * FROM {table} ORDER BY pair_id"):
                i, j = pair_ids(r["pair_id"])
                if i not in keys or j not in keys or r["cols"] != 2:
                    raise ValueError("pair endpoints or index layout")
                ij = take(f"{table}/{i}-{j}/indices", r["data"], "<u4", (r["rows"], 2))
                if len(ij) and (
                    ij[:, 0].max() >= len(keys[i]) or ij[:, 1].max() >= len(keys[j])
                ):
                    raise ValueError("keypoint index out of bounds")
                p = pairs.setdefault((i, j), {})
                if table == "matches":
                    p["raw"] = ij
                    continue
                p.update(verified=ij, config=r["config"])
                for field, shape in (
                    ("F", (3, 3)),
                    ("E", (3, 3)),
                    ("H", (3, 3)),
                    ("qvec", (4,)),
                    ("tvec", (3,)),
                ):
                    if field in r.keys() and r[field]:
                        p[field] = take(
                            f"{table}/{i}-{j}/{field}", r[field], "<f8", shape
                        )
                if len(ij) and "raw" not in p:
                    raise ValueError("verified pair missing raw record")
    return images, cameras, keys, pairs, arrays
