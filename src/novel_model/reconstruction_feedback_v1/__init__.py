"""Transactional mapper-only copies. Never open the parent for writing."""
from contextlib import closing
from pathlib import Path
import sqlite3
from ..geometry_capture_v1.database import MAX_ID, file_sha, logical, snapshot, read_only

PAIR_TABLES = ("matches", "two_view_geometries")


def inspect(db):
    records = logical(db)
    counts = {t:db.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in records["tables"]}
    return dict(records=records, counts=counts)


def make_copy(parent, target, digest, forbidden=()):
    if file_sha(parent) != digest:
        raise ValueError("numerical parent hash mismatch")
    snapshot(parent,target)
    ids = sorted(set(forbidden))
    with closing(read_only(parent)) as src, closing(sqlite3.connect(target)) as dst:
        before = inspect(src)
        for t in before["counts"]:
            cols = {r[1] for r in src.execute(f'PRAGMA table_info("{t}")')}
            if "pair_id" in cols and t not in PAIR_TABLES:
                raise ValueError("unexpected pair source")
        with dst:
            for t in PAIR_TABLES:
                dst.executemany(f"DELETE FROM {t} WHERE pair_id=?",[(v,) for v in ids])
        after = inspect(dst)
        for t in before["counts"]:
            if t not in PAIR_TABLES:
                if before["records"]["tables"][t] != after["records"]["tables"][t]:
                    raise ValueError("non-pair records changed")
                continue
            rows = iter(dst.execute(f"SELECT * FROM {t} ORDER BY pair_id"))
            for row in src.execute(f"SELECT * FROM {t} ORDER BY pair_id"):
                if row["pair_id"] not in ids and tuple(row) != next(rows,None):
                    raise ValueError("retained pair mismatch")
            if next(rows,None) is not None:
                raise ValueError("extra pair")
        if dst.execute("PRAGMA foreign_key_check").fetchone() or dst.execute("PRAGMA integrity_check").fetchone()!=("ok",):
            raise ValueError("SQLite integrity")
    if file_sha(parent)!=digest:
        raise ValueError("parent changed")
    return dict(before=before,after=after,forbidden=ids)
