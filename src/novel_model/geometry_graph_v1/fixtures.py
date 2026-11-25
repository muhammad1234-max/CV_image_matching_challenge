"""Deterministic synthetic observations. Generating group labels never enter graph payloads."""

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from .schemas import CONVENTION, digest, encoded
from .graph_import import import_bytes


def manifest(obj):
    return {**obj["provenance"], "payload_sha256": digest(encoded(obj))}


def refresh_membership(obj):
    obj["provenance"]["membership_sha256"] = digest(encoded(sorted(obj["nodes"])))
    return obj


def fixture(kind="rigid", n=16, seed=17):
    if n < 12:
        raise ValueError("fixture needs at least twelve cameras")
    rng = np.random.default_rng(seed)
    cams = Rotation.random(n, random_state=rng).as_matrix()
    names = [f"image_{i:04}.png" for i in range(n)]
    mid = n // 2
    rows = []

    def add(i, j, count, twist=None):
        relative = cams[j] @ (np.eye(3) if twist is None else twist) @ cams[i].T
        rows.append(
            {
                "i": names[i],
                "j": names[j],
                "inliers": count,
                "rotation_valid": True,
                "rotation": relative.tolist(),
            }
        )

    if kind in ("spectral_tie", "near_tie"):
        for j in range(n):
            for i in range(j):
                add(i, j, 10**10 + int(kind == "near_tie" and i + j == 1))
    else:
        for start, end in ((0, mid), (mid, n)):
            for j in range(start + 1, end):
                for i in range(start, j):
                    if end - start > 15 and i - start >= 4 and j - i > 3:
                        continue
                    r = None
                    if kind == "noisy":
                        r = Rotation.from_rotvec(rng.normal(0, 0.002, 3)).as_matrix()
                    if kind in ("few_loops", "bad_loops") and i != start:
                        if kind == "few_loops" and j - start > 3:
                            continue
                        if kind == "bad_loops":
                            r = Rotation.from_euler("x", 65, degrees=True).as_matrix()
                    add(i, j, 1100 if i == start else 1000, r)
        for t in range(mid):
            if (
                kind == "disconnected"
                or (kind == "sparse" and t > 0)
                or (kind == "few_witnesses" and t >= 5)
            ):
                continue
            i = 0 if kind == "shared_endpoint" else t
            angle = 0
            if kind in ("contradiction", "strong_false", "half"):
                angle = 360 * t / mid if kind != "half" or t >= mid // 2 else 0
            add(
                i,
                mid + t,
                200 if kind == "strong_false" else 4,
                Rotation.from_euler("z", angle, degrees=True).as_matrix(),
            )
            if kind == "coherent_alias":
                add(i, mid + ((t + 1) % mid), 4)
    if kind == "missing":
        for row in rows:
            row.update(rotation_valid=False, rotation=None)
    if kind == "isolated":
        names.append(f"image_{n:04}.png")
    obj = {
        "schema": "geometry_graph_v1",
        "mode": "verified_counts",
        "convention": CONVENTION,
        "nodes": names,
        "edges": rows,
        "provenance": {
            "source_sha256": digest(Path(__file__).read_bytes()),
            "config_sha256": digest(encoded([kind, n, seed])),
            "parent_sha256": digest(b"synthetic-only"),
        },
    }
    return refresh_membership(obj)


def as_graph(obj, cfg=None):
    return import_bytes(encoded(obj), manifest(obj), cfg)
