"""Compare assignments and world-to-camera poses without using labels."""

import csv
import io
from collections import Counter
import numpy as np
from ..geometry_graph_v1.rotation_geometry import degrees, checked_rotation


def parse(data):
    records = {}
    for row in csv.DictReader(io.StringIO(data.decode("utf8"))):
        n = row["image"]
        scene = row["scene"]
        if n in records or row["dataset"] != "imc2023_haiper" or not scene:
            raise ValueError("invalid prediction identity")
        r = np.array([float(x) for x in row["rotation_matrix"].split(";")]).reshape(
            3, 3
        )
        t = np.array([float(x) for x in row["translation_vector"].split(";")]).reshape(
            3
        )
        if scene == "outliers":
            if not (np.isnan(r).all() and np.isnan(t).all()):
                raise ValueError("invalid outlier pose")
        else:
            checked_rotation(r, 1e-5)
            if not np.isfinite(t).all():
                raise ValueError("invalid translation")
        records[n] = (scene, r, -r.T @ t)
    if set(records) != {f"image_{i:04}.png" for i in range(54)}:
        raise ValueError("incomplete opaque predictions")
    return records


def compare(left, right):
    a, b = parse(left), parse(right)
    names = sorted(a)

    def groups(d):
        return sorted(
            sorted(n for n in names if d[n][0] == c)
            for c in set(v[0] for v in d.values())
            if c != "outliers"
        )

    ga, gb = groups(a), groups(b)
    equal = ga == gb
    stats = []
    for group in ga if equal else []:
        x = np.stack([a[n][2] for n in group])
        y = np.stack([b[n][2] for n in group])
        ra = np.stack([a[n][1] for n in group])
        rb = np.stack([b[n][1] for n in group])
        raw_ang = degrees(rb @ ra.transpose(0, 2, 1))
        # Alignment is independent for each model gauge. Orientations constrain degenerate camera layouts.
        u, _, vt = np.linalg.svd((rb.transpose(0, 2, 1) @ ra).sum(axis=0))
        Q = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        xc = (x - x.mean(0)) @ Q.T
        yc = y - y.mean(0)
        scale = (
            float(np.sum(xc * yc) / np.sum(xc * xc)) if np.sum(xc * xc) > 1e-20 else 1.0
        )
        denom = float(np.sqrt(np.mean(np.sum(yc * yc, axis=1))))
        err = np.linalg.norm(scale * xc - yc, axis=1)
        stats.append(
            dict(
                cameras=len(group),
                max_raw_rotation_deg=float(raw_ang.max()),
                max_raw_center_delta=float(np.linalg.norm(x - y, axis=1).max()),
                scale=scale,
                max_aligned_rotation_deg=float(
                    degrees(rb @ Q @ ra.transpose(0, 2, 1)).max()
                ),
                max_aligned_center_fraction=float(err.max() / max(denom, 1e-12)),
            )
        )
    poses = (
        equal
        and bool(stats)
        and all(
            s["scale"] > 0
            and s["max_aligned_rotation_deg"] <= 0.01
            and s["max_aligned_center_fraction"] <= 1e-4
            for s in stats
        )
    )
    return dict(
        same_assignments_and_outliers=equal,
        pose_equivalent=poses,
        per_model=stats,
        original_registered=sum(map(len, ga)),
        pass_through_registered=sum(map(len, gb)),
        original_clusters=len(ga),
        pass_through_clusters=len(gb),
    )
