# ruff: noqa: E402 -- direct script entry points resolve the project root first.
"""Administrative, read-only inspection. Only neutral image IDs and verified endpoints enter the graph."""

import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.novel_model.geometry_graph_v1.graph_import import import_bytes
from src.novel_model.geometry_graph_v1.schemas import (
    CONVENTION,
    digest,
    encoded,
    policy,
)

REPORT = ROOT / "reports/phase4b"
EXPECTED = {
    "ETs": 22,
    "stairs": 51,
    "imc2023_haiper": 54,
    "imc2023_theather_imc2024_church": 76,
}
GEOMETRY_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".h5", ".hdf5", ".npz"}


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def other_candidates(root):
    """Inspect filenames/ZIP central directories, never extract or ingest foreign caches."""
    found = []
    for path in root.rglob("*"):
        if not path.is_file() or REPORT in path.parents:
            continue
        rel = path.relative_to(root).as_posix()
        if path.suffix.lower() in GEOMETRY_SUFFIXES:
            found.append(
                {"file": rel, "members": [path.name], "sha256": file_hash(path)}
            )
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                members = [
                    m
                    for m in archive.namelist()
                    if Path(m).suffix.lower() in GEOMETRY_SUFFIXES
                    or Path(m).name
                    in {
                        "points3D.bin",
                        "images.bin",
                        "cameras.bin",
                        "verified_edges.json",
                    }
                ]
            if members:
                found.append(
                    {"file": rel, "members": members, "sha256": file_hash(path)}
                )
    return sorted(found, key=lambda x: x["file"])


def inspect(row, cfg, lock):
    dataset = row["dataset"]
    path = ROOT / row["archive"]
    if file_hash(path) != row["sha256"]:
        raise ValueError("worker archive hash mismatch")
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if len(set(members)) != len(members):
            raise ValueError("duplicate archive member")
        member_hashes = {name: digest(archive.read(name)) for name in members}
        identity = json.loads(archive.read("identity.json"))
        run = json.loads(archive.read("run.json"))
        if (
            run["status"] != "PASS"
            or run["worker_sha256"] != row["worker_sha256"]
            or identity["config_sha256"] != lock["configuration_sha256"]
            or identity["experiment_id"] != f"ns64_isolated_v1_{dataset}_20261111"
        ):
            raise ValueError("incompatible experiment")
        # Project exactly one column on administrator side; poses and assignments are discarded.
        names = sorted(
            r["image"]
            for r in csv.DictReader(io.StringIO(archive.read("opaque.csv").decode()))
        )
        if names != [f"image_{i:04}.png" for i in range(EXPECTED[dataset])]:
            raise ValueError("opaque node set mismatch")
        summary = next(
            s
            for line in archive.read("stages.jsonl").splitlines()
            if (s := json.loads(line))["stage"] == "verify_matches"
            and s.get("status") == "PASS"
        )
        has_pairs = "verified_edges.json" in members
        provenance = {
            "source_sha256": lock["source_sha256"],
            "config_sha256": identity["config_sha256"],
            "parent_sha256": row["sha256"],
            "membership_sha256": digest(encoded(names)),
        }
        topology = None
        if has_pairs:
            pairs = json.loads(archive.read("verified_edges.json"))
            if len(pairs) != summary["verified_pairs"]:
                raise ValueError("edge inventory and receipt differ")
            obj = {
                "schema": "geometry_graph_v1",
                "mode": "topology_only",
                "convention": CONVENTION,
                "provenance": provenance,
                "nodes": names,
                "edges": [
                    {"i": i, "j": j, "inliers": None, "rotation_valid": False}
                    for i, j in pairs
                ],
            }
            data = encoded(obj)
            trust = {**provenance, "payload_sha256": digest(data)}
            graph = import_bytes(data, trust, cfg)
            topology = {
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "trust": trust,
            }
            (REPORT / f"{dataset}_topology.json").write_bytes(data)
    return {
        **row,
        "identity": identity,
        "member_hashes": member_hashes,
        "node_count": len(names),
        "reported_verified_pairs": summary["verified_pairs"],
        "topology_import": topology,
        "fields": {
            "verified_pairs": "AVAILABLE" if has_pairs else "PARTIALLY AVAILABLE",
            "opaque_image_ids": "AVAILABLE",
            "database_camera_ids": "MISSING",
            "verified_inlier_counts": "MISSING",
            "keypoints": "MISSING",
            "inlier_indices": "MISSING",
            "calibrated_intrinsics": "MISSING",
            "calibrated_relative_rotations": "MISSING",
            "mapper_database": "MISSING",
            "reconstruction": "PARTIALLY AVAILABLE",
        },
    }


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    cfg = policy()
    pins = cfg["reference_file_sha256"]
    for path, expected in pins.items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"changed reference {path}")
    lock = json.loads((ROOT / cfg["benchmark"]["ns64_lock"]).read_bytes())
    index = json.loads((ROOT / cfg["benchmark"]["subset_receipts"]).read_bytes())
    rows = [inspect(row, cfg, lock) for row in index]
    candidates = other_candidates(ROOT)
    result = {
        "schema": "ns64_cache_inventory_v1",
        "source_revision": rows[0]["identity"]["source_revision"],
        "archive_index_sha256": pins[cfg["benchmark"]["subset_receipts"]],
        "four": rows,
        "workspace_geometry_file_candidates": candidates,
        "scan_scope": "Loose geometry files and immediate ZIP members; nested handoff ZIPs are not trusted stage parents.",
    }
    (REPORT / "cache_inventory.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
