"""Convert a verified numerical parent; scoring and private identities live elsewhere."""

import json
from collections import Counter
from pathlib import Path
from .database import inventory, file_sha, save
from .rotations import recover
from ..geometry_graph_v1.graph_import import import_bytes
from ..geometry_graph_v1.schemas import encoded, digest, CONVENTION, METHODS, policy
from ..geometry_graph_v1.decision_engine import decide
from ..geometry_graph_v1.diagnostics import checked_keep_mask


def convert(path, manifest, output):
    path = Path(path)
    output = Path(output)
    output.mkdir(exist_ok=True)
    if file_sha(path) != manifest["backup_sha256"]:
        raise ValueError("numerical parent hash mismatch")
    cfg = policy()
    names = sorted(manifest["image_sha256"])
    images, cameras, keys, pairs, arrays = inventory(
        path, names, {int(i): v for i, v in manifest["offset_sha256"].items()}
    )
    if arrays != manifest["arrays"]:
        raise ValueError("array inventory mismatch")
    rows = []
    details = []
    reasons = Counter()
    for (i, j), p in sorted(pairs.items()):
        if "verified" not in p or not len(p["verified"]):
            continue
        a, b = images[i], images[j]
        ij = p["verified"]
        cal = manifest["focal_quality"]
        r, detail = recover(
            p,
            keys[i][ij[:, 0]],
            keys[j][ij[:, 1]],
            cameras[a["camera_id"]],
            cameras[b["camera_id"]],
            cal[a["name"]],
            cal[b["name"]],
            cfg["rotation"],
        )
        rows.append(
            dict(
                i=a["name"],
                j=b["name"],
                inliers=len(ij),
                rotation_valid=r is not None,
                rotation=None if r is None else r.tolist(),
            )
        )
        details.append(dict(i=a["name"], j=b["name"], **detail))
        reasons[detail.get("reason", "usable")] += 1
    provenance = dict(
        source_sha256=manifest["source_inventory_hash"],
        config_sha256=manifest["config_sha256"],
        parent_sha256=file_sha(path),
        membership_sha256=digest(encoded(names)),
    )
    obj = dict(
        schema="geometry_graph_v1",
        nodes=names,
        edges=rows,
        mode="verified_counts",
        convention=CONVENTION,
        provenance=provenance,
    )
    data = encoded(obj)
    trust = {**provenance, "payload_sha256": digest(data)}
    graph = import_bytes(data, trust, cfg)
    (output / "graph.json").write_bytes(data)
    save(output / "graph_trust.json", trust)
    summaries = {}
    for method in METHODS:
        out = decide(graph, method, cfg)
        keep = checked_keep_mask(out, graph, cfg)
        if method == "pass_through_v1" and not all(keep):
            raise ValueError("nonidentity pass-through")
        save(output / (method + ".json"), out)
        summaries[method] = dict(
            decisions=[r["decision"] for r in out["components"]],
            removed=keep.count(False),
            mask_sha256=out["mask_sha256"],
        )
        if method == "candidate_a_v1":
            components = out["components"]
    sides = [s for c in components for s in c.get("internal", [])]
    stats = dict(
        images=len(names),
        verified_pairs=len(rows),
        pairs_with_support=len(rows),
        usable_rotations=reasons["usable"],
        components=len(components),
        reasons=dict(reasons),
        reliable_sides=sum(s["reason"] == "reliable" for s in sides),
        observed_non_tree_checks=sum(len(s.get("non_tree_checks", [])) for s in sides),
        crossing_witnesses_by_component=[
            len(c["boundary"]["witnesses"]) if "boundary" in c else None
            for c in components
        ],
        methods=summaries,
    )
    save(output / "rotations.json", details)
    save(output / "coverage.json", stats)
    return stats
