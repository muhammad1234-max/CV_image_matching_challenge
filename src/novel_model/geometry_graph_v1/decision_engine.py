"""All masks are aligned to the canonical input edge order. True means keep."""

from collections import defaultdict
from time import perf_counter
import numpy as np
from .schemas import METHODS, digest, encoded, policy
from .spectral_proposal import components, propose
from .internal_consistency import internal_frame
from .boundary_consensus import vote
from .rotation_geometry import degrees


def triangle_control(nodes, edges, cfg):
    neighbors = {n: {} for n in nodes}
    for e in edges:
        if e.rotation is not None:
            neighbors[e.i][e.j] = e.rotation
            neighbors[e.j][e.i] = e.rotation.T
    removed = set()
    checks = []
    for e in edges:
        if e.rotation is None:
            continue
        a, b = neighbors[e.i], neighbors[e.j]
        residuals = [
            float(degrees(b[k].T @ a[k] @ e.rotation.T))
            for k in sorted(a.keys() & b.keys())
        ]
        checks.append([e.i, e.j, residuals])
        if (
            len(residuals) >= cfg["triangle_min_checks"]
            and np.median(residuals) > cfg["triangle_median_limit_deg"]
        ):
            removed.add(e.pair)
    return removed, checks


def decide(graph, method=None, config=None, timings=None):
    cfg = config or policy()
    method = cfg["default_method"] if method is None else method
    if method not in METHODS:
        raise ValueError("unknown method")
    times = defaultdict(float)
    removed = set()
    reports = []
    for nodes, edges in components(graph):
        report = {"nodes": nodes}
        reports.append(report)
        if method == "pass_through_v1":
            report.update(decision="RETAIN", reason="pass_through")
            continue
        if method == "triangle_v1":
            bad, checks = triangle_control(nodes, edges, cfg)
            removed.update(bad)
            report.update(
                decision="PRUNE" if bad else "RETAIN",
                reason="local_triangles",
                checks=checks,
            )
            continue
        start = perf_counter()
        cut, info = propose(nodes, edges, cfg["graph"])
        times["proposal"] += perf_counter() - start
        report["proposal"] = info
        if cut is None:
            report.update(decision="ABSTAIN", reason=info["reason"])
            continue
        a, b = cut
        report["sides"] = cut
        left = set(a)
        inside_a, inside_b, cross = [], [], []
        for e in edges:
            if (e.i in left) != (e.j in left):
                cross.append(e)
            elif e.i in left:
                inside_a.append(e)
            else:
                inside_b.append(e)
        if method == "normalized_cut_v1":
            removed.update(e.pair for e in cross)
            report.update(decision="SPLIT", reason="connected_spectral_proposal")
            continue
        start = perf_counter()
        fa, qa = internal_frame(a, inside_a, cfg["rotation"])
        fb, qb = internal_frame(b, inside_b, cfg["rotation"])
        times["internal"] += perf_counter() - start
        report["internal"] = [qa, qb]
        if fa is None or fb is None:
            report.update(decision="ABSTAIN", reason="internal_evidence")
            continue
        start = perf_counter()
        decision, bad, details = vote(
            cross, fa, fb, (qa["quantile_deg"], qb["quantile_deg"]), cfg["vote"]
        )
        times["consensus"] += perf_counter() - start
        if decision == "SPLIT":
            if method == "per_edge_consensus_v1":
                removed.update(bad)
                decision = "PRUNE"
            else:
                removed.update(e.pair for e in cross)
        report.update(decision=decision, reason=details["reason"], boundary=details)
    pairs = [list(e.pair) for e in graph.edges]
    mask = [e.pair not in removed for e in graph.edges]
    if timings is not None:
        timings.update(times)
    return {
        "schema": "geometry_mask_v1",
        "method": method,
        "input_sha256": graph.payload_sha256,
        "policy_sha256": digest(encoded(cfg)),
        "components": reports,
        "pairs": pairs,
        "keep": mask,
        "mask_sha256": digest(encoded([pairs, mask])),
    }
