from math import ceil
import numpy as np
from scipy.cluster.hierarchy import DisjointSet

from .rotation_geometry import alignment, degrees


def internal_frame(nodes, edges, settings):
    """Build a maximum-weight tree using valid edges within this side only."""
    ordered = sorted(
        (e for e in edges if e.rotation is not None), key=lambda e: (-e.count, e.i, e.j)
    )
    groups = DisjointSet(nodes)
    tree = {n: [] for n in nodes}
    tree_pairs = []
    extra = []
    for e in ordered:
        if groups.merge(e.i, e.j):
            tree_pairs.append(e.pair)
            tree[e.i].append((e.j, e.rotation))
            tree[e.j].append((e.i, e.rotation.T))
        else:
            extra.append(e)
    info = {"tree_pairs": tree_pairs, "valid_internal_edges": len(ordered)}
    if len(tree_pairs) != len(nodes) - 1:
        return None, dict(info, reason="disconnected_rotation_graph")
    root = min(nodes)
    frames = {root: np.eye(3)}
    depth = {root: 0}
    queue = [root]
    while queue:
        i = queue.pop()
        for j, r in tree[i]:
            if j not in frames:
                frames[j] = r @ frames[i]
                depth[j] = depth[i] + 1
                queue.append(j)
    checks = [
        (e.i, e.j, float(degrees(alignment(frames[e.i], frames[e.j], e.rotation))))
        for e in extra
    ]
    info.update(
        frames={k: v.tolist() for k, v in frames.items()},
        tree_depth=depth,
        non_tree_checks=checks,
        checked_nodes=sorted({n for i, j, _ in checks for n in (i, j)}),
    )
    if len(checks) < settings["min_non_tree_checks_per_side"]:
        return None, dict(info, reason="too_few_internal_checks")
    q = sorted(v for _, _, v in checks)[
        ceil(settings["internal_quantile"] * len(checks)) - 1
    ]
    info["quantile_deg"] = q
    if q > settings["max_internal_error_deg"]:
        return None, dict(info, reason="unreliable_internal_frame")
    return frames, dict(info, reason="reliable")
