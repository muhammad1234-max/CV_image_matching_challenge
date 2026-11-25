import numpy as np
from .rotation_geometry import alignment, degrees


def vote(crossings, left, right, internal_quantiles, settings):
    transforms = {}
    selected = []
    used = set()
    for e in sorted(crossings, key=lambda e: (-e.count, e.i, e.j)):
        if e.rotation is None:
            continue
        i, j, r = e.i, e.j, e.rotation
        if i not in left:
            i, j, r = j, i, r.T
        transforms[e.pair] = alignment(left[i], right[j], r)
        if (
            len(selected) < settings["maximum_pairs"]
            and i not in used
            and j not in used
        ):
            selected.append(e.pair)
            used.update((i, j))
    info = {
        "crossing_count": len(crossings),
        "valid_crossing_count": len(transforms),
        "witnesses": selected,
        "alignments": [[*k, q.tolist()] for k, q in transforms.items()],
    }
    if len(selected) < settings["minimum_pairs"]:
        return "ABSTAIN", set(), dict(info, reason="too_few_disjoint_witnesses")
    tau = max(
        settings["radius_floor_deg"],
        settings["radius_multiplier_on_sum_internal_quantiles"]
        * sum(internal_quantiles),
    )
    q = np.stack([transforms[k] for k in selected])
    distances = degrees(q.transpose(0, 2, 1)[:, None] @ q[None, :])
    counts = (distances <= tau).sum(axis=1)
    center = int(counts.argmax())
    split = (
        int(counts[center])
        < len(selected) * settings["split_if_max_fraction_strictly_below"]
    )
    residuals = {k: float(degrees(q[center].T @ m)) for k, m in transforms.items()}
    info.update(
        radius_deg=tau,
        witness_distance_deg=distances.tolist(),
        consensus_counts=counts.tolist(),
        best_center=selected[center],
        best_count=int(counts[center]),
        valid_crossing_residuals=[[*k, v] for k, v in residuals.items()],
    )
    return (
        "SPLIT" if split else "RETAIN",
        {k for k, v in residuals.items() if v > tau},
        dict(info, reason="incompatible_alignments" if split else "common_alignment"),
    )
