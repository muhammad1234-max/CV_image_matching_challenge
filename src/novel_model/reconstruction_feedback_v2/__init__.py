"""Label-free quality rule on common immutable keypoint indices."""
from pathlib import Path
import numpy as np
import yaml
from ..geometry_graph_v1.spectral_proposal import components, propose

CONFIG = Path(__file__).resolve().parents[3]/"configs/novel_model/reconstruction_feedback_v2.yaml"
POLICY = yaml.safe_load(CONFIG.read_bytes())


def candidate(graph):
    choices = []
    for nodes, edges in components(graph):
        sides, _ = propose(nodes, edges, POLICY)
        if not sides:
            continue
        left = set(sides[0])
        nbr = {n:set() for n in nodes}
        for e in edges:
            nbr[e.i].add(e.j)
            nbr[e.j].add(e.i)
        cross = [e.pair for e in edges if (e.i in left) != (e.j in left)]
        median = float(np.median([len(nbr[a]&nbr[b])/len(nbr[a]|nbr[b]) for a,b in cross]))
        choices.append(dict(sides=sides,median=median,cross=cross))
    return min(choices,key=lambda c:(c["median"],c["sides"]),default=None)


def decide(sides, baseline, trial):
    """Branches have a list of model memberships, point counts and per-image residual arrays.
    NaN denotes untriangulated observations at the same immutable feature index.
    """
    diagnostics={}
    def fallback(reason):
        return dict(selected="pass_through",reason=reason,observations=diagnostics)
    if not sides or not trial:
        return fallback("no candidate or incomplete trial")
    left,right=map(frozenset,sides)
    group=left|right
    bg={frozenset(m["images"]):m["points"] for m in baseline["models"]}
    tg={frozenset(m["images"]):m["points"] for m in trial["models"]}
    if min(len(left),len(right))<POLICY["min_side_images"] or left&right or group not in bg:
        return fallback("baseline did not fully register the component as one model")
    if set(tg)!=(set(bg)-{group})|{left,right}:
        return fallback("unexpected registration, outliers or fragmentation")
    if tg[left]+tg[right]<POLICY["minimum_retention"]*bg[group]:
        return fallback("affected point loss")
    for name in sorted(baseline["errors"]):
        a=baseline["errors"][name]
        b=trial["errors"][name]
        if a.shape!=b.shape or np.isinf(a).any() or np.isinf(b).any():
            return fallback("invalid feature arrays")
        ma,mb=np.isfinite(a),np.isfinite(b)
        common=ma&mb
        n=int(common.sum())
        delta=float(np.median(b[common]-a[common])) if n else None
        diagnostics[name]=dict(common=n,baseline=int(ma.sum()),trial=int(mb.sum()),
                               lost=int((ma&~mb).sum()),gained=int((mb&~ma).sum()),median_delta_px=delta)
    for v in diagnostics.values():
        if v["common"]<max(POLICY["minimum_common_per_image"],POLICY["minimum_retention"]*v["baseline"]):
            return fallback("insufficient retained observations")
        if v["median_delta_px"]>POLICY["pixel_margin"]:
            return fallback("image-level regression")
    for side in (left,right):
        if np.median([diagnostics[n]["median_delta_px"] for n in side])>=-POLICY["pixel_margin"]:
            return fallback("both sides require a material paired gain")
    return dict(selected="trial",reason="passed all reconstruction checks",observations=diagnostics)
