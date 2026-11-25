"""A one-trial selector over corresponding observations of one sealed COLMAP parent."""
from contextlib import closing
from pathlib import Path
import re
import numpy as np
import yaml

from .observations import export_observations, immutable_db
from ..geometry_capture_v1.database import file_sha, pair_ids
from ..geometry_graph_v1.schemas import Edge, Graph
from ..reconstruction_feedback_v2 import candidate

CONFIG = Path(__file__).resolve().parents[3] / "configs/novel_model/final_candidate_v1.yaml"
POLICY = yaml.safe_load(CONFIG.read_bytes())


def parent_graph(path, expected, names):
    if file_sha(path) != expected or not 0 < len(names) <= POLICY["max_images"]:
        raise ValueError("parent digest or image budget")
    if names != [f"image_{i:04d}.png" for i in range(len(names))]:
        raise ValueError("expected frozen opaque order")
    with closing(immutable_db(path)) as db:
        by_id = dict(db.execute("SELECT image_id,name FROM images"))
        if sorted(by_id.values()) != names:
            raise ValueError("parent membership")
        edges = []
        for pid, count, cols, size in db.execute(
            "SELECT pair_id,rows,cols,length(data) FROM two_view_geometries ORDER BY pair_id"
        ):
            i, j = pair_ids(pid)
            if min(i,j) not in by_id or j not in by_id or count < 0 or cols != 2 or (size or 0) != count*8:
                raise ValueError("invalid verified edge")
            if count:
                a,b = sorted((by_id[i],by_id[j]))
                edges.append(Edge(a,b,count,None))
    graph = Graph(tuple(names),tuple(sorted(edges,key=lambda e:e.pair)),{},expected,"verified_counts")
    return graph, {v:k for k,v in by_id.items()}


def partition(path, expected, names):
    graph, ids = parent_graph(path, expected, names)
    proposal = candidate(graph)
    if proposal is None:
        return None, []
    from ..geometry_capture_v1.database import MAX_ID
    excluded = []
    for a in proposal["sides"][0]:
        for b in proposal["sides"][1]:
            i,j = sorted((ids[a],ids[b]))
            excluded.append(i*MAX_ID+j)
    return proposal, sorted(excluded)


def memberships(groups, errors):
    flat = [n for g in groups for n in g]
    if (len(flat) != len(set(flat)) or set(flat) != set(errors) or any(not g for g in groups)
            or any(not re.fullmatch(r"image_\d{4}\.png", n) for n in flat)):
        raise ValueError("invalid group/feature inventory")
    return {frozenset(g) for g in groups}


def decide(sides, original, trial):
    evidence = {}
    def result(action, reason):
        return dict(action=action, reason=reason, selected="trial" if action=="ACCEPT" else "original",
                    per_image=evidence)
    if not sides or trial is None:
        return result("ABSTAIN","no proposal or completed trial")
    groups, baseline = original
    changed, variant = trial
    bg, tg = memberships(groups,baseline), memberships(changed,variant)
    left,right = map(frozenset,sides)
    if min(len(left),len(right)) < 6 or left & right or left | right not in bg:
        return result("ABSTAIN","ineligible original component")
    if tg != (bg-{left | right}) | {left,right}:
        return result("REJECT","registration or model membership changed outside the proposed split")
    cap, margin = POLICY["pixel_cap"], POLICY["pixel_margin"]
    for n in sorted(baseline):
        a,b = map(np.asarray,(baseline[n],variant[n]))
        if a.ndim != 1 or a.shape != b.shape or any(np.isinf(x).any() or (x<0).any() for x in (a,b)):
            raise ValueError("invalid indexed residuals")
        present, after = np.isfinite(a), np.isfinite(b)
        common = present & after
        count = int(present.sum())
        if count == 0:
            return result("ABSTAIN","empty original support")
        hard = present & (a >= np.quantile(a[present],POLICY["hard_quantile"]))
        delta = np.minimum(b[common],cap) - np.minimum(a[common],cap)
        lost = int((present & ~after).sum())
        evidence[n] = dict(original=count,common=int(common.sum()),lost=lost,
            gained=int((~present & after).sum()),hard=int(hard.sum()),hard_common=int((hard & after).sum()),
            cost_delta_px=float((delta.sum()+cap*lost)/count))
    if any(v["common"] < POLICY["minimum_common"] for v in evidence.values()):
        return result("ABSTAIN","insufficient paired support")
    for v in evidence.values():
        if any(v[k] < POLICY["minimum_retention"]*v[d] for k,d in (("common","original"),("hard_common","hard"))):
            return result("REJECT","lost observation coverage")
        if v["cost_delta_px"] > margin:
            return result("REJECT","per-image cost regression")
    if any(np.median([evidence[n]["cost_delta_px"] for n in g]) >= -margin for g in (left,right)):
        return result("REJECT","insufficient improvement on both sides")
    return result("ACCEPT","paired support and loss-penalized costs passed")
