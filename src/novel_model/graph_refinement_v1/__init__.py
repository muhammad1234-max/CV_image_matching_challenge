"""B2 CPU diagnostic. Inputs are opaque vertex identifiers and integer support only."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import yaml
from ..geometry_graph_v1.schemas import Graph, Edge
from ..geometry_graph_v1.spectral_proposal import components, propose

CONFIG = Path(__file__).resolve().parents[3] / "configs/novel_model/graph_refinement_v1.yaml"


def import_graph(nodes, rows):
    ids = set(nodes)
    if len(ids) != len(nodes):
        raise ValueError("duplicate node")
    pairs = {}
    for i, j, count in rows:
        key = tuple(sorted((i, j)))
        if i == j or i not in ids or j not in ids or key in pairs:
            raise ValueError("invalid endpoints")
        if type(count) is not int or count < 0:
            raise ValueError("invalid support")
        pairs[key] = Edge(*key, count, None)
    return Graph(tuple(sorted(ids)), tuple(pairs[k] for k in sorted(pairs)), {}, "", "verified_counts")


def refine(graph, settings=None):
    cfg = settings or yaml.safe_load(CONFIG.read_text())
    powers = np.random.default_rng(cfg["seed"]).uniform(*cfg["exponent_range"], cfg["trials"])
    removed = set()
    baseline = set()
    diagnostics = []
    for nodes, edges in components(graph):
        sides, info = (propose(nodes, edges, cfg) if all(e.count for e in edges)
                       else (None, {"reason": "zero_support"}))
        row = dict(nodes=nodes, sides=sides, proposal=info, decision="ABSTAIN")
        diagnostics.append(row)
        if sides is None:
            continue
        left = set(sides[0])
        cross = [e for e in edges if (e.i in left) != (e.j in left)]
        inside = [e.count for e in edges if (e.i in left) == (e.j in left)]
        baseline.update(e.pair for e in cross)
        neighbors = {n: set() for n in nodes}
        for e in edges:
            neighbors[e.i].add(e.j)
            neighbors[e.j].add(e.i)
        overlaps = [len(neighbors[e.i] & neighbors[e.j]) /
                    len(neighbors[e.i] | neighbors[e.j]) for e in cross]
        overlap = float(np.quantile(overlaps, 0.75))
        ratio = float(np.quantile([e.count for e in cross], 0.75) / np.median(inside))
        agrees = []
        for power in powers:
            altered = [replace(e, count=float(e.count) ** power) for e in edges]
            trial, _ = propose(nodes, altered, cfg)
            agrees.append(trial == sides)
        stability = sum(agrees) / len(agrees)
        decision = "ABSTAIN"
        if ratio >= 1 or overlap >= 0.5:
            decision = "RETAIN"
        elif (ratio <= cfg["max_ratio"] and overlap <= cfg["max_overlap"]
              and stability >= cfg["min_stability"] and len(cross) >= cfg["min_boundary"]):
            decision = "SPLIT"
            removed.update(e.pair for e in cross)
        row.update(decision=decision, support_ratio=ratio, overlap=overlap,
                   stability=stability, trial_agreement=agrees, boundary=len(cross))
    pairs = [e.pair for e in graph.edges]
    return dict(nodes=graph.nodes, pairs=pairs, components=diagnostics,
                pass_through=[True for _ in pairs], normalized_cut=[p not in baseline for p in pairs],
                candidate_b2=[p not in removed for p in pairs])
