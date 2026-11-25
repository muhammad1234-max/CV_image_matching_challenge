from collections import defaultdict
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.linalg import ArpackNoConvergence, eigsh


def reachable(start, allowed, neighbors):
    seen = {start}
    stack = [start]
    while stack:
        for other in neighbors[stack.pop()]:
            if other in allowed and other not in seen:
                seen.add(other)
                stack.append(other)
    return seen


def components(graph):
    neighbors = {n: set() for n in graph.nodes}
    by_first = defaultdict(list)
    for e in graph.edges:
        neighbors[e.i].add(e.j)
        neighbors[e.j].add(e.i)
        by_first[e.i].append(e)
    unseen = set(graph.nodes)
    for n in graph.nodes:
        if n in unseen:
            part = reachable(n, unseen, neighbors)
            unseen.difference_update(part)
            nodes = sorted(part)
            yield nodes, [e for i in nodes for e in by_first[i]]


def propose(nodes, edges, settings):
    n = len(nodes)
    minimum = settings["min_side_images"]
    if n < 2 * minimum:
        return None, {"reason": "small_component"}
    if any(e.count is None for e in edges):
        return None, {"reason": "missing_inlier_counts"}
    ids = {s: i for i, s in enumerate(nodes)}
    src = [ids[e.i] for e in edges]
    dst = [ids[e.j] for e in edges]
    w = [e.weight for e in edges]
    adj = coo_matrix((w + w, (src + dst, dst + src)), shape=(n, n)).tocsr()
    degree = np.asarray(adj.sum(axis=1)).ravel()
    inv = 1 / np.sqrt(degree)
    scaled = diags(inv)
    lap = eye(n, format="csr") - scaled @ adj @ scaled
    try:
        if n <= 256:
            values, vec = eigh(lap.toarray(), subset_by_index=(0, 2))
        else:
            start = np.random.default_rng(0).normal(size=n)
            values, vec = eigsh(lap, k=3, which="SA", v0=start, tol=1e-12)
            order = np.argsort(values)
            values, vec = values[order], vec[:, order]
    except (np.linalg.LinAlgError, ArpackNoConvergence):
        return None, {"reason": "eigensolver_failure"}
    info = {"lowest_eigenvalues": values.tolist()}
    if values[2] - values[1] <= settings["eigengap_min"]:
        return None, dict(info, reason="degenerate_fiedler_space")
    x = inv * vec[:, 1]
    # An eigenvector and its negative must give exactly the same tie order.
    if x[np.flatnonzero(np.abs(x) > 1e-14)[0]] < 0:
        x = -x
    order = sorted(range(n), key=lambda i: (round(float(x[i]), 12), nodes[i]))
    info["sweep_order"] = [nodes[i] for i in order]
    info["fiedler_coordinates"] = x.tolist()
    visited = set()
    boundary = volume = 0.0
    total = degree.sum()
    costs = []
    neighbors = [adj.indices[adj.indptr[i] : adj.indptr[i + 1]] for i in range(n)]
    for k, i in enumerate(order, 1):
        weights = adj.data[adj.indptr[i] : adj.indptr[i + 1]]
        inside = sum(v for j, v in zip(neighbors[i], weights) if j in visited)
        boundary += degree[i] - 2 * inside
        volume += degree[i]
        visited.add(i)
        if minimum <= k <= n - minimum:
            costs.append((float(boundary * (1 / volume + 1 / (total - volume))), k))
    info["sweep_costs"] = costs
    universe = set(order)
    best = None
    minimum_connected = None
    for cost, k in sorted(costs):
        if (
            minimum_connected is not None
            and cost > minimum_connected + settings["tie_tolerance"]
        ):
            break
        a = set(order[:k])
        b = universe - a
        if len(reachable(order[0], a, neighbors)) != len(a) or len(
            reachable(order[-1], b, neighbors)
        ) != len(b):
            continue
        if minimum_connected is None:
            minimum_connected = cost
        sides = tuple(
            sorted(
                (tuple(sorted(nodes[i] for i in a)), tuple(sorted(nodes[i] for i in b)))
            )
        )
        if best is None or sides < best:
            best = sides
            info["cost"] = cost
    return best, dict(info, reason="proposed" if best else "no_connected_sweep")
