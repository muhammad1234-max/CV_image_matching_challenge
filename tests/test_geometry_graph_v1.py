"""Synthetic expectations recorded in docs/phase4b_synthetic_validation.md."""

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.novel_model.geometry_graph_v1.boundary_consensus import vote
from src.novel_model.geometry_graph_v1.decision_engine import decide
from src.novel_model.geometry_graph_v1.fixtures import (
    as_graph,
    fixture,
    manifest,
    refresh_membership,
)
from src.novel_model.geometry_graph_v1.graph_import import import_bytes, load_cache
from src.novel_model.geometry_graph_v1.rotation_geometry import degrees
from src.novel_model.geometry_graph_v1.schemas import ROOT, digest, encoded, policy

CFG = policy()
A = "candidate_a_v1"


def run(obj, method=A):
    return decide(as_graph(obj, CFG), method, CFG)


@pytest.mark.parametrize(
    "kind,result",
    [
        ("rigid", "RETAIN"),
        ("contradiction", "SPLIT"),
        ("noisy", "RETAIN"),
        ("strong_false", "SPLIT"),
        ("coherent_alias", "RETAIN"),
        ("few_witnesses", "ABSTAIN"),
        ("few_loops", "ABSTAIN"),
        ("bad_loops", "ABSTAIN"),
        ("missing", "ABSTAIN"),
        ("spectral_tie", "ABSTAIN"),
        ("near_tie", "ABSTAIN"),
        ("shared_endpoint", "ABSTAIN"),
        ("sparse", "ABSTAIN"),
        ("half", "RETAIN"),
    ],
)
def test_geometric_scenarios(kind, result):
    out = run(fixture(kind))
    (report,) = out["components"]
    assert report["decision"] == result
    pairs = {tuple(p) for p, k in zip(out["pairs"], out["keep"]) if not k}
    if result == "SPLIT":
        assert pairs == {
            (f"image_{i:04}.png", f"image_{i + 8:04}.png") for i in range(8)
        }
    else:
        assert not pairs
    if kind in ("near_tie", "spectral_tie"):
        assert report["reason"] == "degenerate_fiedler_space"
    if kind in ("sparse", "few_witnesses", "shared_endpoint"):
        assert report["reason"] == "too_few_disjoint_witnesses"
    if kind == "half":
        assert report["boundary"]["best_count"] == 4


def test_disconnected_rigid_groups():
    obj = fixture("disconnected")
    out = run(obj)
    assert [len(r["nodes"]) for r in out["components"]] == [8, 8]
    assert all(out["keep"])


def test_empty_and_isolated_nodes():
    obj = fixture("isolated")
    assert run(obj)["components"][-1]["reason"] == "small_component"
    obj["edges"] = []
    assert len(run(obj)["components"]) == 17
    obj["nodes"] = []
    assert run(refresh_membership(obj))["components"] == []


def test_noop_immutability():
    graph = as_graph(fixture("contradiction"), CFG)
    before = [e.rotation.tobytes() for e in graph.edges]
    assert all(decide(graph, config=CFG)["keep"])
    decide(graph, A, CFG)
    assert before == [e.rotation.tobytes() for e in graph.edges]
    assert not graph.edges[0].rotation.flags.writeable


@pytest.mark.parametrize(
    "value",
    [
        np.diag([1, 1, -1]).tolist(),
        [[1, 2], [2, 1]],
        (2 * np.eye(3)).tolist(),
        np.full((3, 3), float("nan")).tolist(),
        np.full((3, 3), float("inf")).tolist(),
    ],
)
def test_invalid_rotations(value):
    obj = fixture()
    obj["edges"][0]["rotation"] = value
    data = json.dumps(obj).encode()
    with pytest.raises(ValueError):
        import_bytes(data, {**obj["provenance"], "payload_sha256": digest(data)}, CFG)


@pytest.mark.parametrize("count", [0, -1, None, 2.5, True, 2**65])
def test_counts(count):
    obj = fixture()
    obj["edges"][0]["inliers"] = count
    with pytest.raises(ValueError):
        as_graph(obj, CFG)


@pytest.mark.parametrize(
    "mutation",
    [
        "node_dup",
        "outside",
        "loop",
        "duplicate_reverse",
        "membership",
        "direction",
        "mask_conflict",
        "unknown_schema",
        "node_name",
        "mode",
    ],
)
def test_schema_errors(mutation):
    obj = fixture()
    e = obj["edges"][0]
    if mutation == "node_dup":
        obj["nodes"].append(obj["nodes"][0])
    if mutation == "outside":
        e["i"] = "image_9999.png"
    if mutation == "loop":
        e["j"] = e["i"]
    if mutation == "duplicate_reverse":
        obj["edges"].append({**e, "i": e["j"], "j": e["i"]})
    if mutation == "membership":
        obj["nodes"].pop()
    if mutation == "direction":
        obj["convention"] = "source_from_target"
    if mutation == "mask_conflict":
        e["rotation_valid"] = False
    if mutation == "unknown_schema":
        obj["schema"] = "v0"
    if mutation == "node_name":
        obj["nodes"][0] = "bike.png"
    if mutation == "mode":
        obj["mode"] = "verified_labels"
    with pytest.raises(ValueError):
        as_graph(obj, CFG)


@pytest.mark.parametrize(
    "location,field",
    [("root", "scene"), ("edge", "gt_pose"), ("provenance", "outlier")],
)
def test_administrative_data_rejected(location, field):
    obj = fixture()
    target = {"root": obj, "edge": obj["edges"][0], "provenance": obj["provenance"]}[
        location
    ]
    target[field] = 1
    with pytest.raises(ValueError):
        as_graph(obj, CFG)


def test_digest_and_independent_provenance(tmp_path):
    obj = fixture()
    path = tmp_path / "graph.json"
    path.write_bytes(encoded(obj))
    trusted = manifest(obj)
    assert load_cache(path, trusted, CFG).payload_sha256 == trusted["payload_sha256"]
    with pytest.raises(ValueError, match="provenance"):
        load_cache(path, {**trusted, "source_sha256": "0" * 64}, CFG)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest"):
        load_cache(path, trusted, CFG)


def test_topology_only():
    obj = fixture("missing")
    obj["mode"] = "topology_only"
    for e in obj["edges"]:
        e["inliers"] = None
    out = run(obj)
    assert out["components"][0]["reason"] == "missing_inlier_counts" and all(
        out["keep"]
    )
    assert all(run(obj, "normalized_cut_v1")["keep"])


def test_missing_rotation_omitted():
    obj = fixture("missing")
    for e in obj["edges"]:
        del e["rotation"]
    assert all(e.rotation is None for e in as_graph(obj, CFG).edges)


def test_reverse_and_input_permutation():
    obj = fixture("contradiction")
    before = run(obj)
    obj["nodes"].reverse()
    obj["edges"].reverse()
    for e in obj["edges"]:
        e["i"], e["j"] = e["j"], e["i"]
        e["rotation"] = np.array(e["rotation"]).T.tolist()
    out = run(obj)
    assert before["components"] == out["components"]
    assert before["mask_sha256"] == out["mask_sha256"]


@pytest.mark.parametrize("seed", range(4))
def test_gauge_invariance(seed):
    obj = fixture("contradiction", seed=seed)
    graph = as_graph(obj, CFG)
    report = decide(graph, A, CFG)["components"][0]
    sides = [
        {k: np.array(v) for k, v in side["frames"].items()}
        for side in report["internal"]
    ]
    gauges = Rotation.random(2, random_state=seed + 9).as_matrix()
    frames = [{k: v @ g for k, v in s.items()} for s, g in zip(sides, gauges)]
    cross = [e for e in graph.edges if (e.i in sides[0]) != (e.j in sides[0])]
    decision, _, info = vote(cross, *frames, [0, 0], CFG["vote"])
    assert decision == report["decision"]
    assert np.allclose(
        info["witness_distance_deg"], report["boundary"]["witness_distance_deg"]
    )


def test_renumber_unique_boundary():
    obj = fixture("contradiction")
    before = run(obj)
    names = obj["nodes"]
    mapping = dict(zip(names, np.random.default_rng(456).permutation(names).tolist()))
    for e in obj["edges"]:
        e["i"], e["j"] = mapping[e["i"]], mapping[e["j"]]
    after = run(obj)
    mapped = {
        tuple(sorted(mapping[n] for n in pair))
        for pair, keep in zip(before["pairs"], before["keep"])
        if not keep
    }
    assert mapped == {tuple(p) for p, k in zip(after["pairs"], after["keep"]) if not k}


def test_swap_local_frame_labels():
    graph = as_graph(fixture("contradiction"), CFG)
    report = decide(graph, A, CFG)["components"][0]
    a, b = [
        {k: np.array(v) for k, v in s["frames"].items()} for s in report["internal"]
    ]
    cross = [e for e in graph.edges if (e.i in a) != (e.j in a)]
    original = vote(cross, a, b, [0, 0], CFG["vote"])
    swapped = vote(cross, b, a, [0, 0], CFG["vote"])
    assert original[:2] == swapped[:2]
    assert np.allclose(
        original[2]["witness_distance_deg"], swapped[2]["witness_distance_deg"]
    )


def test_reference_controls():
    obj = fixture("contradiction")
    for method, removed in [
        ("normalized_cut_v1", 8),
        ("triangle_v1", 0),
        ("per_edge_consensus_v1", 7),
        ("pass_through_v1", 0),
    ]:
        assert run(obj, method)["keep"].count(False) == removed
    assert run(fixture("rigid"), "normalized_cut_v1")["keep"].count(False) == 8


def test_triangle_detects_single_bad_edge():
    obj = fixture()
    e = obj["edges"][0]
    e["rotation"] = (
        np.array(e["rotation"]) @ Rotation.from_euler("z", 80, degrees=True).as_matrix()
    ).tolist()
    result = run(obj, "triangle_v1")
    assert [p for p, k in zip(result["pairs"], result["keep"]) if not k] == [
        [e["i"], e["j"]]
    ]


def test_unknown_crossing_edge():
    obj = fixture("contradiction")
    obj["edges"][-1].update(rotation=None, rotation_valid=False)
    assert run(obj)["keep"].count(False) == 8
    assert run(obj, "per_edge_consensus_v1")["keep"].count(False) == 6


def test_cap_and_serialization():
    graph = as_graph(fixture("rigid", n=76), CFG)
    r = decide(graph, A, CFG)
    assert len(r["components"][0]["boundary"]["witnesses"]) == 32
    assert json.loads(encoded(r)) == json.loads(encoded(decide(graph, A, CFG)))
    assert r["mask_sha256"] == digest(encoded([r["pairs"], r["keep"]]))


def test_degree_extremes():
    assert np.allclose(
        degrees(
            Rotation.from_euler(
                "x", np.array([0, 1e-9, 179.999999, 180])[:, None], degrees=True
            ).as_matrix()
        ),
        [0, 1e-9, 179.999999, 180],
        atol=1e-10,
    )


def test_pinned_inputs():
    for path, expected in CFG["reference_file_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    assert (
        digest(
            (ROOT / "configs/novel_model/architecture_proposal_v1.yaml").read_bytes()
        )
        == CFG["proposal_sha256"]
    )


def test_cli_default_noop(tmp_path):
    obj = fixture("contradiction")
    cache, trust, out = (tmp_path / f for f in ("cache", "manifest", "out"))
    cache.write_bytes(encoded(obj))
    trust.write_bytes(encoded(manifest(obj)))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.novel_model.geometry_graph_v1",
            str(cache),
            str(trust),
            "--output",
            str(out),
        ],
        check=True,
        cwd=ROOT,
    )
    result = json.loads(out.read_bytes())
    assert result["method"] == "pass_through_v1" and all(result["keep"])


def test_archived_haiper_is_topology_only():
    report = ROOT / "reports/phase4b"
    rows = json.loads((report / "cache_inventory.json").read_bytes())["four"]
    row = next(r for r in rows if r["dataset"] == "imc2023_haiper")
    trust = row["topology_import"]["trust"]
    graph = load_cache(report / "imc2023_haiper_topology.json", trust, CFG)
    assert (len(graph.nodes), len(graph.edges), graph.mode) == (
        54,
        454,
        "topology_only",
    )
    assert all(e.count is None and e.rotation is None for e in graph.edges)
    out = decide(graph, A, CFG)
    assert all(out["keep"]) and all(
        r["decision"] == "ABSTAIN" for r in out["components"]
    )


def test_nearest_rank_and_adaptive_radius():
    from src.novel_model.geometry_graph_v1.internal_consistency import internal_frame
    from src.novel_model.geometry_graph_v1.schemas import Edge

    nodes = tuple("abcdef")
    edges = [Edge("a", n, 100, np.eye(3)) for n in nodes[1:]]
    edges += [
        Edge(i, j, 4, Rotation.from_euler("z", float(v), degrees=True).as_matrix())
        for i, j, v in [("b", "c", 1), ("b", "d", 2), ("b", "e", 3), ("b", "f", 9)]
    ]
    frame, info = internal_frame(nodes, edges, CFG["rotation"])
    assert frame and info["quantile_deg"] == pytest.approx(9)
    out = run(fixture("noisy"))
    internal = out["components"][0]["internal"]
    assert all(side["reason"] == "reliable" for side in internal)
    graph = as_graph(fixture("rigid"), CFG)
    side = decide(graph, A, CFG)["components"][0]["internal"]
    a, b = [{n: np.array(v) for n, v in s["frames"].items()} for s in side]
    cross = [e for e in graph.edges if (e.i in a) != (e.j in a)]
    assert vote(cross, a, b, (9, 9), CFG["vote"])[2]["radius_deg"] == 36


def test_ineligible_rotations_disconnect_one_side():
    obj = fixture()
    for e in obj["edges"]:
        if "image_0000.png" in (e["i"], e["j"]):
            e.update(rotation=None, rotation_valid=False)
    result = run(obj)
    assert result["components"][0]["decision"] == "ABSTAIN" and all(result["keep"])


def test_multiple_eligible_components_are_not_recursively_split():
    first, second = fixture("contradiction"), fixture("rigid")
    renaming = {n: f"image_{k + 16:04}.png" for k, n in enumerate(second["nodes"])}
    first["nodes"].extend(renaming.values())
    for e in second["edges"]:
        first["edges"].append({**e, "i": renaming[e["i"]], "j": renaming[e["j"]]})
    result = run(refresh_membership(first))
    assert [r["decision"] for r in result["components"]] == ["SPLIT", "RETAIN"]


@pytest.mark.parametrize(
    "damage",
    ["none", "flip", "reorder", "foreign_parent", "wrong_policy", "integer_mask"],
)
def test_mask_replay_checks(damage):
    from src.novel_model.geometry_graph_v1.diagnostics import checked_keep_mask

    graph = as_graph(fixture("contradiction"), CFG)
    receipt = json.loads(encoded(decide(graph, A, CFG)))
    if damage == "none":
        assert checked_keep_mask(receipt, graph, CFG).count(False) == 8
        return
    if damage == "flip":
        receipt["keep"][0] = not receipt["keep"][0]
    if damage == "reorder":
        receipt["pairs"].reverse()
    if damage == "foreign_parent":
        receipt["input_sha256"] = "7" * 64
    if damage == "wrong_policy":
        receipt["policy_sha256"] = "2" * 64
    if damage == "integer_mask":
        receipt["keep"][0] = 1
    with pytest.raises(ValueError):
        checked_keep_mask(receipt, graph, CFG)


def test_equal_cost_connected_sweep_ties():
    from src.novel_model.geometry_graph_v1.spectral_proposal import propose
    from src.novel_model.geometry_graph_v1.schemas import Edge

    names = [f"image_{i:04}.png" for i in range(13)]
    edges = [Edge(i, j, 1, None) for i, j in zip(names, names[1:])]
    cut, info = propose(names, edges, CFG["graph"])
    assert cut[0] == tuple(names[:6])
    assert len(info["sweep_costs"]) == 2


def test_sparse_solver_large_component():
    result = run(fixture("rigid", n=270))
    assert result["components"][0]["decision"] == "RETAIN" and all(result["keep"])
