from itertools import combinations
import random
import pytest
from src.novel_model.graph_refinement_v1 import import_graph, refine

CASES = ["clean", "weak", "dense", "repeat", "unequal", "sparse_true", "noisy",
         "disconnected", "isolated", "ambiguous_single"]


def fixture(case):
    n = 21 if case == "unequal" else 16
    nodes = [str(i) for i in range(n)]
    rng = random.Random(17)
    rows = []
    for i, j in combinations(range(n), 2):
        inside = (i < 8) == (j < 8)
        if case == "isolated":
            continue
        if inside or case == "clean":
            count = rng.randint(60, 140) if case == "noisy" else 100
        elif case == "dense" and (i+j) % 2 == 0:
            count = 100
        elif case == "repeat" and i < 3 and j < 11:
            count = 100
        elif case != "disconnected" and (i,j) in [(0,8),(1,9)]:
            if case == "sparse_true" and i:
                continue
            count = 4
        else:
            continue
        rows.append((str(i), str(j), count))
    return nodes, rows


@pytest.mark.parametrize("case", CASES)
def test_replay(case):
    nodes, rows = fixture(case)
    g = import_graph(nodes, rows)
    r = refine(g)
    assert r == refine(import_graph(nodes[::-1], rows[::-1]))
    assert all(r["pass_through"]) and len(r["pairs"]) == len(rows)
    assert len(r["nodes"]) == len(nodes)
    rename = dict(zip(nodes, reversed(nodes)))
    other = refine(import_graph(nodes, [(rename[a],rename[b],v) for a,b,v in rows]))
    def cut(result):
        return {frozenset(pair) for pair,keep in zip(result["pairs"],result["candidate_b2"]) if not keep}
    assert {frozenset(rename[n] for n in pair) for pair in cut(r)} == cut(other)
    if case in ("weak","unequal","noisy","ambiguous_single"):
        assert cut(r) == {frozenset(("0","8")),frozenset(("1","9"))}
    if case in ("clean","sparse_true","isolated","disconnected"):
        assert not cut(r)
    if case == "sparse_true":
        assert r["components"][0]["decision"] == "ABSTAIN"
