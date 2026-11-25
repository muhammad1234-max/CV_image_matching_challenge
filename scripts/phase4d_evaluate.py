"""Run from repository root with python -m scripts.phase4d_evaluate."""
import hashlib
import json
from pathlib import Path
import runpy
import time
import psutil
from src.novel_model.graph_refinement_v1 import CONFIG, import_graph, refine

out = Path("reports/phase4d")
parent = out / "replay"
graph = json.loads((parent / "graph.json").read_bytes())
trust = json.loads((parent / "graph_trust.json").read_bytes())
assert hashlib.sha256((parent / "graph.json").read_bytes()).hexdigest() == trust["payload_sha256"]
validation = json.loads((parent / "validation.json").read_bytes())
assert validation["parent_verified"] and validation["graph_decisions_reproduced"]
rows = [(e["i"], e["j"], e["inliers"]) for e in graph["edges"]]
start = time.perf_counter()
result = refine(import_graph(graph["nodes"], rows))
elapsed = time.perf_counter() - start
rss = psutil.Process().memory_info()
result["resources"] = dict(seconds=elapsed, process_peak_bytes=getattr(rss,"peak_wset",None))
result["config_sha256"] = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
original = json.loads((parent / "normalized_cut_v1.json").read_bytes())
assert list(map(list,result["pairs"])) == original["pairs"]
assert result["normalized_cut"] == original["keep"]
assert all(c["nodes"] == o["nodes"] and list(map(list,c["sides"])) == o["sides"]
           for c,o in zip(result["components"],original["components"],strict=True))
(out / "haiper.json").write_text(json.dumps(result,indent=2))
tests = runpy.run_path("tests/test_graph_refinement_v1.py")
table = ["| Fixture | B2 component decisions | B2 removed | Ncut removed | Pass-through removed |",
         "|---|---|---:|---:|---:|"]
for name in tests["CASES"]:
    fixture = refine(import_graph(*tests["fixture"](name)))
    (out / (name+".json")).write_text(json.dumps(fixture,indent=2))
    decisions = ", ".join(c["decision"] for c in fixture["components"])
    table.append(f"| {name} | {decisions} | {fixture['candidate_b2'].count(False)} | {fixture['normalized_cut'].count(False)} | 0 |")
Path("docs/phase4d_synthetic_results.md").write_text("# Synthetic comparisons\n\n"+"\n".join(table)+"\n\nWeak and ambiguous_single have identical observations and opposite intended decisions. The latter is a false split of one scene. Dense and repeat encode misleading edges between cliques. These fixtures measure behavior, not competition accuracy.\n")
print(result["resources"])
for c in result["components"]:
    print(len(c["nodes"]), list(map(len,c["sides"])), c["decision"],c["boundary"],c["overlap"],c["support_ratio"],c["stability"])
