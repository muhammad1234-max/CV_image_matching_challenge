# ruff: noqa: E402 -- direct script entry points resolve the project root first.
"""Profile CPU synthetic graphs only, one child process per graph size."""

import argparse
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import scipy
from src.novel_model.geometry_graph_v1.fixtures import fixture, as_graph, manifest
from src.novel_model.geometry_graph_v1.schemas import policy, encoded, METHODS
from src.novel_model.geometry_graph_v1.graph_import import import_bytes
from src.novel_model.geometry_graph_v1.decision_engine import decide

REPORT = ROOT / "reports/phase4b"


def measure(n):
    cfg = policy()
    obj = fixture("rigid", n)
    data, trust = encoded(obj), manifest(obj)
    process = psutil.Process()
    before = process.memory_info().rss
    samples = []
    for _ in range(3):
        times = {}
        start = perf_counter()
        graph = import_bytes(data, trust, cfg)
        imported = perf_counter()
        out = decide(graph, "candidate_a_v1", cfg, times)
        end = perf_counter()
        samples.append({**times, "import": imported - start, "total": end - start})
    rss = process.memory_info()
    return {
        "n": n,
        "edges": len(graph.edges),
        "payload_bytes": len(data),
        "outcome": out["components"][0]["decision"],
        "rss_before_bytes": before,
        "process_peak_working_set_bytes": getattr(rss, "peak_wset", None),
        "rss_after_bytes": rss.rss,
        "samples_seconds": samples,
        "median_seconds": {
            k: statistics.median(row.get(k, 0) for row in samples)
            for k in ("import", "proposal", "internal", "consensus", "total")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=int)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(measure(args.child)))
        return
    REPORT.mkdir(exist_ok=True, parents=True)
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    results = []
    for n in [22, 51, 76, 138, 225]:
        raw = subprocess.check_output(
            [sys.executable, __file__, "--child", str(n)], env=env
        )
        results.append(json.loads(raw))
    profile = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas_threads_requested": 1,
        "fixture": "rigid, seed 17, fixed sparse construction; median of 3 in fresh process per size",
        "rows": results,
    }
    (REPORT / "cpu_profile.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )
    cfg = policy()
    obj = fixture("contradiction")
    (REPORT / "synthetic_graph.json").write_bytes(encoded(obj))
    (REPORT / "synthetic_manifest.json").write_bytes(encoded(manifest(obj)))
    graph = as_graph(obj, cfg)
    for method in METHODS:
        (REPORT / f"example_{method}.json").write_bytes(
            encoded(decide(graph, method, cfg))
        )


if __name__ == "__main__":
    main()
