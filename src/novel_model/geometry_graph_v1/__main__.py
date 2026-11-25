"""Read a cache and a separately trusted manifest. Write a versioned JSON mask."""

import argparse
import json
from pathlib import Path
from .schemas import METHODS, policy
from .graph_import import load_cache
from .decision_engine import decide


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg = policy()
    graph = load_cache(args.cache, json.loads(args.manifest.read_bytes()), cfg)
    result = decide(graph, args.method, cfg)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, allow_nan=False, indent=2)


if __name__ == "__main__":
    main()
