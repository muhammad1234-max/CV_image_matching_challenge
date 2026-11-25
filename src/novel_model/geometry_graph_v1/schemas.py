"""Validated graph and policy types. Node order is canonical opaque-ID order."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import hashlib
import json

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/novel_model/geometry_graph_v1.yaml"
PROPOSAL_DIGEST = "b0fb39e016e84428daada0b48bad89ca99153c05bfcc61f40d91ed9300a842b9"
CONVENTION = "target_camera_from_source_camera"
METHODS = (
    "pass_through_v1",
    "candidate_a_v1",
    "normalized_cut_v1",
    "triangle_v1",
    "per_edge_consensus_v1",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


@dataclass(frozen=True)
class Edge:
    i: str
    j: str
    count: int | None
    rotation: np.ndarray | None

    @property
    def pair(self):
        return (self.i, self.j)

    @property
    def weight(self):
        if self.count is None:
            raise ValueError("per-pair inlier count unavailable")
        return float(np.log1p(float(self.count)))


@dataclass(frozen=True)
class Graph:
    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]
    provenance: Mapping[str, str]
    payload_sha256: str
    mode: str


def policy(path=CONFIG):
    config = yaml.safe_load(Path(path).read_bytes())
    proposal = Path(path).with_name(config["proposal"]).read_bytes()
    if (
        digest(proposal) != PROPOSAL_DIGEST
        or config["proposal_sha256"] != PROPOSAL_DIGEST
    ):
        raise ValueError("phase 4A proposal pin mismatch")
    if (
        config["schema"] != "geometry_graph_v1"
        or set(config["methods"]) != set(METHODS)
        or config["default_method"] not in METHODS
    ):
        raise ValueError("unsupported method configuration")
    return {**yaml.safe_load(proposal), **config}
