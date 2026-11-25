"""Only opaque identifiers and whitelisted pair observations cross this interface."""

from pathlib import Path
import json
import re
from types import MappingProxyType

from .rotation_geometry import checked_rotation
from .schemas import CONVENTION, Edge, Graph, digest, encoded, policy

ID = re.compile(r"image_[0-9]{4,10}(?:\.png)?\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
PROVENANCE = {"source_sha256", "config_sha256", "parent_sha256", "membership_sha256"}


def fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"expected exactly {sorted(names)}")


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def load_cache(path, trusted_manifest, config=None):
    return import_bytes(Path(path).read_bytes(), trusted_manifest, config)


def import_bytes(data, trusted_manifest, config=None):
    """Caller supplies an independently trusted manifest, never one read from this payload."""
    fields(trusted_manifest, PROVENANCE | {"payload_sha256"})
    if any(
        not isinstance(s, str) or not HEX.fullmatch(s)
        for s in trusted_manifest.values()
    ):
        raise ValueError("invalid manifest digest")
    if digest(data) != trusted_manifest["payload_sha256"]:
        raise ValueError("payload digest mismatch")
    obj = json.loads(data, object_pairs_hook=unique_keys)
    fields(obj, {"schema", "nodes", "edges", "mode", "convention", "provenance"})
    p = obj["provenance"]
    fields(p, PROVENANCE)
    if any(p[k] != trusted_manifest[k] for k in PROVENANCE):
        raise ValueError("cache provenance mismatch")
    if obj["schema"] != "geometry_graph_v1" or obj["convention"] != CONVENTION:
        raise ValueError("unsupported cache schema or frame convention")
    if obj["mode"] not in {"verified_counts", "topology_only"}:
        raise ValueError("invalid field completeness mode")
    if not isinstance(obj["nodes"], list) or not all(
        isinstance(i, str) and ID.fullmatch(i) for i in obj["nodes"]
    ):
        raise ValueError("invalid opaque node ID")
    nodes = tuple(sorted(obj["nodes"]))
    members = set(nodes)
    if len(nodes) != len(members) or digest(encoded(nodes)) != p["membership_sha256"]:
        raise ValueError("duplicate or incomplete node membership")
    if not isinstance(obj["edges"], list):
        raise ValueError("edge list required")
    tolerance = (config or policy())["rotation_matrix_atol"]
    edges = {}
    for row in obj["edges"]:
        required = {"i", "j", "inliers", "rotation_valid"}
        if isinstance(row, dict) and "rotation" in row:
            required.add("rotation")
        fields(row, required)
        i, j = row["i"], row["j"]
        if (
            not isinstance(i, str)
            or not isinstance(j, str)
            or i not in members
            or j not in members
            or i == j
        ):
            raise ValueError("invalid pair endpoints")
        count = row["inliers"]
        if obj["mode"] == "verified_counts":
            if type(count) is not int or not 0 < count <= 2**63 - 1:
                raise ValueError("invalid verified inlier count")
        elif count is not None or row["rotation_valid"]:
            raise ValueError("topology-only cache cannot declare measurements")
        raw = row.get("rotation")
        if type(row["rotation_valid"]) is not bool or row["rotation_valid"] != (
            raw is not None
        ):
            raise ValueError("rotation validity mask conflicts with measurement")
        r = None if raw is None else checked_rotation(raw, tolerance)
        if i > j:
            i, j = j, i
            r = None if r is None else r.T
        if (i, j) in edges:
            raise ValueError("duplicate undirected pair")
        edges[i, j] = Edge(i, j, count, r)
    return Graph(
        nodes,
        tuple(edges[k] for k in sorted(edges)),
        MappingProxyType(dict(p)),
        trusted_manifest["payload_sha256"],
        obj["mode"],
    )
