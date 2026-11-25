"""Validate a returned mask against the exact parent and ordered pair inventory."""

from .schemas import digest, encoded


def checked_keep_mask(receipt, graph, config):
    if (
        receipt.get("schema") != "geometry_mask_v1"
        or receipt.get("method") not in config["methods"]
    ):
        raise ValueError("unknown mask schema or method")
    if receipt.get("input_sha256") != graph.payload_sha256 or receipt.get(
        "policy_sha256"
    ) != digest(encoded(config)):
        raise ValueError("mask was generated for a different input or configuration")
    pairs = [list(edge.pair) for edge in graph.edges]
    mask = receipt.get("keep")
    if (
        receipt.get("pairs") != pairs
        or not isinstance(mask, list)
        or len(mask) != len(pairs)
        or any(type(value) is not bool for value in mask)
    ):
        raise ValueError("mask pair order, cardinality or type mismatch")
    if receipt.get("mask_sha256") != digest(encoded([pairs, mask])):
        raise ValueError("mask checksum mismatch")
    return tuple(mask)
