"""CPU-only scene-boundary prototype. Use load_cache and decide as entry points."""

from .graph_import import load_cache
from .decision_engine import decide

__all__ = ["decide", "load_cache"]
