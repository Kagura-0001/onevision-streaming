from __future__ import annotations

from typing import Any

from .client import ClientSession, OVClient, OVResponseError, OVSessionError
from .types import Event

_LAZY_EXPORTS = {
    "AsyncLocalSession": ("ov_streaming.engine", "AsyncLocalSession"),
    "AsyncStreamingEngine": ("ov_streaming.engine", "AsyncStreamingEngine"),
    "CodecConfig": ("ov_streaming.codec", "CodecConfig"),
    "CodecWindow": ("ov_streaming.codec", "CodecWindow"),
    "LocalSession": ("ov_streaming.engine", "LocalSession"),
    "StreamingEngine": ("ov_streaming.engine", "StreamingEngine"),
    "TransformersBackend": ("ov_streaming.transformers_backend", "TransformersBackend"),
    "VideoFrame": ("ov_streaming.media", "VideoFrame"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attr_name = target
    return getattr(import_module(module_name), attr_name)


__all__ = [
    "AsyncLocalSession",
    "AsyncStreamingEngine",
    "ClientSession",
    "CodecConfig",
    "CodecWindow",
    "Event",
    "LocalSession",
    "OVClient",
    "OVResponseError",
    "OVSessionError",
    "StreamingEngine",
    "TransformersBackend",
    "VideoFrame",
]
