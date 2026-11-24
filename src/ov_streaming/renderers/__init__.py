from importlib import import_module

from .base import Renderer

RENDERERS = {
    "univia": "ov_streaming.renderers.univia:UniVIARenderer",
    "ov21-window": "ov_streaming.renderers.ov21:OV21WindowRenderer",
    "ov21": "ov_streaming.renderers.ov21:OV21Renderer",
    "ov25-window": "ov_streaming.renderers.ov21:OV21WindowRenderer",
    "ov25": "ov_streaming.renderers.ov21:OV21Renderer",
}


def load_renderer(name: str) -> type[Renderer]:
    path = RENDERERS.get(name, name)
    if ":" not in path:
        raise ValueError(
            f"unknown renderer {name!r}; use {', '.join(RENDERERS)} or module:Class"
        )
    module, _, cls = path.partition(":")
    return getattr(import_module(module), cls)


__all__ = ["RENDERERS", "Renderer", "load_renderer"]
