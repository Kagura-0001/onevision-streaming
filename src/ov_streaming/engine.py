"""In-process Python API for single-script LLaVA-OneVision-2.5/2.1 streaming.

Two backends behind one session interface:

- ``backend="vllm"``: concurrent Fork-and-Commit on official ``vllm==0.25.1``
  through ``ov_streaming.vllm_plugin`` (or a fork with native branching).
- ``backend="transformers"``: sequential Fork-and-Commit with plain
  PyTorch/Transformers through ``TransformersBackend``.

Both share ``OVSession`` and the OV renderers. Sessions accept video files,
live URLs, external frames and live H.264 bytes; see ``local_session``.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from .local_session import AsyncLocalSession, LocalSession, _LoopThread
from .renderers import Renderer, load_renderer
from .session import OVSession

DEFAULT_MAX_PIXELS = 258048


class AsyncStreamingEngine:
    """In-process streaming engine hosting vLLM or Transformers directly."""

    def __init__(
        self,
        model: str | Path,
        *,
        backend: Literal["vllm", "transformers"] = "vllm",
        renderer: str = "ov25",
        max_model_len: int | None = 32768,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        engine_kwargs: Mapping[str, Any] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        _backend_instance: Any = None,
        _renderer_instance: Renderer | None = None,
    ) -> None:
        self.model = str(model)
        self.backend_name = backend
        self.renderer_name = renderer
        self._closed = False
        self._vllm_engine: Any = None

        if _backend_instance is not None and _renderer_instance is not None:
            self.backend = _backend_instance
            self.renderer = _renderer_instance
            return

        renderer_class = load_renderer(renderer)

        if backend == "vllm":
            os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
            from .vllm_plugin import register as register_vllm_plugin

            register_vllm_plugin()

            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.tokenizers import cached_tokenizer_from_config
            from vllm.v1.engine.async_llm import AsyncLLM

            from .backend import VLLMBackend

            merged_args: dict[str, Any] = {
                "async_scheduling": False,
                **renderer_class.engine_args,
            }
            if max_model_len is not None:
                merged_args["max_model_len"] = max_model_len
            mm_kwargs = dict(merged_args.get("mm_processor_kwargs") or {})
            mm_kwargs["max_pixels"] = max_pixels
            merged_args["mm_processor_kwargs"] = mm_kwargs

            if engine_kwargs:
                for key, val in engine_kwargs.items():
                    if isinstance(val, Mapping) and isinstance(
                        merged_args.get(key), Mapping
                    ):
                        merged_args[key] = {**merged_args[key], **val}
                    else:
                        merged_args[key] = val

            engine_args = AsyncEngineArgs(model=self.model, **merged_args)
            model_config = engine_args.create_model_config()
            tokenizer = cached_tokenizer_from_config(model_config)
            self.renderer = renderer_class(tokenizer, model_config)
            self._vllm_engine = AsyncLLM.from_engine_args(engine_args)
            self.backend = VLLMBackend(self._vllm_engine)
        elif backend == "transformers":
            from .transformers_backend import TransformersBackend

            tf_backend = TransformersBackend.from_pretrained(
                self.model,
                max_model_len=max_model_len,
                **(dict(model_kwargs) if model_kwargs else {}),
            )
            self.backend = tf_backend
            if hasattr(renderer_class, "from_pretrained"):
                self.renderer = renderer_class.from_pretrained(
                    self.model,
                    tokenizer=tf_backend.tokenizer,
                    hf_config=tf_backend.model.config,
                    max_pixels=max_pixels,
                )
            else:
                from types import SimpleNamespace

                model_config = SimpleNamespace(
                    model=self.model,
                    hf_config=tf_backend.model.config,
                    multimodal_config=SimpleNamespace(
                        mm_processor_kwargs={"max_pixels": max_pixels}
                    ),
                )
                self.renderer = renderer_class(tf_backend.tokenizer, model_config)
        else:
            raise ValueError(f"unsupported backend {backend!r}; use 'vllm' or 'transformers'")

    def session(
        self,
        *,
        session_id: str | None = None,
        enable: Sequence[str] | None = None,
        max_responses: int = 3,
        max_response_tokens: int | None = None,
        max_caption_tokens: int | None = None,
        caption_prompt: str | None = None,
        max_visual_frames: int | None = None,
        compact_to_visual_frames: int | None = None,
        pin_first_question: bool = False,
        answer_commit: Literal["on_done", "client"] = "on_done",
        verification: str | None = None,
        verification_timeout: float = 10.0,
    ) -> AsyncLocalSession:
        ov_session = OVSession(
            self.backend,
            self.renderer,
            session_id or uuid.uuid4().hex,
            enable=enable,
            max_responses=max_responses,
            max_response_tokens=max_response_tokens,
            max_caption_tokens=max_caption_tokens,
            caption_prompt=caption_prompt,
            max_visual_frames=max_visual_frames,
            compact_to_visual_frames=compact_to_visual_frames,
            pin_first_question=pin_first_question,
            answer_commit=answer_commit,
            verification=verification,
            verification_timeout=verification_timeout,
        )
        return AsyncLocalSession(ov_session)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._vllm_engine is not None:
            self._vllm_engine.shutdown()
        elif hasattr(self.backend, "close"):
            self.backend.close()

    async def __aenter__(self) -> AsyncStreamingEngine:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

class StreamingEngine:
    """Synchronous single-script entry point for OV2.5/2.1 streaming inference."""

    def __init__(
        self,
        model: str | Path,
        *,
        backend: Literal["vllm", "transformers"] = "vllm",
        renderer: str = "ov25",
        max_model_len: int | None = 32768,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        engine_kwargs: Mapping[str, Any] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        _backend_instance: Any = None,
        _renderer_instance: Renderer | None = None,
    ) -> None:
        self._loop = _LoopThread()
        self._closed = False
        try:
            self._inner = self._loop.run(
                self._create_inner(
                    model,
                    backend=backend,
                    renderer=renderer,
                    max_model_len=max_model_len,
                    max_pixels=max_pixels,
                    engine_kwargs=engine_kwargs,
                    model_kwargs=model_kwargs,
                    _backend_instance=_backend_instance,
                    _renderer_instance=_renderer_instance,
                )
            )
        except BaseException:
            self._loop.stop()
            raise

    @staticmethod
    async def _create_inner(*args: Any, **kwargs: Any) -> AsyncStreamingEngine:
        return AsyncStreamingEngine(*args, **kwargs)

    def session(self, **options: Any) -> LocalSession:
        inner_session = self._inner.session(**options)
        return LocalSession(self._loop, inner_session)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._loop.run(self._inner.aclose())
        finally:
            self._loop.stop()

    def __enter__(self) -> StreamingEngine:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "AsyncLocalSession",
    "AsyncStreamingEngine",
    "DEFAULT_MAX_PIXELS",
    "LocalSession",
    "StreamingEngine",
]
