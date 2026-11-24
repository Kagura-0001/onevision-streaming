"""Zero-vLLM PyTorch/Transformers backend for OVStreaming.

Implements the same backend contract as ``VLLMBackend`` (``open_source``,
``add_branch``, ``abort``, ``count_tokens``, ``max_model_len``) so it plugs
directly into ``OVSession`` and ``OV21Renderer`` with identical prompt,
timestamp, ``<answer_memory>``, compaction, and codec I+P semantics.

Unlike the vLLM backend, branching here clones the current ``DynamicCache``
(Full-Attention KV + GatedDeltaNet ``conv_states`` / ``recurrent_states``) and
decodes sequentially under a single model lock. This gives zero-dependency
streaming inference on standard PyTorch + Transformers environments without
requiring vLLM.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from ._checkpoint import is_local_checkpoint as _is_local_checkpoint
from ._checkpoint import load_mrope_positions_fn
from .types import BackendOutput, VisualFrame


def _load_mrope_positions_fn(model_ref: str, hf_config: Any) -> Any | None:
    """The checkpoint's M-RoPE helper, or None to let the model compute positions."""
    try:
        return load_mrope_positions_fn(model_ref, hf_config)
    except Exception:
        return None


def _clone_tensor_or_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [_clone_tensor_or_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_tensor_or_value(v) for v in value)
    if isinstance(value, dict):
        return {k: _clone_tensor_or_value(v) for k, v in value.items()}
    return copy.deepcopy(value)


def clone_past_key_values(past_key_values: Any) -> Any:
    """Create an independent GPU/CPU copy of a Transformers hybrid cache.

    Handles Transformers 5.x ``DynamicCache`` containing both full-attention
    layers (``keys``/``values``) and linear-attention / GDN layers
    (``conv_states``/``recurrent_states``), as well as legacy tuple caches.
    """
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "layers") and isinstance(past_key_values.layers, list):
        cloned = copy.copy(past_key_values)
        new_layers = []
        for layer in past_key_values.layers:
            layer_copy = copy.copy(layer)
            if hasattr(layer, "__dict__"):
                for attr, val in layer.__dict__.items():
                    setattr(layer_copy, attr, _clone_tensor_or_value(val))
            new_layers.append(layer_copy)
        cloned.layers = new_layers
        return cloned
    return _clone_tensor_or_value(past_key_values)


def _infer_model_device(model: Any) -> torch.device:
    if hasattr(model, "device") and model.device is not None:
        dev = torch.device(model.device)
        if dev.type != "meta":
            return dev
    try:
        param = next(model.parameters())
        if param.device.type != "meta":
            return param.device
    except StopIteration:
        pass
    return torch.device("cpu")


def _extract_stop_token_ids(model: Any, tokenizer: Any) -> set[int]:
    stop_ids: set[int] = set()
    for source in (
        getattr(model, "generation_config", None),
        getattr(model, "config", None),
        tokenizer,
    ):
        eos = getattr(source, "eos_token_id", None)
        if isinstance(eos, int):
            stop_ids.add(eos)
        elif isinstance(eos, (list, tuple, set)):
            stop_ids.update(int(x) for x in eos if x is not None)
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(im_end) == 1:
        stop_ids.add(int(im_end[0]))
    return stop_ids


class TransformersBackend:
    """In-process PyTorch/Transformers streaming backend (no vLLM required)."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_model_len: int | None = None,
        mrope_positions_fn: Any | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = (
            torch.device(device) if device is not None else _infer_model_device(model)
        )
        self._mrope_positions_fn = mrope_positions_fn
        self._stop_token_ids = _extract_stop_token_ids(model, tokenizer)
        self._lock = asyncio.Lock()
        self._sources: dict[str, TransformersSourceRequest] = {}
        self._branches: dict[str, TransformersBranch] = {}

        if max_model_len is not None:
            self.max_model_len = max_model_len
        else:
            text_cfg = getattr(getattr(model, "config", None), "text_config", None)
            self.max_model_len = getattr(
                text_cfg,
                "max_position_embeddings",
                getattr(getattr(model, "config", None), "max_position_embeddings", None),
            )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        dtype: torch.dtype | str = torch.bfloat16,
        device_map: Any = "auto",
        attn_implementation: str | None = None,
        max_model_len: int | None = None,
        **model_kwargs: Any,
    ) -> TransformersBackend:
        checkpoint = str(model_name_or_path)
        local_files_only = _is_local_checkpoint(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "dtype": dtype,
            "device_map": device_map,
            **model_kwargs,
        }
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForImageTextToText.from_pretrained(checkpoint, **load_kwargs)
        model.eval()
        mrope_fn = _load_mrope_positions_fn(checkpoint, model.config)
        return cls(
            model,
            tokenizer,
            max_model_len=max_model_len,
            mrope_positions_fn=mrope_fn,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def open_source(
        self, request_id: str, *, cache_salt: str
    ) -> TransformersSourceRequest:
        source = TransformersSourceRequest(self, request_id, cache_salt)
        self._sources[request_id] = source
        return source

    async def add_branch(
        self,
        source_request_id: str,
        request_id: str,
        *,
        suffix_token_ids: Sequence[int],
        max_tokens: int,
    ) -> TransformersBranch:
        source = self._sources.get(source_request_id)
        if source is None:
            raise ValueError(f"branch source {source_request_id!r} does not exist")
        if source.num_tokens <= 0:
            raise RuntimeError("branch source has no computed prefix")

        async with self._lock:
            cloned_cache = await asyncio.to_thread(clone_past_key_values, source.cache)
            snapshot_len = source.num_tokens
            snapshot_delta = source.rope_delta
            use_3d_pos = source.uses_3d_positions

        branch = TransformersBranch(
            self,
            request_id=request_id,
            cache=cloned_cache,
            prefix_len=snapshot_len,
            rope_delta=snapshot_delta,
            uses_3d_positions=use_3d_pos,
            suffix_token_ids=tuple(int(t) for t in suffix_token_ids),
            max_tokens=max_tokens,
        )
        self._branches[request_id] = branch
        return branch

    async def abort(self, request_id: str) -> None:
        branch = self._branches.pop(request_id, None)
        if branch is not None:
            branch.cancel()
        source = self._sources.pop(request_id, None)
        if source is not None:
            source.release()

    def close(self) -> None:
        for branch in list(self._branches.values()):
            branch.cancel()
        self._branches.clear()
        for source in list(self._sources.values()):
            source.release()
        self._sources.clear()


class TransformersSourceRequest:
    """Persistent incremental source state backed by a Transformers hybrid cache."""

    def __init__(
        self, backend: TransformersBackend, request_id: str, cache_salt: str
    ) -> None:
        self.backend = backend
        self.request_id = request_id
        self.cache_salt = cache_salt
        self.cache: Any = None
        self.prompt_token_ids: list[int] = []
        self.patch_positions: list[torch.Tensor] = []
        self.num_tokens: int = 0
        self.rope_delta: int = 0
        self.uses_3d_positions: bool = False
        self._closed = False

    def _compute_chunk_positions(
        self,
        new_token_ids: list[int],
        new_patch_positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int, bool]:
        past_len = len(self.prompt_token_ids)
        chunk_len = len(new_token_ids)
        all_tokens = self.prompt_token_ids + new_token_ids
        all_patches = (
            [*self.patch_positions, new_patch_positions]
            if new_patch_positions is not None
            else list(self.patch_positions)
        )
        mrope_fn = self.backend._mrope_positions_fn
        config = getattr(self.backend.model, "config", None)
        if mrope_fn is not None and config is not None and all_patches:
            vision_cfg = getattr(config, "vision_config", None)
            merge_size = getattr(vision_cfg, "spatial_merge_size", 3)
            image_token_id = getattr(config, "image_token_id", -1)
            video_token_id = getattr(config, "video_token_id", -1)
            full_pos, delta = mrope_fn(
                torch.tensor([all_tokens], dtype=torch.long),
                None,
                torch.cat(all_patches, dim=0),
                merge_size,
                image_token_id,
                video_token_id,
            )
            chunk_pos = full_pos[:, :, past_len : past_len + chunk_len].contiguous()
            return chunk_pos, int(delta.item()), True

        # Text-only prefix (or model without custom M-RoPE helper).
        if self.uses_3d_positions:
            base = torch.arange(past_len, past_len + chunk_len, dtype=torch.long)
            chunk_pos = (
                (base + self.rope_delta).view(1, 1, -1).expand(3, 1, -1).contiguous()
            )
            return chunk_pos, self.rope_delta, True
        chunk_pos = torch.arange(
            past_len, past_len + chunk_len, dtype=torch.long
        ).unsqueeze(0)
        return chunk_pos, 0, False

    def _run_step_sync(
        self,
        new_token_ids: list[int],
        frames: Sequence[VisualFrame],
        allowed_token_ids: Sequence[int],
    ) -> int:
        if self._closed:
            raise RuntimeError("source request is closed")

        device = self.backend.device
        past_len = len(self.prompt_token_ids)
        chunk_len = len(new_token_ids)

        mm_kwargs: dict[str, Any] = {}
        new_patch_positions: torch.Tensor | None = None
        if frames:
            frame_dicts = [frame.data for frame in frames]
            if isinstance(frame_dicts[0], Mapping):
                batched = {
                    key: torch.cat([item[key] for item in frame_dicts], dim=0)
                    for key in frame_dicts[0]
                }
                if "patch_positions" in batched:
                    new_patch_positions = batched["patch_positions"].detach().cpu()
                for key, tensor in batched.items():
                    if isinstance(tensor, torch.Tensor):
                        mm_kwargs[key] = tensor.to(device)
                    else:
                        mm_kwargs[key] = tensor

        position_ids, rope_delta, uses_3d = self._compute_chunk_positions(
            new_token_ids, new_patch_positions
        )
        input_ids = torch.tensor([new_token_ids], dtype=torch.long, device=device)
        cache_position = torch.arange(
            past_len, past_len + chunk_len, dtype=torch.long, device=device
        )

        forward_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "position_ids": position_ids.to(device),
            "cache_position": cache_position,
            "past_key_values": self.cache,
            "use_cache": True,
            **mm_kwargs,
        }
        with torch.inference_mode():
            outputs = self.backend.model(**forward_kwargs)

        self.cache = outputs.past_key_values
        self.prompt_token_ids.extend(new_token_ids)
        if new_patch_positions is not None:
            self.patch_positions.append(new_patch_positions)
        self.num_tokens = len(self.prompt_token_ids)
        self.rope_delta = rope_delta
        self.uses_3d_positions = uses_3d

        last_logits = outputs.logits[0, -1]
        allowed = [int(tid) for tid in allowed_token_ids]
        allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=last_logits.device)
        best_idx = int(torch.argmax(last_logits.index_select(0, allowed_tensor)).item())
        return allowed[best_idx]

    async def step(
        self,
        prompt: str,
        frames: Sequence[VisualFrame],
        *,
        allowed_token_ids: Sequence[int],
    ) -> int:
        new_token_ids = list(
            self.backend.tokenizer.encode(prompt, add_special_tokens=False)
        )
        if not new_token_ids:
            raise ValueError("step prompt must encode to at least one token")
        async with self.backend._lock:
            return await asyncio.to_thread(
                self._run_step_sync, new_token_ids, frames, allowed_token_ids
            )

    def release(self) -> None:
        self._closed = True
        self.cache = None
        self.prompt_token_ids.clear()
        self.patch_positions.clear()

    async def close(self) -> None:
        self.backend._sources.pop(self.request_id, None)
        self.release()

    async def abort(self) -> None:
        self.backend._sources.pop(self.request_id, None)
        self.release()


class TransformersBranch:
    """Autoregressive branch decoding from a cloned Transformers hybrid cache."""

    def __init__(
        self,
        backend: TransformersBackend,
        *,
        request_id: str,
        cache: Any,
        prefix_len: int,
        rope_delta: int,
        uses_3d_positions: bool,
        suffix_token_ids: tuple[int, ...],
        max_tokens: int,
    ) -> None:
        self.backend = backend
        self.request_id = request_id
        self.cache = cache
        self.cur_len = prefix_len
        self.rope_delta = rope_delta
        self.uses_3d_positions = uses_3d_positions
        self.suffix_token_ids = suffix_token_ids
        self.max_tokens = max_tokens
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.cache = None

    def _make_text_positions(self, start: int, length: int) -> torch.Tensor:
        device = self.backend.device
        base = torch.arange(start, start + length, dtype=torch.long, device=device)
        if self.uses_3d_positions:
            return (
                (base + self.rope_delta).view(1, 1, -1).expand(3, 1, -1).contiguous()
            )
        return base.unsqueeze(0)

    def _decode_one_step_sync(self, token_ids: Sequence[int]) -> int:
        device = self.backend.device
        length = len(token_ids)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        position_ids = self._make_text_positions(self.cur_len, length)
        cache_position = torch.arange(
            self.cur_len, self.cur_len + length, dtype=torch.long, device=device
        )
        with torch.inference_mode():
            outputs = self.backend.model(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=self.cache,
                use_cache=True,
            )
        self.cache = outputs.past_key_values
        self.cur_len += length
        return int(torch.argmax(outputs.logits[0, -1]).item())

    async def outputs(self) -> AsyncIterator[BackendOutput]:
        generated_ids: list[int] = []
        emitted_text = ""
        try:
            if not self.suffix_token_ids:
                raise ValueError(
                    "TransformersBranch requires non-empty suffix_token_ids"
                )
            next_input: Sequence[int] = self.suffix_token_ids
            for step_index in range(self.max_tokens):
                if self._cancelled:
                    return
                async with self.backend._lock:
                    if self._cancelled:
                        return
                    next_token = await asyncio.to_thread(
                        self._decode_one_step_sync, next_input
                    )
                if next_token in self.backend._stop_token_ids:
                    yield BackendOutput("", "stop")
                    return
                generated_ids.append(next_token)
                decoded = self.backend.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                # Hold back incomplete UTF-8 replacement sequences until completed.
                delta = ""
                if not decoded.endswith("\ufffd"):
                    delta = decoded[len(emitted_text) :]
                    emitted_text = decoded
                finish_reason = (
                    "length" if step_index + 1 >= self.max_tokens else None
                )
                if finish_reason == "length" and not delta and len(decoded) > len(emitted_text):
                    delta = decoded[len(emitted_text) :]
                    emitted_text = decoded
                yield BackendOutput(delta, finish_reason)
                if finish_reason is not None:
                    return
                next_input = (next_token,)
        finally:
            self.backend._branches.pop(self.request_id, None)
            self.cache = None


__all__ = [
    "TransformersBackend",
    "TransformersBranch",
    "TransformersSourceRequest",
    "clone_past_key_values",
]
