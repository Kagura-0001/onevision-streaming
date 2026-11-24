from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer, Qwen2VLImageProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from .._checkpoint import (
    is_local_checkpoint as _is_local_checkpoint,
    load_dynamic_ref,
    module_candidates_from_auto_map,
)
from ..types import VisualFrame
from .base import Renderer


def _video_processor_module_candidates(hf_config: Any, checkpoint: str = "") -> list[str]:
    model_type = getattr(hf_config, "model_type", "")
    fallbacks = (
        ("video_processing_llava_onevision2_5", "video_processing_llava_onevision2")
        if model_type == "llava_onevision2_5"
        else ("video_processing_llava_onevision2", "video_processing_llava_onevision2_5")
    )
    return module_candidates_from_auto_map(
        checkpoint,
        auto_map_keys=("AutoVideoProcessor",),
        fallback_modules=fallbacks,
        hf_config=hf_config,
    )


def _load_build_patch_positions(
    checkpoint: str, hf_config: Any, *, local_files_only: bool
) -> Any:
    refs = [
        f"{mod}.build_patch_positions"
        for mod in _video_processor_module_candidates(hf_config, checkpoint)
    ]
    return load_dynamic_ref(
        checkpoint,
        refs,
        loader=lambda ref, ckpt, **_: get_class_from_dynamic_module(
            ref, ckpt, local_files_only=local_files_only
        ),
    )


class OV21Renderer(Renderer):
    engine_args = {
        "hf_overrides": {
            "model_type": "llava_onevision2_qwen35",
            "architectures": ["LlavaOnevision2Qwen35ForConditionalGeneration"],
        },
        "trust_remote_code": True,
        "enable_mm_embeds": True,
        "mamba_cache_mode": "none",
        # Training keeps the GatedDeltaNet state in fp32; a bf16 cache rounds it
        # at every append and every decoded token.
        "mamba_ssm_cache_dtype": "float32",
        "enable_prefix_caching": False,
        "limit_mm_per_prompt": {"image": 1024, "video": 0},
    }
    controls = {
        "silence": "<silence>",
        "response": "<response>",
        "caption": "<captioning>",
    }
    memory_tags = {**Renderer.memory_tags, "caption": "answer_memory"}
    optional_controls = {"caption": "<caption_begin>\n"}
    fallback_fps = 1000.0  # Preserve the original renderer fallback.
    system_prompt = "You are a real-time streaming video assistant."
    timestep_millis = True

    def __init__(self, tokenizer, model_config):
        if len(tokenizer.encode("<captioning>", add_special_tokens=False)) != 1:
            # Some OV2.1 checkpoints are trained without captioning.
            self.controls = {k: v for k, v in self.controls.items() if k != "caption"}
        super().__init__(tokenizer, model_config)
        checkpoint = str(model_config.model)
        local_files_only = _is_local_checkpoint(checkpoint)
        self.image_processor = Qwen2VLImageProcessor.from_pretrained(
            checkpoint, local_files_only=local_files_only
        )
        self.build_patch_positions = _load_build_patch_positions(
            checkpoint,
            model_config.hf_config,
            local_files_only=local_files_only,
        )
        self.merge_size = model_config.hf_config.vision_config.spatial_merge_size
        self.patch_size = model_config.hf_config.vision_config.patch_size
        multimodal_config = getattr(model_config, "multimodal_config", None)
        kwargs = (
            getattr(multimodal_config, "mm_processor_kwargs", None) or {}
            if multimodal_config is not None
            else {}
        )
        self.max_pixels = kwargs.get("max_pixels", 200000)

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path,
        *,
        tokenizer: Any = None,
        hf_config: Any = None,
        max_pixels: int = 200000,
    ) -> OV21Renderer:
        """Build a renderer directly from a local path or Hugging Face repo ID."""
        checkpoint = str(model)
        local_files_only = _is_local_checkpoint(checkpoint)
        if hf_config is None:
            hf_config = AutoConfig.from_pretrained(
                checkpoint,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
        model_config = SimpleNamespace(
            model=checkpoint,
            hf_config=hf_config,
            multimodal_config=SimpleNamespace(
                mm_processor_kwargs={"max_pixels": max_pixels}
            ),
        )
        return cls(tokenizer, model_config)

    def render_frame(self, image, timestamp_sec, *, frame_index=None, fps=None):
        # Without the source fps, frame indices count milliseconds.
        fps = fps or self.fallback_fps
        if frame_index is None:
            frame_index = round(timestamp_sec * fps)
        data = dict(
            self.image_processor(
                images=[image],
                min_pixels=4096,
                max_pixels=self.max_pixels,
                return_tensors="pt",
            )
        )
        data["patch_positions"] = self.build_patch_positions(
            data["image_grid_thw"],
            spatial_merge_size=self.merge_size,
            frame_indices=[torch.tensor([frame_index])],
        )
        pads = int(data["image_grid_thw"].prod()) // self.merge_size**2
        timestamp_sec = frame_index / fps
        text = (
            f"<{timestamp_sec:.1f} seconds><|vision_start|>"
            f"{'<|image_pad|>' * pads}<|vision_end|>\n"
        )
        return VisualFrame(text, data)

    def render_codec_frame(
        self, image, timestamp_sec, *, codec, frame_index=None, fps=None
    ):
        """Consume row-major source positions without resizing a packed canvas."""
        if not isinstance(codec, dict) or codec.get("layout") != "row_major":
            raise ValueError("codec.layout must be row_major")
        if codec.get("patch_size") != self.patch_size:
            raise ValueError("codec patch_size does not match the checkpoint")
        if fps is None or not math.isfinite(fps) or fps <= 0:
            raise ValueError("codec canvases require positive finite source fps")
        if not math.isfinite(timestamp_sec) or timestamp_sec < 0:
            raise ValueError("codec timestamp_sec must be finite and non-negative")
        width, height = image.size
        factor = self.patch_size * self.merge_size
        if width % factor or height % factor or width * height > self.max_pixels:
            raise ValueError("codec canvas must fit max_pixels and the merge grid")
        h, w = height // self.patch_size, width // self.patch_size
        raw = codec.get("patch_positions")
        if not isinstance(raw, list) or len(raw) != h * w:
            raise ValueError("codec patch_positions must match the canvas grid")
        if any(
            not isinstance(row, list)
            or len(row) != 3
            or any(type(v) is not int or v < 0 or v > 2**31 - 1 for v in row)
            for row in raw
        ):
            raise ValueError("codec positions must be non-negative integer triplets")
        positions = torch.tensor(raw, dtype=torch.int64)
        if (positions[:, 1] >= h).any() or (positions[:, 2] >= w).any():
            raise ValueError("codec source coordinates exceed the source grid")
        # Compare in frames: fps is r_frame_rate rounded to six decimals, so a
        # frame exactly at the cutoff (13978 at 13978/583 fps = 583 s) can land
        # a few microseconds past it in seconds. A real next frame is a whole
        # frame later.
        if int(positions[:, 0].max()) > timestamp_sec * fps + 0.05:
            raise ValueError("codec positions contain a future source frame")
        m = self.merge_size
        positions = (
            positions.reshape(h // m, m, w // m, m, 3)
            .permute(0, 2, 1, 3, 4)
            .reshape(-1, 3)
        )
        # One merged token must represent one contiguous source block, exactly
        # as in training. A different merge size cannot silently reuse a canvas.
        blocks = positions.reshape(-1, m * m, 3)
        offsets = torch.tensor([[y, x] for y in range(m) for x in range(m)])
        if (
            not torch.equal(blocks[:, :, 0], blocks[:, :1, 0].expand(-1, m * m))
            or not (blocks[:, :, 1:] - blocks[:, :1, 1:] == offsets).all()
        ):
            raise ValueError(
                "codec source blocks do not match the checkpoint merge size"
            )
        data = dict(
            self.image_processor(images=[image], do_resize=False, return_tensors="pt")
        )
        if data["image_grid_thw"].tolist() != [[1, h, w]]:
            raise ValueError("image processor changed the codec canvas grid")
        data["patch_positions"] = positions
        # Frozen training schema: one wrapper per complete canvas, with a
        # point for one source frame and a range for several source frames.
        # Keep patch t-coordinates intact: the label does not change RoPE.
        start, end = (
            int(v) / fps for v in (positions[:, 0].min(), positions[:, 0].max())
        )
        start, end = round(start, 1), round(end, 1)
        if int(positions[:, 0].min()) == int(positions[:, 0].max()):
            timestamp = f"<{start:.1f} seconds>"
        else:
            timestamp = f"<{start:.1f} to {end:.1f} seconds>"
        text = (
            timestamp
            + "<|vision_start|>"
            + "<|image_pad|>" * (h * w // (m * m))
            + "<|vision_end|>\n"
        )
        return VisualFrame(text, data)

    def render_user(self, turn):
        frames = "".join(frame.text for frame in turn.frames)
        return (
            "".join(f"{self.render_memory(memory)}\n" for memory in turn.questions)
            + (frames or f"<{turn.timestamp_sec:.3f} seconds>\n")
            + "".join(f"{self.render_memory(memory)}\n" for memory in turn.memories)
        )

    def render_archive(self, archive):
        # No newline after the last block: the first kept turn follows directly.
        return "\n".join(map(self.render_memory, archive))


class OV21WindowRenderer(OV21Renderer):
    """OV21 with the archive written as a sliding-window evaluation writes it.

    lmms-eval's streaming protocol keeps the newest frames and puts the
    questions that left the window ahead of the oldest kept frame, each ending
    with a newline like a question inside a turn. A session with
    max_visual_frames and compact_to_visual_frames both set to the window then
    builds that protocol's prompt, token for token, at every decision.
    """

    # Evaluation uses source-frame indices; prefer the actual source fps.
    fallback_fps = 30.0

    def render_archive(self, archive):
        return "".join(f"{self.render_memory(memory)}\n" for memory in archive)
