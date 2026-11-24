from __future__ import annotations

from qwen_vl_utils import fetch_image

from ..types import VisualFrame
from .base import Renderer


class UniVIARenderer(Renderer):
    """The ms-swift template ``qwen3_vl_ov_source_policy`` used in training."""

    engine_args = {
        "limit_mm_per_prompt": {"image": 1024, "video": 0},
        "mm_processor_kwargs": {"min_pixels": 4096, "max_pixels": 131072},
    }
    controls = {"silence": "<silence>", "response": "</response>"}

    def __init__(self, tokenizer, model_config):
        super().__init__(tokenizer, model_config)
        kwargs = model_config.multimodal_config.mm_processor_kwargs
        self.pixels = {key: kwargs[key] for key in ("min_pixels", "max_pixels")}
        self.patch_size = model_config.hf_config.vision_config.patch_size

    def render_frame(self, image, timestamp_sec, **_):
        # Training resizes with this call; the vLLM processor, given the same
        # pixel bounds, then keeps the size.
        image = fetch_image(
            {"image": image, **self.pixels}, image_patch_size=self.patch_size
        )
        text = (
            f"<{timestamp_sec:.1f} seconds>"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
        )
        return VisualFrame(text, image)

    def render_user(self, turn):
        # Answers committed since the last turn go ahead of the frames.
        return (
            "".join(map(self.render_memory, turn.memories))
            + "".join(frame.text for frame in turn.frames)
            + "".join(map(self.render_memory, turn.questions))
        )

    def archive_turn(self, turn):
        return (*turn.memories, *turn.questions)
