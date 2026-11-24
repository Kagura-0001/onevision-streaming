from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from ov_streaming.renderers import load_renderer, ov21
from ov_streaming.renderers.ov21 import (
    OV21Renderer,
    OV21WindowRenderer,
    _is_local_checkpoint,
    _video_processor_module_candidates,
)

OV25_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<silence>",
    "<response>",
    "<captioning>",
)


class _DummyTokenizer:
    def __init__(self, tokens=OV25_TOKENS):
        self.vocab = {t: i for i, t in enumerate(tokens)}

    def get_added_vocab(self):
        return dict(self.vocab)

    def encode(self, text, add_special_tokens=False):
        if text in self.vocab:
            return [self.vocab[text]]
        return [100 + i for i, _ in enumerate(text.split())]


class _DummyImageProcessor:
    def __call__(self, images, *, do_resize=True, min_pixels=None, max_pixels=None, return_tensors="pt"):
        w, h = images[0].size
        return {
            "pixel_values": torch.zeros((h // 16) * (w // 16), 3 * 16 * 16),
            "image_grid_thw": torch.tensor([[1, h // 16, w // 16]]),
        }


class OV25CompatibilityTest(unittest.TestCase):
    def test_renderer_aliases(self):
        self.assertIs(load_renderer("ov25"), OV21Renderer)
        self.assertIs(load_renderer("ov25-window"), OV21WindowRenderer)

    def test_local_checkpoint_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(_is_local_checkpoint(tmp))
            self.assertTrue(_is_local_checkpoint(Path(tmp)))
        self.assertFalse(
            _is_local_checkpoint("lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct")
        )

    def test_video_processor_module_candidates_prefers_auto_map(self):
        cfg = SimpleNamespace(
            model_type="llava_onevision2_5",
            auto_map={
                "AutoVideoProcessor": "video_processing_llava_onevision2_5.LlavaOnevision2_5VideoProcessor"
            },
        )
        self.assertEqual(
            _video_processor_module_candidates(cfg),
            ["video_processing_llava_onevision2_5", "video_processing_llava_onevision2"],
        )

        legacy_cfg = SimpleNamespace(model_type="llava_onevision2_qwen35")
        self.assertEqual(
            _video_processor_module_candidates(legacy_cfg),
            ["video_processing_llava_onevision2", "video_processing_llava_onevision2_5"],
        )

    def test_from_pretrained_resolves_hf_repo_without_local_files_only(self):
        seen_calls = []

        def fake_get_dynamic(ref, checkpoint, *, local_files_only):
            seen_calls.append((ref, checkpoint, local_files_only))
            if ref.startswith("video_processing_llava_onevision2_5."):
                return lambda grid, *, spatial_merge_size, frame_indices: torch.zeros(
                    int(grid.prod()), 3, dtype=torch.long
                )
            raise OSError("missing legacy module")

        hf_config = SimpleNamespace(
            model_type="llava_onevision2_5",
            auto_map={
                "AutoVideoProcessor": "video_processing_llava_onevision2_5.LlavaOnevision2_5VideoProcessor"
            },
            vision_config=SimpleNamespace(patch_size=16, spatial_merge_size=3),
        )
        with (
            patch.object(
                ov21.Qwen2VLImageProcessor,
                "from_pretrained",
                return_value=_DummyImageProcessor(),
            ) as mock_proc,
            patch.object(ov21, "get_class_from_dynamic_module", side_effect=fake_get_dynamic),
        ):
            renderer = OV21Renderer.from_pretrained(
                "lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct",
                tokenizer=_DummyTokenizer(),
                hf_config=hf_config,
                max_pixels=258048,
            )

        mock_proc.assert_called_once_with(
            "lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct",
            local_files_only=False,
        )
        self.assertEqual(
            seen_calls,
            [
                (
                    "video_processing_llava_onevision2_5.build_patch_positions",
                    "lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct",
                    False,
                )
            ],
        )
        self.assertEqual(renderer.max_pixels, 258048)

        # Verify codec canvas (48x48 = 3x3 patches, merge_size=3) renders cleanly.
        canvas = Image.new("RGB", (48, 48))
        positions = [[24, y, x] for y in range(3) for x in range(3)]
        frame = renderer.render_codec_frame(
            canvas,
            1.0,
            codec={"layout": "row_major", "patch_size": 16, "patch_positions": positions},
            fps=24.0,
        )
        self.assertIn("<1.0 seconds><|vision_start|><|image_pad|><|vision_end|>\n", frame.text)
        self.assertEqual(tuple(frame.data["patch_positions"].shape), (9, 3))


if __name__ == "__main__":
    unittest.main()
