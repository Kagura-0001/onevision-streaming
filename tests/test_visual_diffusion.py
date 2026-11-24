import hashlib
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch

from ov_streaming import qa
from ov_streaming.visual_diffusion import (
    apply_visual_diffusion,
    validate_visual_diffusion,
)


def payload(pixels):
    return {
        "prompt_token_ids": [10, 20, 30],
        "multi_modal_data": {
            "image": {
                "pixel_values": pixels,
                "image_grid_thw": torch.tensor([[1, 2, 3]]),
                "patch_positions": torch.tensor([[5, 6, 7]]),
            },
        },
    }


class VisualDiffusionTest(unittest.TestCase):
    def test_matches_official_equation_and_input_dtype(self):
        # Independent execution of the pinned VCD expression and global
        # randn_like under a temporary RNG scope. Production uses a private RNG.
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
            clean = torch.linspace(-3, 3, 24).reshape(4, 6).to(dtype)
            for step in (0, 500, 999):
                with self.subTest(dtype=dtype, step=step), torch.random.fork_rng(devices=[]):
                    torch.manual_seed(12345)
                    beta = torch.sigmoid(torch.linspace(-6, 6, 1000, dtype=torch.float32))
                    beta = beta * (0.5e-2 - 1e-5) + 1e-5
                    alpha = torch.cumprod(1 - beta, dim=0)
                    expected = (alpha.sqrt()[step] * clean.clone()
                                + (1 - alpha).sqrt()[step] * torch.randn_like(clean))
                actual, _ = apply_visual_diffusion(
                    payload(clean), {"noise_step": step, "seed": 12345}
                )
                result = actual["multi_modal_data"]["image"]["pixel_values"]
                self.assertTrue(torch.equal(result, expected))
                self.assertEqual(result.dtype, clean.dtype)

    def test_copies_pixels_preserves_other_fields_and_reports_byte_hashes(self):
        clean = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()
        before = clean.clone()
        original = payload(clean)
        images = original["multi_modal_data"]["image"]
        actual, metadata = apply_visual_diffusion(
            original, {"noise_step": 500, "seed": 2**63 - 1}
        )
        changed = actual["multi_modal_data"]["image"]
        self.assertIsNot(actual, original)
        self.assertIs(actual["prompt_token_ids"], original["prompt_token_ids"])
        self.assertIs(changed["image_grid_thw"], images["image_grid_thw"])
        self.assertIs(changed["patch_positions"], images["patch_positions"])
        self.assertIs(images["pixel_values"], clean)
        self.assertTrue(torch.equal(clean, before))
        self.assertNotEqual(changed["pixel_values"].data_ptr(), clean.data_ptr())
        self.assertEqual(metadata["dtype"], "torch.float32")
        self.assertEqual(metadata["shape"], [6, 4])
        self.assertEqual(metadata["clean_tensor_sha256"],
                         hashlib.sha256(clean.contiguous().numpy().tobytes()).hexdigest())
        self.assertEqual(metadata["noisy_tensor_sha256"], hashlib.sha256(
            changed["pixel_values"].contiguous().numpy().tobytes()).hexdigest())
        self.assertGreater(changed["pixel_values"].max().item(), 1)  # No clamp.

    def test_private_rng_reproducible_across_concurrent_requests(self):
        original = payload(torch.zeros(8, 16))
        state = torch.random.get_rng_state().clone()
        def draw(seed):
            return apply_visual_diffusion(original, {"noise_step": 500, "seed": seed})
        first = [draw(seed) for seed in (1, 2)]
        with ThreadPoolExecutor(4) as pool:
            repeated = list(pool.map(draw, [2, 1, 2, 1]))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        for item, seed in zip(repeated, [2, 1, 2, 1]):
            self.assertEqual(item[1], first[seed - 1][1])
        self.assertNotEqual(first[0][1]["noisy_tensor_sha256"],
                            first[1][1]["noisy_tensor_sha256"])

    def test_disabled_path_is_identical_and_does_not_hash(self):
        original = {"prompt_token_ids": [1]}  # No image needed without opt-in.
        with patch("ov_streaming.visual_diffusion._tensor_sha256") as digest:
            actual, metadata = apply_visual_diffusion(original, None)
        self.assertIs(actual, original)
        self.assertIsNone(metadata)
        digest.assert_not_called()

    def test_rejects_invalid_options_without_coercion(self):
        invalid = [False, [], {}, {"noise_step": 500},
                   {"noise_step": 500, "seed": 1, "extra": 2}]
        invalid += [{"noise_step": value, "seed": 1}
                    for value in (-1, 1000, True, 500.0, "500")]
        invalid += [{"noise_step": 500, "seed": value}
                    for value in (-1, 2**63, False, 1.0, "1")]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_visual_diffusion(value)
        self.assertEqual(validate_visual_diffusion({"noise_step": 500, "seed": 0}),
                         {"noise_step": 500, "seed": 0})

    def test_rejects_missing_integer_empty_and_nonfinite_pixels(self):
        cases = [{"prompt_token_ids": []}, payload(torch.zeros(2, dtype=torch.long)),
                 payload(torch.empty(0)), payload(torch.tensor([float("nan")])),
                 payload(torch.tensor([float("inf")]))]
        for original in cases:
            with self.subTest(original=original), self.assertRaises(ValueError):
                apply_visual_diffusion(original, {"noise_step": 500, "seed": 0})


class VisualDiffusionHTTPTest(unittest.IsolatedAsyncioTestCase):
    async def test_default_and_noisy_http_preserve_sampling_and_clean_source(self):
        import httpx

        original = payload(torch.arange(24, dtype=torch.float32).reshape(4, 6))
        before = original["multi_modal_data"]["image"]["pixel_values"].clone()
        app = qa.build_app(SimpleNamespace(), None, served_model="qa")
        app.state.backend = SimpleNamespace(max_model_len=256)
        result = {"text": "A", "finish_reason": "length",
                  "usage": {"input_tokens": 3, "output_tokens": 1},
                  "first_token_logprobs": {str(i): -float(i) for i in range(32, 36)}}
        generate = AsyncMock(return_value=result)
        base = {"messages": [], "max_tokens": 1, "stop": [],
                "logprob_token_ids": [32, 33, 34, 35]}
        with (patch.object(qa, "render_packed", return_value=("prompt", [object()])),
              patch.object(qa, "engine_prompt", return_value=original),
              patch.object(qa, "answer_once", generate)):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://qa") as client:
                for extra in ({}, {"visual_diffusion": None}):
                    response = await client.post("/v1/qa", json={**base, **extra})
                    self.assertEqual(response.status_code, 200)
                    self.assertNotIn("visual_diffusion", response.json())
                    self.assertIs(generate.call_args.args[1], original)
                response = await client.post("/v1/qa", json={
                    **base, "visual_diffusion": {"noise_step": 500, "seed": 42}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["first_token_logprobs"],
                         result["first_token_logprobs"])
        self.assertEqual(set(response.json()["visual_diffusion"]), {
            "noise_step", "seed", "dtype", "shape",
            "clean_tensor_sha256", "noisy_tensor_sha256"})
        self.assertEqual(generate.call_args.kwargs["max_tokens"], 1)
        self.assertEqual(generate.call_args.kwargs["logprob_token_ids"], [32, 33, 34, 35])
        self.assertEqual(generate.call_args.kwargs["temperature"], 0)
        self.assertIsNot(generate.call_args.args[1], original)
        self.assertTrue(torch.equal(original["multi_modal_data"]["image"]["pixel_values"], before))

    async def test_invalid_http_fails_before_engine_generation(self):
        import httpx

        app = qa.build_app(SimpleNamespace(), None, served_model="qa")
        app.state.backend = SimpleNamespace(max_model_len=256)
        with (patch.object(qa, "render_packed", return_value=("prompt", [])),
              patch.object(qa, "engine_prompt", return_value={"prompt_token_ids": [1]}),
              patch.object(qa, "answer_once", AsyncMock()) as generate):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://qa") as client:
                for settings in ({"noise_step": True, "seed": 0},
                                 {"noise_step": 500, "seed": 0}):
                    response = await client.post("/v1/qa", json={
                        "messages": [], "max_tokens": 1, "visual_diffusion": settings})
                    self.assertEqual(response.status_code, 422)
        generate.assert_not_called()
