import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from ov_streaming import qa


class RawLogitsTest(unittest.IsolatedAsyncioTestCase):
    def app(self, mode, rows=None):
        self.calls = []
        calls = self.calls

        class Engine:
            async def generate(self, payload, params, request_id):
                calls.append((payload, params, request_id))
                yield SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[SimpleNamespace(
                        text="outside", token_ids=[999], finish_reason="length",
                        logprobs=rows,
                    )],
                )

        app = qa.build_app(SimpleNamespace(), None, served_model="qa",
                           score_output_mode=mode)
        app.state.backend = SimpleNamespace(max_model_len=512, engine=Engine())
        return app

    async def request(self, app, body):
        with (patch.object(qa, "render_packed", return_value=("prompt", [])),
              patch.object(qa, "engine_prompt", return_value={"prompt_token_ids": [1, 2]})):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://qa") as client:
                return await client.post("/v1/qa", json={"messages": [], **body})

    async def test_raw_first_position_values_are_unchanged_and_unmasked(self):
        values = {32: 18.5, 33: -3.25, 34: 0.0, 35: 2.75}
        first = {key: SimpleNamespace(logprob=value) for key, value in values.items()}
        first[999] = SimpleNamespace(logprob=25.0)  # Generated token outside candidates.
        second = {key: SimpleNamespace(logprob=-99.0) for key in values}
        app = self.app("raw_logits", [first, second])
        response = await self.request(app, {
            "max_tokens": 1, "stop": [], "request_id": "raw",
            "logits_token_ids": [32, 33, 34, 35],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["first_token_logits"],
                         {str(key): value for key, value in values.items()})
        self.assertEqual(response.json()["value_semantics"], "pre_sampling_logits")
        self.assertNotIn("first_token_logprobs", response.json())
        self.assertEqual(response.json()["text"], "outside")
        params = self.calls[0][1]  # Real installed vLLM SamplingParams.
        self.assertEqual(params.logprob_token_ids, [32, 33, 34, 35])
        self.assertEqual(params.num_logprobs, 4)
        self.assertIsNone(params.logprobs)
        self.assertIsNone(params.allowed_token_ids)
        self.assertIsNone(params.logit_bias)
        # vLLM canonicalizes greedy top_k=-1 to 0 (no top-k filtering).
        self.assertEqual((params.max_tokens, params.temperature, params.top_p,
                          params.top_k, params.stop), (1, 0, 1, 0, []))

    async def test_wrong_modes_and_ambiguous_or_invalid_ids_precede_render(self):
        for mode in qa.SCORE_OUTPUT_MODES:
            app = self.app(mode)
            forbidden = ("logprob_token_ids" if mode == "raw_logits"
                         else "logits_token_ids")
            cases = [{forbidden: [32]}, {forbidden: None},
                     {"logprob_token_ids": [32], "logits_token_ids": [32]},
                     {"logprob_token_ids": None, "logits_token_ids": None}]
            if mode == "raw_logits":
                cases += [{"logits_token_ids": value} for value in
                          (None, [], "32", [True], [32.0], [-1], [32, 32],
                           list(range(129)))]
            with (patch.object(qa, "render_packed") as render,
                  patch.object(qa, "answer_once") as generate):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                             base_url="http://qa") as client:
                    for body in cases:
                        with self.subTest(mode=mode, body=body):
                            response = await client.post("/v1/qa", json=body)
                            self.assertEqual(response.status_code, 422)
                render.assert_not_called()
                generate.assert_not_called()
            self.assertFalse(self.calls)

    async def test_raw_missing_or_nonfinite_is_502_without_partial_output(self):
        cases = [None, [], [{}], [{32: SimpleNamespace(logprob=1.0)}]]
        for invalid in (float("nan"), float("inf"), float("-inf"), None, "bad"):
            row = {key: SimpleNamespace(logprob=1.0) for key in range(32, 36)}
            row[35] = SimpleNamespace(logprob=invalid)
            cases.append([row])
        for rows in cases:
            with self.subTest(rows=rows):
                app = self.app("raw_logits", rows)
                response = await self.request(app, {
                    "max_tokens": 1, "logits_token_ids": [32, 33, 34, 35],
                })
                self.assertEqual(response.status_code, 502)
                self.assertNotIn("first_token_logits", response.json())
                self.assertIn("logit", response.json()["detail"])
                self.assertEqual(len(self.calls), 1)

    async def test_ordinary_generation_is_identical_in_both_modes(self):
        responses, params = [], []
        for mode in qa.SCORE_OUTPUT_MODES:
            app = self.app(mode)
            response = await self.request(app, {"request_id": "ordinary"})
            self.assertEqual(response.status_code, 200)
            result = response.json()
            result.pop("elapsed_sec")
            responses.append(result)
            params.append(self.calls[0][1])
        self.assertEqual(responses[0], responses[1])
        self.assertEqual(set(responses[0]), {
            "request_id", "frames", "text", "finish_reason", "usage",
        })
        for actual in params:
            self.assertIsNone(actual.logprob_token_ids)
            self.assertIsNone(actual.num_logprobs)
            self.assertIsNone(actual.allowed_token_ids)
            self.assertIsNone(actual.logit_bias)
            self.assertEqual((actual.max_tokens, actual.temperature), (256, 0))

    async def test_models_advertises_raw_channel_without_changing_default(self):
        for mode in qa.SCORE_OUTPUT_MODES:
            app = self.app(mode)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://qa") as client:
                response = await client.get("/v1/models")
            expected = {"id": "qa", "object": "model", "owned_by": "ov-qa"}
            if mode == "raw_logits":
                expected["score_output_mode"] = "raw_logits"
            self.assertEqual(response.json(), {"object": "list", "data": [expected]})
        with self.assertRaises(ValueError):
            qa.build_app(SimpleNamespace(), None, served_model="qa",
                         score_output_mode="processed_logits")


class RawLogitsEngineArgsTest(unittest.TestCase):
    def test_univia_keeps_its_training_pixels_unless_max_pixels_is_passed(self):
        engine = qa.build_engine_args(qa.parse_args(["--model", "unused", "--renderer", "univia"]))
        self.assertEqual(engine.mm_processor_kwargs, {"min_pixels": 4096, "max_pixels": 131072})
        self.assertEqual(engine.limit_mm_per_prompt, {"image": 1024, "video": 0})
        self.assertFalse(engine.hf_overrides)
        self.assertFalse(engine.async_scheduling)
        override = qa.build_engine_args(qa.parse_args(
            ["--model", "unused", "--renderer", "univia", "--max-pixels", "100352"]))
        self.assertEqual(override.mm_processor_kwargs, {"min_pixels": 4096, "max_pixels": 100352})
        # The renderer's class dict is not mutated by an override.
        again = qa.build_engine_args(qa.parse_args(["--model", "unused", "--renderer", "univia"]))
        self.assertEqual(again.mm_processor_kwargs["max_pixels"], 131072)

    def test_ov21_engine_args_and_explicit_max_pixels_are_unchanged(self):
        engine = qa.build_engine_args(qa.parse_args(["--model", "unused", "--max-pixels", "313000"]))
        self.assertEqual(engine.mm_processor_kwargs, {"max_pixels": 313000})
        self.assertEqual(engine.hf_overrides["model_type"], "llava_onevision2_qwen35")
        self.assertTrue(engine.trust_remote_code)
        self.assertFalse(engine.enable_prefix_caching)

    def test_renderer_accepts_module_class_like_serve(self):
        args = qa.parse_args(["--model", "unused", "--renderer", "ov_streaming.renderers.univia:UniVIARenderer"])
        self.assertEqual(args.renderer_class.__name__, "UniVIARenderer")

    def test_actual_async_engine_args_receive_only_requested_mode_change(self):
        from vllm.engine.arg_utils import AsyncEngineArgs

        default = qa.parse_args(["--model", "unused-no-model-load", "--max-pixels", "100000"])
        raw = qa.parse_args(["--model", "unused-no-model-load", "--max-pixels", "100000",
                             "--score-output-mode", "raw_logits"])
        self.assertEqual(default.score_output_mode, "raw_logprobs")
        a, b = qa.build_engine_args(default), qa.build_engine_args(raw)
        self.assertIsInstance(a, AsyncEngineArgs)
        self.assertIsInstance(b, AsyncEngineArgs)
        self.assertEqual(a.logprobs_mode, "raw_logprobs")
        self.assertEqual(b.logprobs_mode, "raw_logits")
        self.assertEqual({k: v for k, v in vars(a).items() if k != "logprobs_mode"},
                         {k: v for k, v in vars(b).items() if k != "logprobs_mode"})
        self.assertFalse(b.enable_prefix_caching)
        self.assertEqual(b.mamba_cache_mode, "none")
