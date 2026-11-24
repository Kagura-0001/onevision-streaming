import base64
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from ov_streaming import qa


def image_part(image, **extra):
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return {
        "type": "image",
        "data": base64.b64encode(buffer.getvalue()).decode(),
        "timestamp": 1.96,
        "frame_index": 49,
        "fps": 25.0,
        **extra,
    }


class PackedQATest(unittest.IsolatedAsyncioTestCase):
    async def test_packed_question_follows_images_without_control_turns(self):
        seen = []

        class Renderer:
            system_prompt = "Video assistant"

            def render_frame(self, decoded, at, *, frame_index, fps):
                seen.append((decoded.size, at, frame_index, fps))
                return SimpleNamespace(text="<1.96 seconds><image>\n", data="pixels")

        image = image_part(Image.new("RGB", (8, 8), (13, 21, 34)))
        prompt, frames = qa.render_packed(
            Renderer(),
            [
                {
                    "role": "user",
                    "content": [
                        image,
                        {"type": "text", "text": "Which object?\nA. Cup\nB. Ball"},
                    ],
                }
            ],
        )
        self.assertEqual(seen, [((8, 8), 1.96, 49, 25.0)])
        self.assertEqual(
            prompt,
            "<|im_start|>system\nVideo assistant<|im_end|>\n<|im_start|>user\n"
            "<1.96 seconds><image>\nWhich object?\nA. Cup\nB. Ball<|im_end|>\n<|im_start|>assistant\n",
        )
        self.assertEqual(len(frames), 1)

    async def test_source_pixels_crop_only_declared_padding_and_pad_to_ov_grid(self):
        original = Image.new("RGB", (17, 19), (13, 21, 34))
        encoded = Image.new("RGB", (32, 32), (255, 0, 0))
        encoded.paste(original, (0, 0))
        seen = []

        class Renderer:
            system_prompt = "Video assistant"
            preserve_source_pixels = True
            image_processor = SimpleNamespace(patch_size=16, merge_size=3)

            def render_frame(self, image, *args, **kwargs):
                seen.append(image.copy())
                return SimpleNamespace(text="image", data=None)

        part = image_part(encoded, source_size=[17, 19])
        qa.render_packed(Renderer(), [{"role": "user", "content": [part]}])
        self.assertEqual(seen[0].size, (48, 48))
        self.assertEqual(seen[0].crop((0, 0, 17, 19)).tobytes(), original.tobytes())
        self.assertEqual(seen[0].getpixel((17, 19)), (0, 0, 0))
        part["source_size"] = [33, 19]
        with self.assertRaises(ValueError):
            qa.render_packed(Renderer(), [{"role": "user", "content": [part]}])

    async def test_answer_uses_ordinary_greedy_generation_with_full_budget(self):
        calls = []

        class Engine:
            async def generate(self, payload, params, request_id):
                calls.append((payload, params, request_id))
                yield SimpleNamespace(
                    prompt_token_ids=[1, 2, 3],
                    outputs=[
                        SimpleNamespace(text="B", token_ids=[4], finish_reason="stop")
                    ],
                )

        payload = {"prompt_token_ids": [1, 2, 3]}
        result = await qa.answer_once(
            SimpleNamespace(engine=Engine()),
            payload,
            max_tokens=256,
            stop=[],
            request_id="qa",
        )
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], payload)
        params = calls[0][1]
        self.assertEqual(
            (params.max_tokens, params.temperature, params.allowed_token_ids),
            (256, 0, None),
        )
        self.assertFalse(params.skip_special_tokens)
        self.assertIsNone(params.logprob_token_ids)
        self.assertIsNone(params.num_logprobs)
        self.assertIsNone(params.logit_bias)
        self.assertEqual(
            result,
            {
                "text": "B",
                "finish_reason": "stop",
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        )

    async def test_reference_sampling_reaches_engine_without_control_mask(self):
        seen = []

        class Engine:
            async def generate(self, payload, params, request_id):
                seen.append(params)
                yield SimpleNamespace(
                    prompt_token_ids=[1],
                    outputs=[SimpleNamespace(
                        text="</Silence>", token_ids=[2], finish_reason="stop"
                    )],
                )

        app = qa.build_app(SimpleNamespace(), None, served_model="qa")
        app.state.backend = SimpleNamespace(max_model_len=256, engine=Engine())
        import httpx
        with (
            patch.object(qa, "render_packed", return_value=("prompt", [])),
            patch.object(qa, "engine_prompt", return_value={"prompt_token_ids": [1]}),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://qa"
            ) as client:
                response = await client.post("/v1/qa", json={
                    "messages": [], "max_tokens": 128, "temperature": 0.7,
                    "top_p": 0.8, "top_k": 20, "seed": 42,
                })
        self.assertEqual(response.status_code, 200)
        params = seen[0]
        self.assertEqual(
            (params.temperature, params.top_p, params.top_k, params.seed, params.max_tokens),
            (0.7, 0.8, 20, 42, 128),
        )
        self.assertIsNone(params.allowed_token_ids)
        self.assertEqual(response.json()["text"], "</Silence>")

    async def test_reference_interval_replaces_only_native_time_label(self):
        frame = SimpleNamespace(
            text="<1.96 seconds><|vision_start|>pixels<|vision_end|>\n",
            data=object(),
        )
        renderer = SimpleNamespace(system_prompt="s", render_frame=lambda *a, **k: frame)
        part = image_part(Image.new("RGB", (8, 8)), timestamp_text="")
        prompt, frames = qa.render_packed(renderer, [{"role": "user", "content": [
            {"type": "text", "text": "<1.750s-2.000s>"}, part,
        ]}])
        self.assertIn("<1.750s-2.000s><|vision_start|>pixels<|vision_end|>\n", prompt)
        self.assertNotIn("seconds", prompt)
        self.assertIs(frames[0], frame)

    async def test_http_rejects_over_budget_input_without_generation(self):
        import httpx

        backend = SimpleNamespace(max_model_len=3)
        app = qa.build_app(SimpleNamespace(), None, served_model="qa")
        app.state.backend = backend
        with (
            patch.object(qa, "render_packed", return_value=("prompt", [])),
            patch.object(
                qa, "engine_prompt", return_value={"prompt_token_ids": [1, 2, 3]}
            ),
            patch.object(qa, "answer_once") as generate,
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://qa"
            ) as client:
                result = await client.post(
                    "/v1/qa", json={"messages": [], "max_tokens": 1}
                )
            self.assertEqual(result.status_code, 422)
            self.assertIn("frames were not truncated", result.json()["detail"])
            generate.assert_not_called()


    async def test_selected_first_token_logprobs_are_raw_complete_and_unmasked(self):
        calls = []
        expected = {"32": -1.125, "33": -4.25, "34": -20.0, "35": -100.0}
        first = {
            int(token_id): SimpleNamespace(logprob=value, rank=100 + index)
            for index, (token_id, value) in enumerate(expected.items())
        }
        # The generated token is outside the requested set. Requested values need
        # not be top-k candidates, nor sum to one over these four candidates.
        first[999] = SimpleNamespace(logprob=-0.125, rank=1)

        class Engine:
            async def generate(self, payload, params, request_id):
                calls.append((payload, params, request_id))
                yield SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[SimpleNamespace(
                        text="At", token_ids=[999], finish_reason="length",
                        logprobs=[first],
                    )],
                )

        payload = {"prompt_token_ids": [1, 2]}
        result = await qa.answer_once(
            SimpleNamespace(engine=Engine()), payload,
            max_tokens=1, stop=[], request_id="selected",
            logprob_token_ids=[32, 33, 34, 35],
        )
        self.assertIs(calls[0][0], payload)
        params = calls[0][1]  # Actual installed vLLM SamplingParams, not a stub.
        self.assertEqual(params.logprob_token_ids, [32, 33, 34, 35])
        self.assertEqual(params.num_logprobs, 4)
        self.assertIsNone(params.logprobs)
        self.assertIsNone(params.allowed_token_ids)
        self.assertIsNone(params.logit_bias)
        self.assertEqual((params.max_tokens, params.temperature), (1, 0))
        self.assertEqual(result["first_token_logprobs"], expected)
        self.assertEqual(result["text"], "At")
        self.assertEqual(result["usage"]["output_tokens"], 1)

    async def test_http_selected_logprobs_use_first_position_only(self):
        import httpx

        seen = []
        first = {i: SimpleNamespace(logprob=-float(i)) for i in range(32, 36)}
        second = {i: SimpleNamespace(logprob=-0.5) for i in range(32, 36)}

        class Engine:
            async def generate(self, payload, params, request_id):
                seen.append(params)
                yield SimpleNamespace(
                    prompt_token_ids=[1],
                    outputs=[SimpleNamespace(
                        text="At 1", token_ids=[900, 901], finish_reason="length",
                        logprobs=[first, second],
                    )],
                )

        app = qa.build_app(SimpleNamespace(), None, served_model="qa")
        app.state.backend = SimpleNamespace(max_model_len=256, engine=Engine())
        with (
            patch.object(qa, "render_packed", return_value=("prompt", [])),
            patch.object(qa, "engine_prompt", return_value={"prompt_token_ids": [1]}),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://qa"
            ) as client:
                response = await client.post("/v1/qa", json={
                    "messages": [], "max_tokens": 2,
                    "logprob_token_ids": [32, 33, 34, 35],
                })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(seen[0].logprob_token_ids, [32, 33, 34, 35])
        self.assertEqual(response.json()["first_token_logprobs"], {
            str(i): -float(i) for i in range(32, 36)
        })

    async def test_http_omitted_or_null_logprobs_preserves_default_response(self):
        import httpx

        seen = []

        class Engine:
            async def generate(self, payload, params, request_id):
                seen.append(params)
                yield SimpleNamespace(
                    prompt_token_ids=[1],
                    outputs=[SimpleNamespace(
                        text="B", token_ids=[33], finish_reason="stop"
                    )],
                )

        app = qa.build_app(SimpleNamespace(), None, served_model="qa")
        app.state.backend = SimpleNamespace(max_model_len=512, engine=Engine())
        with (
            patch.object(qa, "render_packed", return_value=("prompt", [])),
            patch.object(qa, "engine_prompt", return_value={"prompt_token_ids": [1]}),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://qa"
            ) as client:
                for optional in ({}, {"logprob_token_ids": None}):
                    with self.subTest(optional=optional):
                        response = await client.post("/v1/qa", json={
                            "messages": [], "request_id": "default", **optional,
                        })
                        self.assertEqual(response.status_code, 200)
                        body = response.json()
                        self.assertEqual(set(body), {
                            "request_id", "frames", "elapsed_sec", "text",
                            "finish_reason", "usage",
                        })
                        self.assertEqual(body["text"], "B")
        for params in seen:
            self.assertIsNone(params.logprob_token_ids)
            self.assertIsNone(params.num_logprobs)
            self.assertIsNone(params.allowed_token_ids)
            self.assertIsNone(params.logit_bias)
            self.assertEqual((params.temperature, params.max_tokens), (0, 256))

    async def test_http_invalid_token_ids_fail_before_preprocessing_or_generation(self):
        import httpx

        app = qa.build_app(SimpleNamespace(), None, served_model="qa")
        app.state.backend = SimpleNamespace(max_model_len=256)
        with (
            patch.object(qa, "render_packed") as render,
            patch.object(qa, "answer_once") as generate,
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://qa"
            ) as client:
                for token_ids in ([], "32", [True], [32.0], [-1], [32, 32], list(range(129))):
                    with self.subTest(token_ids=token_ids):
                        response = await client.post("/v1/qa", json={
                            "messages": [], "max_tokens": 1,
                            "logprob_token_ids": token_ids,
                        })
                        self.assertEqual(response.status_code, 422)
                        self.assertIn("logprob_token_ids", response.json()["detail"])
            render.assert_not_called()
            generate.assert_not_called()

    async def test_http_missing_or_nonfinite_logprobs_fail_without_partial_values(self):
        import httpx

        cases = [
            (None, "no first-token"),
            ([], "no first-token"),
            ([{}], "missing requested token IDs"),
            ([{32: SimpleNamespace(logprob=-1)}], "missing requested token IDs"),
        ]
        for invalid in (float("nan"), float("inf"), float("-inf"), None):
            row = {i: SimpleNamespace(logprob=-1.0) for i in range(32, 36)}
            row[35] = SimpleNamespace(logprob=invalid)
            cases.append(([row], "logprob for token ID 35"))

        class Engine:
            rows = None

            async def generate(self, payload, params, request_id):
                yield SimpleNamespace(
                    prompt_token_ids=[1],
                    outputs=[SimpleNamespace(
                        text="A", token_ids=[32], finish_reason="length",
                        logprobs=self.rows,
                    )],
                )

        engine = Engine()
        app = qa.build_app(SimpleNamespace(), None, served_model="qa")
        app.state.backend = SimpleNamespace(max_model_len=256, engine=engine)
        with (
            patch.object(qa, "render_packed", return_value=("prompt", [])),
            patch.object(qa, "engine_prompt", return_value={"prompt_token_ids": [1]}),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://qa"
            ) as client:
                for rows, expected in cases:
                    with self.subTest(rows=rows):
                        engine.rows = rows
                        response = await client.post("/v1/qa", json={
                            "messages": [], "max_tokens": 1,
                            "logprob_token_ids": [32, 33, 34, 35],
                        })
                        self.assertEqual(response.status_code, 502)
                        self.assertIn(expected, response.json()["detail"])
                        self.assertNotIn("first_token_logprobs", response.json())
