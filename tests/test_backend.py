from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from ov_streaming.backend import VLLMBackend
from ov_streaming.types import BackendOutput, VisualFrame


class FakeTokenizer:
    def encode(self, text, add_special_tokens):
        return [ord(char) for char in text]


class FakeCollector:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def get_nowait(self):
        return None

    async def get(self):
        return self.outputs.pop(0)


class FakeEngine:
    def __init__(self, token_ids=(), branch_outputs=()):
        self.model_config = SimpleNamespace(max_model_len=4096)
        self.token_ids = list(token_ids)
        self.collector = FakeCollector(branch_outputs)
        self.chunks = []
        self.branches = []
        self.aborted = []
        self.model_config = SimpleNamespace(max_model_len=4096)

    def get_tokenizer(self):
        return FakeTokenizer()

    async def generate(self, inputs, params, request_id):
        prompt_token_ids = []
        try:
            async for chunk in inputs:
                self.chunks.append(chunk)
                prompt_token_ids = [
                    *prompt_token_ids,
                    *chunk.prompt["prompt_token_ids"],
                ]
                token_ids = [self.token_ids.pop(0)]
                yield SimpleNamespace(
                    prompt_token_ids=prompt_token_ids,
                    outputs=[SimpleNamespace(token_ids=token_ids)],
                )
        except GeneratorExit:
            self.aborted.append(request_id)
            raise

    async def add_branch_request(
        self, source_id, request_id, params, *, suffix_token_ids
    ):
        self.branches.append((source_id, request_id, params, suffix_token_ids))
        return self.collector


def frame(patches):
    data = {
        "pixel_values": torch.ones(patches, 3),
        "image_grid_thw": torch.tensor([[1, 1, patches]]),
    }
    return VisualFrame("", data)


class VLLMBackendTest(unittest.IsolatedAsyncioTestCase):
    async def test_source_steps_feed_one_streaming_request(self):
        engine = FakeEngine([7, 8])
        source = VLLMBackend(engine).open_source("source", cache_salt="salt")

        first = await source.step("ab", [frame(4)], allowed_token_ids=(7, 8))
        second = await source.step("c", [frame(4), frame(2)], allowed_token_ids=(7, 8))
        await source.close()

        self.assertEqual((first, second), (7, 8))
        self.assertEqual(source.num_tokens, 3)
        self.assertEqual(engine.aborted, [])
        self.assertEqual(source.num_tokens, 3)
        self.assertEqual(source.backend.max_model_len, 4096)
        self.assertEqual(source.backend.count_tokens("abc"), 3)
        one, two = engine.chunks
        self.assertEqual(one.prompt["prompt_token_ids"], [97, 98])
        self.assertEqual(one.prompt["cache_salt"], "salt")
        self.assertNotIn("cache_salt", two.prompt)
        image = two.prompt["multi_modal_data"]["image"]
        self.assertEqual(image["pixel_values"].shape, (6, 3))
        self.assertEqual(image["image_grid_thw"].tolist(), [[1, 1, 4], [1, 1, 2]])
        params = two.sampling_params
        self.assertEqual(
            (params.max_tokens, params.temperature, params.allowed_token_ids),
            (1, 0.0, [7, 8]),
        )

    async def test_abort_ends_the_source_request(self):
        engine = FakeEngine([7])
        source = VLLMBackend(engine).open_source("source", cache_salt="salt")
        await source.step("a", (), allowed_token_ids=(7,))
        await source.abort()
        self.assertEqual(engine.aborted, ["source"])

    async def test_branch_streams_until_finished(self):
        engine = FakeEngine(
            branch_outputs=[
                SimpleNamespace(
                    outputs=[SimpleNamespace(text=" At 3", finish_reason=None)],
                    finished=False,
                ),
                SimpleNamespace(
                    outputs=[SimpleNamespace(text=" seconds", finish_reason="stop")],
                    finished=True,
                ),
            ]
        )
        branch = await VLLMBackend(engine).add_branch(
            "source", "response", suffix_token_ids=[9], max_tokens=5
        )
        outputs = [output async for output in branch.outputs()]

        self.assertEqual(
            outputs,
            [
                BackendOutput(" At 3", None),
                BackendOutput(" seconds", "stop"),
            ],
        )
        ((source_id, request_id, params, suffix),) = engine.branches
        self.assertEqual((source_id, request_id, suffix), ("source", "response", [9]))
        self.assertEqual((params.max_tokens, params.temperature), (5, 0.0))

    async def test_engine_error_fails_the_branch(self):
        engine = FakeEngine(
            branch_outputs=[
                SimpleNamespace(
                    outputs=[SimpleNamespace(text="", finish_reason="error")],
                    finished=True,
                )
            ]
        )
        branch = await VLLMBackend(engine).add_branch(
            "source", "response", suffix_token_ids=[9], max_tokens=5
        )
        with self.assertRaises(RuntimeError):
            async for _ in branch.outputs():
                pass


if __name__ == "__main__":
    unittest.main()
