from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from typing import Any

import torch
from vllm.engine.protocol import StreamingInput
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind, SamplingParams

from .types import BackendOutput, VisualFrame


class VLLMBackend:
    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.tokenizer = engine.get_tokenizer()
        self.max_model_len = engine.model_config.max_model_len

    def open_source(self, request_id: str, *, cache_salt: str) -> SourceRequest:
        return SourceRequest(self, request_id, cache_salt)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    async def add_branch(
        self,
        source_request_id: str,
        request_id: str,
        *,
        suffix_token_ids: Sequence[int],
        max_tokens: int,
    ) -> Branch:
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            output_kind=RequestOutputKind.DELTA,
        )
        collector = await self.engine.add_branch_request(
            source_request_id,
            request_id,
            params,
            suffix_token_ids=suffix_token_ids,
        )
        return Branch(collector)

    async def abort(self, request_id: str) -> None:
        await self.engine.abort(request_id)


class SourceRequest:
    """A resumable vLLM request that samples one control token per input."""

    def __init__(self, backend: VLLMBackend, request_id: str, cache_salt: str) -> None:
        self.backend = backend
        self.request_id = request_id
        self.cache_salt = cache_salt
        self._inputs: asyncio.Queue[StreamingInput | None] = asyncio.Queue()
        self._outputs: AsyncGenerator[Any, None] | None = None
        # Prompt tokens the engine holds after the last step, images expanded.
        self.num_tokens = 0

    async def step(
        self,
        prompt: str,
        frames: Sequence[VisualFrame],
        *,
        allowed_token_ids: Sequence[int],
    ) -> int:
        params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            allowed_token_ids=list(allowed_token_ids),
            output_kind=RequestOutputKind.DELTA,
        )
        engine_prompt = TokensPrompt(
            prompt_token_ids=self.backend.tokenizer.encode(
                prompt, add_special_tokens=False
            )
        )
        if frames:
            images = [frame.data for frame in frames]
            if isinstance(images[0], Mapping):
                # vLLM takes processed images as one dict of batched tensors.
                images = {
                    key: torch.cat([image[key] for image in images])
                    for key in images[0]
                }
            engine_prompt["multi_modal_data"] = {"image": images}
        if self._outputs is None:
            engine_prompt["cache_salt"] = self.cache_salt
            self._outputs = self.backend.engine.generate(
                self._read_inputs(), params, self.request_id
            )
        self._inputs.put_nowait(StreamingInput(engine_prompt, params))
        output = await anext(self._outputs)
        self.num_tokens = len(output.prompt_token_ids)
        (token_id,) = output.outputs[0].token_ids
        return token_id

    async def close(self) -> None:
        self._inputs.put_nowait(None)
        async for _ in self._outputs:
            pass

    async def abort(self) -> None:
        if self._outputs is not None:
            # generate() aborts the request when it is closed early.
            await self._outputs.aclose()

    async def _read_inputs(self) -> AsyncGenerator[StreamingInput, None]:
        while (item := await self._inputs.get()) is not None:
            yield item


class Branch:
    def __init__(self, collector: Any) -> None:
        self._collector = collector

    async def outputs(self) -> AsyncIterator[BackendOutput]:
        while True:
            output = self._collector.get_nowait() or await self._collector.get()
            completion = output.outputs[0]
            if completion.finish_reason == "error":
                raise RuntimeError("the engine failed the branch")
            yield BackendOutput(completion.text, completion.finish_reason)
            if output.finished:
                return
