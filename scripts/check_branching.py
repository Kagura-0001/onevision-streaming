"""Check streamed sessions and their branches against one-shot prefill on a GPU.

Runs OVSession with the chosen renderer on a vLLM engine and forces the control
of every input, so silence, every branch control, compaction and cancellation
are all exercised. Every control decision and the first tokens of every branch
are then prefilled again as one prompt and must give the same tokens with
logprobs within 0.1. Engine flags are vLLM's own, for example:

    python scripts/check_branching.py --renderer ov21 \\
        --model /path/to/checkpoint --max-model-len 16384 \\
        --tensor-parallel-size 2 --mamba-cache-mode align --enable-prefix-caching
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import uuid
from collections import deque
from collections.abc import Mapping
from itertools import chain

import torch
from PIL import Image, ImageDraw
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import StreamingInput
from vllm.inputs import TokensPrompt
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM

from ov_streaming.backend import VLLMBackend
from ov_streaming.serve import parse_engine_args
from ov_streaming.session import OVSession
from ov_streaming.types import Memory

LOGPROBS = 5
# Branch tokens compared with one-shot prefill; later tokens drift on near-ties.
COMPARED_TOKENS = 16
# Long enough that a branch is still running when it is cancelled.
HELD_TOKENS = 4096


class RecordingCollector:
    def __init__(self, collector):
        self.collector = collector
        self.token_ids = []
        self.logprobs = []

    def get_nowait(self):
        return self.record(self.collector.get_nowait())

    async def get(self):
        return self.record(await self.collector.get())

    def record(self, output):
        if output is not None and output.outputs:
            self.token_ids += output.outputs[0].token_ids
            self.logprobs += output.outputs[0].logprobs
        return output


class ScriptedEngine:
    """The AsyncLLM that VLLMBackend drives, with scripted controls and logprobs."""

    def __init__(self, engine):
        self.engine = engine
        self.model_config = engine.model_config
        self.controls = deque()
        self.body_tokens = None
        # Source request id -> the prompts it was given.
        self.sources = {}
        self.decisions = []
        self.branches = []

    def get_tokenizer(self):
        return self.engine.get_tokenizer()

    async def abort(self, request_id):
        await self.engine.abort(request_id)

    async def generate(self, inputs, params, request_id):
        prompts = self.sources[request_id] = []

        async def force(inputs):
            async for item in inputs:
                prompts.append(item.prompt)
                forced = item.sampling_params.clone()
                forced.allowed_token_ids = [self.controls.popleft()]
                forced.logprobs = LOGPROBS
                yield StreamingInput(item.prompt, forced)

        outputs = self.engine.generate(force(inputs), params, request_id)
        # Closing this generator must close the engine's, which aborts the request.
        async with contextlib.aclosing(outputs):
            async for output in outputs:
                self.decisions.append((list(prompts), output.outputs[0]))
                yield output

    async def add_branch_request(
        self, source_request_id, request_id, params, *, suffix_token_ids
    ):
        params = params.clone()
        params.logprobs = LOGPROBS
        if self.body_tokens is not None:
            params.max_tokens = self.body_tokens
            params.ignore_eos = True
        collector = RecordingCollector(
            await self.engine.add_branch_request(
                source_request_id,
                request_id,
                params,
                suffix_token_ids=suffix_token_ids,
            )
        )
        prompts = list(self.sources[source_request_id])
        self.branches.append((prompts, list(suffix_token_ids), params, collector))
        return collector


class Probe:
    def __init__(self, engine, renderer, frames, name, **options):
        self.name = name
        self.renderer = renderer
        self.frames = frames
        self.engine = ScriptedEngine(engine)
        self.tokenizer = engine.get_tokenizer()
        self.session = OVSession(VLLMBackend(self.engine), renderer, name, **options)
        self.events = []
        self.changed = asyncio.Event()
        self.reader = asyncio.create_task(self.consume())

    async def consume(self):
        async for event in self.session.events():
            self.events.append(event)
            self.changed.set()

    async def wait(self, event_type, **fields):
        while True:
            self.changed.clear()
            for event in self.events:
                if event.type == "error":
                    raise RuntimeError(event.error)
                if event.type == event_type and all(
                    getattr(event, key) == value for key, value in fields.items()
                ):
                    return event
            await self.changed.wait()

    async def send(self, index, control="silence", question=None):
        self.engine.controls.append(self.renderer.control_ids[control])
        content = [
            {
                "type": "image_pil",
                "image_pil": self.frames[index],
                "timestamp_sec": float(index),
            }
        ]
        if question:
            content.append({"type": "text", "text": question})
        input_id = f"frame-{index}"
        await self.session.send_messages(
            input_id=input_id, messages=[{"role": "user", "content": content}]
        )
        await self.session.drain()
        return await self.wait("input.done", input_id=input_id)

    def source_text(self):
        prompts = list(self.engine.sources.values())[-1]
        return self.tokenizer.decode(join_token_ids(prompts))

    async def close(self):
        await self.session.close()
        closed = await self.wait("session.closed")
        assert closed.status == "completed", closed
        await self.reader


def join_token_ids(prompts):
    return list(chain.from_iterable(prompt["prompt_token_ids"] for prompt in prompts))


async def prefill(engine, prompts, suffix, params):
    # A fresh cache salt keeps the reference from reusing the source's blocks.
    prompt = TokensPrompt(
        prompt_token_ids=join_token_ids(prompts) + suffix, cache_salt=uuid.uuid4().hex
    )
    images = [
        item["multi_modal_data"]["image"]
        for item in prompts
        if "multi_modal_data" in item
    ]
    if images:
        if isinstance(images[0], Mapping):
            images = {
                key: torch.cat([image[key] for image in images]) for key in images[0]
            }
        else:
            images = list(chain.from_iterable(images))
        prompt["multi_modal_data"] = {"image": images}
    async for output in engine.generate(prompt, params, f"reference-{uuid.uuid4()}"):
        pass
    return output.outputs[0]


def compare(name, token_ids, logprobs, expected):
    tokens = list(expected.token_ids)
    assert list(token_ids[: len(tokens)]) == tokens, (name, list(token_ids), tokens)
    difference = max(
        abs(actual[token].logprob - reference[token].logprob)
        for actual, reference in zip(logprobs, expected.logprobs)
        for token in actual.keys() & reference.keys()
    )
    # Different prefill and decode batch shapes can change BF16 rounding.
    assert difference < 0.1, (name, difference)
    row = {"check": name, "tokens": tokens, "max_logprob_difference": difference}
    print(json.dumps(row), flush=True)


async def compare_with_prefill(engine, probe):
    for index, (prompts, decision) in enumerate(probe.engine.decisions):
        params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=LOGPROBS,
            allowed_token_ids=list(decision.token_ids),
        )
        expected = await prefill(engine, prompts, [], params)
        name = f"{probe.name}/decision-{index}"
        compare(name, decision.token_ids, decision.logprobs, expected)
    for index, (prompts, suffix, params, branch) in enumerate(probe.engine.branches):
        if not branch.token_ids:
            continue
        params = SamplingParams(
            temperature=0.0,
            max_tokens=min(COMPARED_TOKENS, len(branch.token_ids)),
            logprobs=LOGPROBS,
            ignore_eos=params.ignore_eos,
        )
        expected = await prefill(engine, prompts, suffix, params)
        name = f"{probe.name}/branch-{index}"
        compare(name, branch.token_ids, branch.logprobs, expected)


async def check_silence(probe):
    for index in range(4):
        await probe.send(index, question="Watch the scene." if index == 0 else None)
    assert not any(event.type == "response.created" for event in probe.events)
    await probe.close()
    assert len(probe.engine.sources) == 1


async def check_memory(probe, control):
    # The source moves on and is replaced while the branch is generating.
    opening = await probe.send(0, control, "Describe what is happening briefly.")
    await probe.send(1)
    await probe.send(2)
    await probe.wait("context.compacted", input_id="frame-2")
    done = await probe.wait("response.done", response_id=opening.response_id)
    assert (done.status, done.reason) == ("completed", None), done
    output = await probe.wait(
        "response.output_text.done", response_id=opening.response_id
    )
    assert output.text, output
    memory = probe.renderer.render_memory(Memory(control, 0.0, output.text))
    await probe.send(3)
    assert probe.source_text().count(memory) == 1
    await probe.send(4)
    compacted = await probe.wait("context.compacted", input_id="frame-4")
    assert compacted.metadata["retained_visual_frames"] == 1
    assert probe.source_text().count(memory) == 1
    await probe.close()
    assert len(probe.engine.sources) == 3


async def check_cancel(probe):
    tag = f"<{probe.renderer.memory_tags['response']}>"
    probe.engine.body_tokens = HELD_TOKENS
    opening = await probe.send(0, "response", "Describe everything you can see.")
    assert await probe.session.cancel_response(opening.response_id)
    done = await probe.wait("response.done", response_id=opening.response_id)
    assert done.status == "cancelled", done
    await probe.send(1)
    assert tag not in probe.source_text()

    # Cancelling one of two running branches leaves the other intact.
    held = await probe.send(2, "response")
    probe.engine.body_tokens = COMPARED_TOKENS
    cut = await probe.send(3, "response")
    assert await probe.session.cancel_response(held.response_id)
    done = await probe.wait("response.done", response_id=cut.response_id)
    assert (done.status, done.reason) == ("completed", "length"), done
    await probe.send(4)
    assert tag not in probe.source_text()

    probe.engine.body_tokens = HELD_TOKENS
    opening = await probe.send(5, "response")
    await probe.session.abort()
    done = await probe.wait("response.done", response_id=opening.response_id)
    assert done.status == "cancelled", done
    closed = await probe.wait("session.closed")
    assert closed.status == "cancelled", closed


def draw_frames():
    frames = []
    for index in range(6):
        frame = Image.new("RGB", (448, 336), (40, 90, 160))
        x = 32 + 64 * index
        ImageDraw.Draw(frame).rectangle((x, 128, x + 64, 192), fill=(240, 200, 40))
        frames.append(frame)
    return frames


async def check_branching(args):
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs.from_cli_args(args))
    renderer = args.renderer_class(engine.get_tokenizer(), engine.model_config)
    if args.images:
        frames = [Image.open(path).convert("RGB") for path in args.images]
    else:
        frames = draw_frames()

    async def run(name, check, *check_args, **options):
        probe = Probe(engine, renderer, frames, name, **options)
        row = {"check": name, "status": "failed"}
        try:
            await asyncio.wait_for(check(probe, *check_args), args.timeout)
            await asyncio.wait_for(compare_with_prefill(engine, probe), args.timeout)
            row["status"] = "passed"
        finally:
            try:
                await asyncio.wait_for(probe.session.abort(), args.timeout)
                await asyncio.wait_for(probe.reader, args.timeout)
            finally:
                row["events"] = [event.to_dict() for event in probe.events]
                print(json.dumps(row, ensure_ascii=False), flush=True)

    compact = {"max_visual_frames": 2, "compact_to_visual_frames": 1}
    try:
        await run("silence-and-close", check_silence)
        for control in renderer.controls:
            if control != "silence":
                await run(f"{control}-memory", check_memory, control, **compact)
        await run("cancel-and-abort", check_cancel)
    finally:
        engine.shutdown()


def parse_args():
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument(
        "--renderer",
        required=True,
        help="univia, ov21, or module:Class of a Renderer subclass",
    )
    parser.add_argument(
        "--images",
        nargs=6,
        metavar="PATH",
        help="six frames to stream, one per second; a moving square by default",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180,
        help="seconds allowed for each check, its comparison and its cleanup",
    )
    AsyncEngineArgs.add_cli_args(parser)
    return parse_engine_args(parser, parser)


def main():
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    asyncio.run(check_branching(parse_args()))


if __name__ == "__main__":
    main()
