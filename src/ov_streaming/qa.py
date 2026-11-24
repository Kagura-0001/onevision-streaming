"""Serve packed, single-generation OV QA using the existing OV vLLM runtime.

The renderer supplies the checkpoint's image processing and position encoding.
This entry point never samples a streaming control token or opens a session.
"""

import argparse
import asyncio
import contextlib
import math
import os
import time
import uuid
from collections.abc import Mapping

from .image_cache import DecodedImageCache, PreparedFrameCache, decode_rgb
from .visual_diffusion import apply_visual_diffusion, validate_visual_diffusion


def _render_image(renderer, part, image_cache):
    from PIL import Image

    decoded = (image_cache.decode(part["data"])
               if image_cache is not None else decode_rgb(part["data"]))
    with decoded as image:
        if "codec" in part:
            return renderer.render_codec_frame(
                image, part["timestamp"], codec=part["codec"],
                frame_index=part.get("frame_index"), fps=part["fps"],
            )
        if getattr(renderer, "preserve_source_pixels", False):
            source_width, source_height = map(int, part["source_size"])
            if not (0 < source_width <= image.width and 0 < source_height <= image.height):
                raise ValueError("source_size must fit inside the encoded image")
            # Remove only declared Qwen padding, then use OV's patch grid.
            image = image.crop((0, 0, source_width, source_height))
            processor = renderer.image_processor
            factor = int(processor.patch_size * processor.merge_size)
            width = (source_width + factor - 1) // factor * factor
            height = (source_height + factor - 1) // factor * factor
            padded = Image.new("RGB", (width, height))
            padded.paste(image, (0, 0))
            image = padded
        return renderer.render_frame(
            image, part["timestamp"], frame_index=part["frame_index"], fps=part["fps"],
        )


def render_packed(renderer, messages, *, image_cache=None, prepared_frame_cache=None):
    """Preserve message/content order, including the question after the frames."""
    chunks, frames = [], []
    if (not messages or messages[0]["role"] != "system") and renderer.system_prompt:
        chunks.append(f"<|im_start|>system\n{renderer.system_prompt}<|im_end|>\n")
    for message in messages:
        role = message["role"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported message role: {role}")
        chunks.append(f"<|im_start|>{role}\n")
        content = message["content"]
        if isinstance(content, str):
            chunks.append(content)
        else:
            for part in content:
                if part["type"] == "text":
                    chunks.append(part["text"])
                elif part["type"] == "image":
                    # Codec position maps are per-canvas metadata. Keep this
                    # path uncached until they are explicitly part of a key.
                    if prepared_frame_cache is None or "codec" in part:
                        frame = _render_image(renderer, part, image_cache)
                    else:
                        preserve = getattr(renderer, "preserve_source_pixels", False)
                        source_size = tuple(map(int, part["source_size"])) if preserve else None
                        # The bound method also retains its renderer, so an old
                        # cache entry cannot match a recycled object identity.
                        key = (id(renderer), renderer.render_frame, preserve,
                               source_size, part["data"],
                               part["timestamp"], part["frame_index"], part["fps"])
                        frame = prepared_frame_cache.get_or_prepare(
                            key, lambda: _render_image(renderer, part, image_cache),
                        )
                    frame_text = frame.text
                    if "timestamp_text" in part:
                        # Some published polling protocols supply an interval as
                        # a separate text part. Keep OV vision tokens and source
                        # patch positions while replacing only the time label.
                        timestamp_text = part["timestamp_text"]
                        if not isinstance(timestamp_text, str):
                            raise ValueError("timestamp_text must be a string")
                        _, marker, vision_text = frame_text.partition("<|vision_start|>")
                        if not marker:
                            raise ValueError("timestamp_text requires an OV vision frame")
                        frame_text = timestamp_text + marker + vision_text
                    chunks.append(frame_text)
                    frames.append(frame)
                else:
                    raise ValueError(f"unsupported content type: {part['type']}")
        chunks.append("<|im_end|>\n")
    chunks.append("<|im_start|>assistant\n")
    return "".join(chunks), frames


def engine_prompt(backend, prompt, frames):
    from vllm.inputs import TokensPrompt

    payload = TokensPrompt(
        prompt_token_ids=backend.tokenizer.encode(prompt, add_special_tokens=False)
    )
    if frames:
        images = [frame.data for frame in frames]
        if isinstance(images[0], Mapping):
            import torch

            images = {
                key: torch.cat([image[key] for image in images]) for key in images[0]
            }
        payload["multi_modal_data"] = {"image": images}
    return payload


SCORE_OUTPUT_MODES = ("raw_logprobs", "raw_logits")


class LogprobUnavailableError(RuntimeError):
    """The engine did not return complete finite selected-token scores."""


def validate_logprob_token_ids(value, *, field="logprob_token_ids"):
    """Validate the opt-in engine request without changing ordinary sampling."""
    if value is None:
        return None
    if not isinstance(value, list) or not 1 <= len(value) <= 128:
        raise ValueError(f"{field} must be a list of 1 to 128 token IDs")
    if any(type(token_id) is not int or token_id < 0 for token_id in value):
        raise ValueError(f"{field} must contain nonnegative integer IDs")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} must not contain duplicate IDs")
    return list(value)


def validate_score_request(body, score_output_mode):
    """Reject score-channel ambiguity before rendering or engine generation."""
    if score_output_mode not in SCORE_OUTPUT_MODES:
        raise ValueError(f"unsupported score_output_mode: {score_output_mode}")
    if "logprob_token_ids" in body and "logits_token_ids" in body:
        raise ValueError("logprob_token_ids and logits_token_ids are mutually exclusive")
    field = ("logits_token_ids" if score_output_mode == "raw_logits"
             else "logprob_token_ids")
    forbidden = ("logprob_token_ids" if score_output_mode == "raw_logits"
                 else "logits_token_ids")
    if forbidden in body:
        raise ValueError(f"{forbidden} is not supported in {score_output_mode} mode")
    if field == "logits_token_ids" and field in body and body[field] is None:
        raise ValueError("logits_token_ids must be a list of 1 to 128 token IDs")
    return validate_logprob_token_ids(body.get(field), field=field)


def _first_token_values(completion, token_ids, *, kind):
    # vLLM's engine-wide raw_logits mode deliberately uses the same Logprob
    # carrier. Its .logprob float is an unnormalized logit in that mode.
    logprobs = getattr(completion, "logprobs", None)
    if not logprobs or not isinstance(logprobs[0], Mapping):
        raise LogprobUnavailableError(f"vLLM returned no first-token {kind}")
    first = logprobs[0]
    missing = [token_id for token_id in token_ids if token_id not in first]
    if missing:
        raise LogprobUnavailableError(
            f"vLLM first-token {kind} missing requested token IDs: {missing}"
        )
    values = {}
    for token_id in token_ids:
        try:
            value = float(first[token_id].logprob)
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            raise LogprobUnavailableError(
                f"vLLM returned invalid {kind[:-1]} for token ID {token_id}"
            ) from error
        if not math.isfinite(value):
            raise LogprobUnavailableError(
                f"vLLM returned nonfinite {kind[:-1]} for token ID {token_id}"
            )
        values[str(token_id)] = value
    return values


def first_token_logprobs(completion, token_ids):
    """Keep requested first-token raw logprobs; never top-k/zero-fill/normalize."""
    return _first_token_values(completion, token_ids, kind="logprobs")


def first_token_logits(completion, token_ids):
    """Read the raw-logits engine carrier without softmax or score transforms."""
    return _first_token_values(completion, token_ids, kind="logits")


async def answer_once(
    backend, payload, *, max_tokens, stop, request_id,
    temperature=0.0, top_p=1.0, top_k=-1, seed=None,
    logprob_token_ids=None, logits_token_ids=None,
    score_output_mode="raw_logprobs",
):
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    requested = {}
    if logprob_token_ids is not None:
        requested["logprob_token_ids"] = logprob_token_ids
    if logits_token_ids is not None:
        requested["logits_token_ids"] = logits_token_ids
    token_ids = validate_score_request(requested, score_output_mode)
    optional = {} if token_ids is None else {"logprob_token_ids": token_ids}
    # No control-token selection, branch request, or response-token prefill.
    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        max_tokens=max_tokens,
        stop=stop or None,
        skip_special_tokens=False,
        output_kind=RequestOutputKind.FINAL_ONLY,
        **optional,
    )
    result = None
    async for output in backend.engine.generate(payload, params, request_id):
        result = output
    if result is None or not result.outputs:
        raise RuntimeError("vLLM returned no completion")
    completion = result.outputs[0]
    if completion.finish_reason == "error":
        raise RuntimeError("vLLM generation failed")
    response = {
        "text": completion.text,
        "finish_reason": completion.finish_reason,
        "usage": {
            "input_tokens": len(result.prompt_token_ids),
            "output_tokens": len(completion.token_ids),
        },
    }
    if token_ids is not None:
        if score_output_mode == "raw_logits":
            response["first_token_logits"] = first_token_logits(completion, token_ids)
            response["value_semantics"] = "pre_sampling_logits"
        else:
            response["first_token_logprobs"] = first_token_logprobs(completion, token_ids)
    return response


def build_app(
    renderer, lifespan, *, served_model, preprocess_workers=4,
    decoded_image_cache_mib=0, prepared_frame_cache_mib=0,
    score_output_mode="raw_logprobs",
):
    from fastapi import FastAPI, HTTPException

    validate_score_request({}, score_output_mode)
    app = FastAPI(lifespan=lifespan)
    app.state.score_output_mode = score_output_mode
    preparation = asyncio.Semaphore(preprocess_workers)
    if decoded_image_cache_mib < 0:
        raise ValueError("decoded_image_cache_mib must be nonnegative")
    image_cache = (DecodedImageCache(decoded_image_cache_mib * 1024 * 1024)
                   if decoded_image_cache_mib else None)
    app.state.decoded_image_cache = image_cache
    if prepared_frame_cache_mib < 0:
        raise ValueError("prepared_frame_cache_mib must be nonnegative")
    frame_cache = (PreparedFrameCache(prepared_frame_cache_mib * 1024 * 1024)
                   if prepared_frame_cache_mib else None)
    app.state.prepared_frame_cache = frame_cache

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": served_model, "object": "model", "owned_by": "ov-qa",
                      **({"score_output_mode": score_output_mode}
                         if score_output_mode == "raw_logits" else {})}],
        }

    @app.post("/v1/qa")
    async def answer(body: dict):
        started = time.monotonic()
        request_id = str(body.get("request_id") or uuid.uuid4().hex)
        backend = app.state.backend
        try:
            token_ids = validate_score_request(body, score_output_mode)
            diffusion = validate_visual_diffusion(body.get("visual_diffusion"))
            max_tokens = int(body.get("max_tokens", 256))
            if not 1 <= max_tokens <= 4096:
                raise ValueError("max_tokens must be between 1 and 4096")
            temperature = float(body.get("temperature", 0))
            top_p = float(body.get("top_p", 1))
            top_k = int(body.get("top_k", -1))
            seed = body.get("seed")
            if seed is not None:
                seed = int(seed)

            def prepare():
                prompt, frames = render_packed(
                    renderer, body["messages"], image_cache=image_cache,
                    prepared_frame_cache=frame_cache,
                )
                payload = engine_prompt(backend, prompt, frames)
                if (
                    len(payload["prompt_token_ids"]) + max_tokens
                    > backend.max_model_len
                ):
                    raise ValueError(
                        "packed QA prompt exceeds max_model_len; frames were not truncated"
                    )
                payload, diffusion_metadata = apply_visual_diffusion(payload, diffusion)
                return payload, len(frames), diffusion_metadata

            async with preparation:
                payload, frame_count, diffusion_metadata = await asyncio.to_thread(prepare)
            result = await answer_once(
                backend,
                payload,
                max_tokens=max_tokens,
                stop=body.get("stop"),
                request_id=request_id,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                **({("logits_token_ids" if score_output_mode == "raw_logits"
                     else "logprob_token_ids"): token_ids}
                   if token_ids is not None else {}),
                **({"score_output_mode": score_output_mode}
                   if score_output_mode == "raw_logits" else {}),
            )
            return {
                "request_id": request_id,
                "frames": frame_count,
                "elapsed_sec": time.monotonic() - started,
                **result,
                **({"visual_diffusion": diffusion_metadata}
                   if diffusion_metadata is not None else {}),
            }
        except LogprobUnavailableError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except asyncio.CancelledError:
            await backend.abort(request_id)
            raise

    return app


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--renderer", default="ov21",
        help="univia, ov21, or module:Class of a Renderer subclass",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--max-pixels", type=int, default=None,
        help="Override the renderer's mm_processor_kwargs max_pixels (default: the renderer's own)",
    )
    parser.add_argument(
        "--prepared-frame-cache-mib", type=int, default=0,
        help="Per-process byte-bounded prepared CPU frame cache in MiB (0 disables)",
    )
    parser.add_argument(
        "--preserve-source-pixels",
        action="store_true",
        help="Use explicit source_size, pad to the OV patch grid, and disable image resizing",
    )
    parser.add_argument("--preprocess-workers", type=int, default=4)
    parser.add_argument(
        "--decoded-image-cache-mib", type=int, default=0,
        help="Per-process byte-bounded decoded RGB cache in MiB (0 disables)",
    )
    parser.add_argument(
        "--score-output-mode", choices=SCORE_OUTPUT_MODES, default="raw_logprobs",
        help="Engine-wide selected-token score channel; raw_logits rejects legacy logprob requests",
    )
    return parser


def parse_args(argv=None, parser=None):
    """CLI flags over the renderer's engine defaults, merged as ``serve`` merges them."""
    from ov_streaming.serve import parse_engine_args

    parser = parser or argument_parser()
    args = parse_engine_args(parser, parser, argv)
    if args.max_pixels is not None:
        # --max-pixels is this entry point's mm_processor_kwargs flag.
        args.mm_processor_kwargs = (
            getattr(args, "mm_processor_kwargs", {}) | {"max_pixels": args.max_pixels}
        )
    return args


def build_engine_args(args):
    """Bind the declared HTTP score channel to the actual vLLM engine mode."""
    from vllm.engine.arg_utils import AsyncEngineArgs

    renderer_args = {name: getattr(args, name) for name in args.renderer_class.engine_args}
    if hasattr(args, "mm_processor_kwargs"):
        renderer_args["mm_processor_kwargs"] = args.mm_processor_kwargs
    return AsyncEngineArgs(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        async_scheduling=args.async_scheduling,
        logprobs_mode=args.score_output_mode,
        **renderer_args,
    )


def main():
    parser = argument_parser()
    args = parse_args(parser=parser)
    if args.decoded_image_cache_mib < 0:
        parser.error("--decoded-image-cache-mib must be nonnegative")
    if args.prepared_frame_cache_mib < 0:
        parser.error("--prepared-frame-cache-mib must be nonnegative")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    import uvicorn
    from ov_streaming.backend import VLLMBackend
    from vllm.tokenizers import cached_tokenizer_from_config
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = build_engine_args(args)
    config = engine_args.create_model_config()
    renderer = args.renderer_class(cached_tokenizer_from_config(config), config)
    if args.preserve_source_pixels and not hasattr(renderer, "image_processor"):
        # Source pixels are padded to, and then kept on, the renderer's own patch grid.
        parser.error(f"--preserve-source-pixels needs a renderer with an image_processor; {args.renderer} has none")
    renderer.preserve_source_pixels = args.preserve_source_pixels
    if args.preserve_source_pixels:
        renderer.image_processor.do_resize = False

    @contextlib.asynccontextmanager
    async def lifespan(app):
        engine = AsyncLLM.from_engine_args(engine_args)
        app.state.backend = VLLMBackend(engine)
        try:
            yield
        finally:
            engine.shutdown()

    app = build_app(
        renderer,
        lifespan,
        served_model=args.model,
        preprocess_workers=args.preprocess_workers,
        decoded_image_cache_mib=args.decoded_image_cache_mib,
        prepared_frame_cache_mib=args.prepared_frame_cache_mib,
        score_output_mode=args.score_output_mode,
    )
    uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=30)


if __name__ == "__main__":
    main()
