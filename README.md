# onevision-streaming (`ov-streaming`)

English | [简体中文](README_CN.md)

`onevision-streaming` (Python package: `ov-streaming`) provides real-time streaming video inference for **LLaVA-OneVision-2.5** (as well as OV2.1 and UniVIA) in both single-script Python and HTTP/SSE serving modes.

- **Concurrent real-time streaming on official `vllm==0.25.1`**: installs a vLLM plugin automatically (`vllm.general_plugins`) — no custom vLLM fork required. The model continues ingesting new frames while generating answers.
- **Lightweight `transformers` backend**: runs with PyTorch + `transformers` (`>=5.7.0,<5.10`) without installing vLLM, ideal for quick prototyping and single-stream debugging.
- **Unified streaming inputs**: supports local video files, network streams (e.g. `rtsp://`), external frame iterables (camera / custom sampling), and live H.264/MPEG-TS byte streams.
- **Both `frames` and `codec` (`I+P` canvas) modes**: switch freely between standard frame sampling and codec-based `I+P` canvases across sessions on the same engine.
- **Built-in HTTP/SSE service & async client**: share one model engine across multiple concurrent sessions, with sliding-window compaction, continuous captioning (`enable=["caption"]`), and client-controlled answer commits.

## Supported Models

| `--renderer` | Model Family | Default Model / Notes |
| :--- | :--- | :--- |
| `ov25-window` (default) | **LLaVA-OneVision-2.5** | [`lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct`](https://huggingface.co/lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct) with sliding-window compaction |
| `ov25` | **LLaVA-OneVision-2.5** | Uncompacted prompt layout |
| `ov21` / `ov21-window` | OV2.1 | Compatibility aliases for OV2.1 checkpoints |
| `univia` | UniVIA | Qwen3-VL streaming checkpoints |

## Installation

### Option 1: vLLM backend (concurrent real-time streaming)

```bash
python3.12 -m venv .venv-ov-vllm   # Python 3.10 / 3.11 / 3.12
source .venv-ov-vllm/bin/activate

# Make sure torch matches your machine's CUDA version
pip install "vllm==0.25.1" "transformers>=5.7.0,<5.10" "qwen-vl-utils>=0.0.14"
pip install "ov-streaming @ git+https://github.com/Kagura-0001/onevision-streaming.git"
```

### Option 2: Transformers backend (no vLLM required)

If you also want to use `codec` mode in the same environment, use **Python 3.11** (`codec-video-prep==0.3.0` currently provides Linux x86_64 wheels for Python 3.10/3.11 with `numpy<2`):

```bash
python3.11 -m venv .venv-ov-tf
source .venv-ov-tf/bin/activate

pip install "torch>=2.4" torchvision "transformers>=5.7.0,<5.10" "accelerate>=1.1.0" \
    "numpy>=1.23,<2" "opencv-python-headless<4.12" "qwen-vl-utils>=0.0.14"
pip install "ov-streaming @ git+https://github.com/Kagura-0001/onevision-streaming.git"

# Optional: enable codec I+P mode (currently hosted on TestPyPI)
pip install --no-deps --index-url https://test.pypi.org/simple/ "codec-video-prep==0.3.0"
```

> **Using `codec` mode together with `vllm==0.25.1`**
>
> `vllm==0.25.1` requires `numpy>=2`, whereas `codec-video-prep==0.3.0` requires Python 3.10/3.11 and `numpy<2`. To use `video_backend="codec"` with `backend="vllm"`, create a lightweight Python 3.11 environment containing `codec-video-prep==0.3.0`, `numpy<2`, `pillow`, and `httpx`, then point `OV_STREAMING_CODEC_PYTHON` (or `codec_python=...`) at its Python binary:
>
> ```bash
> export OV_STREAMING_CODEC_PYTHON=/path/to/.venv-codec-py311/bin/python
> ```

## Quick Start

### Video files (`frames` and `codec` modes)

```python
from ov_streaming import CodecConfig, StreamingEngine

MODEL = "lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct"

with StreamingEngine(
    MODEL,
    backend="vllm",            # or "transformers" (no vLLM required)
    renderer="ov25-window",    # "ov25-window" (sliding window) or "ov25"
    max_model_len=32768,       # up to 262144 for long uncompacted streams
    max_pixels=258048,         # per-frame / per-canvas pixel limit (e.g. 100352 / 258048 / 313000)
) as engine:
    # 1) Frame-sampled mode
    with engine.session(
        enable=[],                     # proactive alert default; set ["caption"] for continuous narration
        max_visual_frames=96,          # sliding-window limit (None disables compaction on <3 min clips)
        compact_to_visual_frames=64,   # visual frames retained after compaction
        max_response_tokens=128,       # per-response token limit (128 for alerts, 256 for timing)
        max_responses=3,               # max concurrent in-flight responses
    ) as session:
        for event in session.stream_video(
            "example.mp4",
            video_backend="frames",
            fps=1.0,
            questions=[(0.0, "Notify me when the door opens.")],
        ):
            if event.type == "response.output_text.done":
                print(f"[{event.timestamp_sec:.1f}s] ({event.control}) {event.text}")

    # 2) Switch to codec I+P mode on the same engine (requires max_pixels >= pixel_budget)
    codec_config = CodecConfig(strategy="rdo", pixel_budget=258048, p_canvases=1)
    with engine.session(
        enable=[],
        max_visual_frames=192,         # I+1P emits 2 canvases/s (192 -> 128 covers ~96s -> 64s)
        compact_to_visual_frames=128,
        max_response_tokens=128,
        max_responses=3,
    ) as session:
        for event in session.stream_video(
            "example.mp4",
            video_backend="codec",
            codec_config=codec_config,
            # codec_python="/path/to/.venv-codec-py311/bin/python",
            questions=[(0.0, "Notify me when the door opens.")],
        ):
            if event.type == "response.output_text.done":
                print(f"[{event.timestamp_sec:.1f}s] ({event.control}) {event.text}")
```

### Live streams & manual push

The same `session` accepts external decoded frames (`stream_frames`), live H.264/MPEG-TS byte chunks (`stream_h264`), network URLs (`stream_video("rtsp://...")`), or step-by-step pushes (`send_frame`, `send_codec_window`, `send_question`):

```python
# External decoded frames (sync or async iterable of VideoFrame or (image, timestamp_sec))
for event in session.stream_frames(camera_frames(), questions=[(0.0, "What is happening?")]):
    if event.type == "response.output_text.done":
        print(f"[{event.timestamp_sec:.1f}s] ({event.control}) {event.text}")

# Live H.264/MPEG-TS byte stream (arbitrary chunk sizes; declare CFR fps and native resolution)
for event in session.stream_h264(ts_chunks, fps=30.0, width=1920, height=1080, codec_config=codec_config):
    if event.type == "response.output_text.done":
        print(f"[{event.timestamp_sec:.1f}s] ({event.control}) {event.text}")
```

For full async concurrency (pushing inputs in one task while consuming `events()` in another), use `AsyncStreamingEngine` and `AsyncLocalSession`.

### Recommended Settings

| Profile | `video_backend` | `fps` / `p_canvases` | `max_pixels` / `pixel_budget` | `max_visual_frames` → `compact_to` | `max_response_tokens` | `enable` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| General default | `frames` or `codec` | `1.0` fps / `1` P | `258048` | `96` → `64` (`192` → `128` for `I+1P`) | `128` | `[]` |
| Proactive alert (OmniMMI PA) | `frames` | `1.0` fps | `258048` | `None` (or `96` → `64` for live) | `128` | `[]` |
| Event timing (OVO-Timing) | `codec` | `p_canvases=2` (I+2P) | `100352` | `None` (or `3×` horizon for live) | `256` | `[]` |
| Continuous narration | `frames` or `codec` | `1.0` fps / `1` P | `258048` | `96` → `64` (`192` → `128` for `I+1P`) | `128` | `["caption"]` |

- In `codec` mode, `max_visual_frames` and `compact_to_visual_frames` count canvases (`1 + p_canvases` per second). Scale them by `1 + p_canvases` to match the same time span as 1 fps `frames`.
- Continuous narration is controlled by `enable=["caption"]`, with optional `caption_prompt` and `max_caption_tokens`.

## HTTP/SSE Serving

Install the serving extras (`pip install "ov-streaming[serve] @ git+..."`) and start the server:

```bash
ov-streaming serve lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct \
    --renderer ov25-window \
    --max-model-len 32768 \
    --mm-processor-kwargs '{"max_pixels":258048}'
```

The service listens on `127.0.0.1:8011` by default and accepts standard vLLM engine flags (`ov-streaming serve --help=all`). Tensor parallelism is supported; pipeline/data/context parallelism and speculative decoding are not.

### Async Client (`OVClient`)

```python
import asyncio
from pathlib import Path
from ov_streaming import OVClient


async def main():
    client = OVClient("http://127.0.0.1:8011")
    async with client.session(max_visual_frames=96, compact_to_visual_frames=64) as session:
        async def print_answers():
            async for event in session.responses():
                print(f"[{event.timestamp_sec:.1f}s] ({event.control}) {event.text}")

        reader = asyncio.create_task(print_answers())
        for second, path in enumerate(sorted(Path("frames").glob("*.jpg"))):
            content = [{"type": "image_url", "image_url": {"url": path}, "timestamp_sec": second}]
            if second == 3:
                content.append({"type": "text", "text": "Tell me when the water boils."})
            await session.send_messages(messages=[{"role": "user", "content": content}])
        await session.close()
        await reader


asyncio.run(main())
```

### HTTP Endpoints

| Method and Path | Description |
| :--- | :--- |
| `POST /v1/ov/sessions` | Create a session (`max_responses`, `max_visual_frames`, `compact_to_visual_frames`, `enable`, `caption_prompt`, `max_caption_tokens`, `answer_commit`) |
| `POST /v1/ov/sessions/{sid}/messages` | Queue one input turn (frames and/or question) |
| `GET /v1/ov/sessions/{sid}/events` | Server-Sent Events stream (`input.done`, `response.created`, `response.output_text.done`, `response.done`, `context.compacted`, `session.closed`) |
| `DELETE /v1/ov/sessions/{sid}/responses/{rid}` | Cancel a running response branch |
| `POST /v1/ov/sessions/{sid}/responses/{rid}/commit` | Commit client-delivered answer text when `answer_commit="client"` |
| `POST /v1/ov/sessions/{sid}/close` | Drain queued inputs and running responses, then close the session |
| `DELETE /v1/ov/sessions/{sid}` | Abort the session immediately |
| `GET /v1/models` | Return the served model name |

### Long-Session Compaction

When a session approaches `--max-model-len` or exceeds `max_visual_frames`, it automatically compacts by starting a new context window from the newest turns up to `compact_to_visual_frames`, while preserving questions and committed answers from dropped turns as compact text memory. Running response branches finish cleanly on their existing KV cache.

## Documentation & Apps

- [中文说明 (README_CN.md)](README_CN.md) / [快速上手指南](docs/quickstart_ov25.md)
- [Codec inference (client & WebSocket server gateway)](docs/codec.md)
- [Prompt renderers & custom formats](docs/renderers.md)
- [Verification branch](docs/verification.md)
- [Packed QA caches](docs/packed_qa.md) · [Raw logits](docs/packed_raw_logits.md) · [Visual diffusion](docs/visual_diffusion.md)
- [Voice app (Pipecat + Qwen3-ASR / Qwen3-TTS)](apps/voice/README.md)

## Tests

Unit tests run on CPU without loading model weights:

```bash
pip install -e ".[serve,test]"
pytest
```

For GPU branching parity checks against one-shot prefill:

```bash
python scripts/check_branching.py \
    --renderer ov25-window \
    --model lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct \
    --max-model-len 16384
```

## License

Apache-2.0. See [LICENSE](LICENSE).
