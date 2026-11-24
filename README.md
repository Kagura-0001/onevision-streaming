# onevision-streaming

**English** | [中文](README_CN.md)

Streaming video inference for [LLaVA-OneVision-2.5-4B-Instruct](https://huggingface.co/lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct) (also compatible with OV2.1 and UniVIA).

The package (`ov-streaming`) runs with two backends: a plugin for official `vllm==0.25.1` that keeps reading incoming frames while answers decode in parallel, and a `transformers` backend that needs no vLLM for single-stream testing. Both accept local video files, camera or RTSP streams, and raw H.264/MPEG-TS bytes, and can switch between frame sampling (`frames`) and `codec` (`I+P` canvases).

## Install

With vLLM (concurrent streaming, Python 3.10–3.12):

```bash
pip install "vllm==0.25.1" "transformers>=5.7.0,<5.10" "qwen-vl-utils>=0.0.14"
pip install "ov-streaming @ git+https://github.com/Kagura-0001/onevision-streaming.git"
```

Without vLLM (PyTorch + Transformers only):

```bash
pip install "ov-streaming[transformers] @ git+https://github.com/Kagura-0001/onevision-streaming.git"
```

For `codec` mode, install `codec-video-prep==0.3.0` in a Python 3.10/3.11 environment (`numpy<2`):

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple \
    "codec-video-prep==0.3.0"
```

Because `vllm==0.25.1` requires `numpy>=2`, using `codec` with `backend="vllm"` delegates decoding to a separate Python 3.11 environment containing `codec-video-prep==0.3.0`, `numpy<2`, `pillow`, and `httpx`. Point `OV_STREAMING_CODEC_PYTHON=/path/to/py311/bin/python` (or pass `codec_python=...`) at that interpreter.

## Quickstart

```python
from ov_streaming import StreamingEngine

MODEL = "lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct"

with StreamingEngine(MODEL, backend="vllm", max_model_len=32768, max_pixels=258048) as engine:
    with engine.session(
        max_visual_frames=96,
        compact_to_visual_frames=64,
        max_response_tokens=128,
    ) as session:
        for ev in session.stream_video(
            "example.mp4",
            fps=1.0,
            questions=[(0.0, "Notify me when the door opens.")],
        ):
            if ev.type == "response.output_text.done":
                print(f"[{ev.timestamp_sec:.1f}s] {ev.text}")
```

Pass `backend="transformers"` to run without vLLM. To turn on continuous narration, pass `enable=["caption"]` to `engine.session(...)` (with optional `caption_prompt` and `max_caption_tokens`).

### Codec mode

Pass `video_backend="codec"` with a `CodecConfig`. Because `I+1P` produces 2 canvases per second, scale `max_visual_frames` and `compact_to_visual_frames` by `1 + p_canvases` to keep the same time span as 1 fps frames (or set both to `None` for short clips under ~3 minutes). One engine can switch between `frames` and `codec` across sessions as long as `max_pixels >= pixel_budget`:

```python
from ov_streaming import CodecConfig

codec = CodecConfig(strategy="rdo", pixel_budget=258048, p_canvases=1)
with engine.session(max_visual_frames=192, compact_to_visual_frames=128) as session:
    for ev in session.stream_video(
        "example.mp4",
        video_backend="codec",
        codec_config=codec,
        questions=[(0.0, "Notify me when the door opens.")],
    ):
        if ev.type == "response.output_text.done":
            print(f"[{ev.timestamp_sec:.1f}s] {ev.text}")
```

### Live streams

`stream_frames` takes any sync or async iterable of `VideoFrame` or `(image, timestamp_sec)` tuples, and `stream_h264` takes raw H.264/MPEG-TS byte chunks (with `fps`, `width`, `height`). `stream_video` also opens `rtsp://` URLs. For manual pacing, call `send_frame`, `send_codec_window`, and `send_question` (or use `AsyncStreamingEngine` for concurrent async push and event consumption).

Runnable scripts are in [examples/stream_video.py](examples/stream_video.py) and [examples/stream_camera.py](examples/stream_camera.py).

## Serving

Install the server extras (`pip install "ov-streaming[serve] @ git+..."`) and start the HTTP/SSE service:

```bash
ov-streaming serve lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct \
    --renderer ov25-window \
    --max-model-len 32768 \
    --mm-processor-kwargs '{"max_pixels":258048}'
```

The server listens on `127.0.0.1:8011` and accepts standard vLLM engine flags (`ov-streaming serve --help=all`). Supported `--renderer` values are `ov25-window` (default), `ov25`, `ov21`, `ov21-window`, and `univia`. See [examples/async_client.py](examples/async_client.py) for `OVClient` usage.

| Endpoint | Description |
| :--- | :--- |
| `POST /v1/ov/sessions` | Create a session |
| `POST /v1/ov/sessions/{sid}/messages` | Queue a frame and/or question |
| `GET /v1/ov/sessions/{sid}/events` | SSE stream (`input.done`, `response.output_text.done`, `response.done`, `context.compacted`, `session.closed`) |
| `DELETE /v1/ov/sessions/{sid}/responses/{rid}` | Cancel a running response |
| `POST /v1/ov/sessions/{sid}/responses/{rid}/commit` | Commit delivered text when `answer_commit="client"` |
| `POST /v1/ov/sessions/{sid}/close` | Drain and close |
| `DELETE /v1/ov/sessions/{sid}` | Abort immediately |

## Docs

- [Codec client & WebSocket gateway](docs/codec.md)
- [Prompt renderers](docs/renderers.md)
- [Voice app (Pipecat + Qwen3-ASR / Qwen3-TTS)](apps/voice/README.md)

## Tests

```bash
pip install -e ".[serve,test]"
pytest
```

## License

Apache-2.0. See [LICENSE](LICENSE).
