# onevision-streaming

[English](README.md) | **中文**

[LLaVA-OneVision-2.5-4B-Instruct](https://huggingface.co/lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct) 的流式视频推理库（同时兼容 OV2.1 和 UniVIA）。

安装包名为 `ov-streaming`，提供两种后端：基于官方 `vllm==0.25.1` 的插件后端（模型生成回答期间仍持续接收后续画面），以及不依赖 vLLM 的 `transformers` 单路后端。两种后端均支持视频文件、摄像头/RTSP 流、原始 H.264/MPEG-TS 字节流，可在抽帧（`frames`）和 `codec`（I+P 画布）两种模式间切换。

## 安装

使用 vLLM（实时并发，Python 3.10–3.12）：

```bash
pip install "vllm==0.25.1" "transformers>=5.7.0,<5.10" "qwen-vl-utils>=0.0.14"
pip install "ov-streaming @ git+https://github.com/Kagura-0001/onevision-streaming.git"
```

仅使用 Transformers（不装 vLLM）：

```bash
pip install "ov-streaming[transformers] @ git+https://github.com/Kagura-0001/onevision-streaming.git"
```

使用 `codec` 模式时，在 Python 3.10/3.11 环境（`numpy<2`）安装 `codec-video-prep==0.3.0`：

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple \
    "codec-video-prep==0.3.0"
```

`vllm==0.25.1` 依赖 `numpy>=2`，在 `backend="vllm"` 下开启 `codec` 模式时，另建一个 Python 3.11 环境（安装 `codec-video-prep==0.3.0`、`numpy<2`、`pillow`、`httpx`），设置 `OV_STREAMING_CODEC_PYTHON=/path/to/py311/bin/python`（或传入 `codec_python` 参数）即可。

## 快速开始

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

不装 vLLM 时传 `backend="transformers"`。需要持续旁白时在 `engine.session(...)` 中加 `enable=["caption"]`（可选 `caption_prompt`、`max_caption_tokens`）。

### Codec 模式

传入 `video_backend="codec"` 和 `CodecConfig`。`I+1P` 每秒输出 2 张画布，保持与 1 fps 相同的历史时长时，把 `max_visual_frames` 和 `compact_to_visual_frames` 按 `1 + p_canvases` 倍数放大即可（3 分钟以内短视频也可都设为 `None`）。同一个引擎只要满足 `max_pixels >= pixel_budget`，可在不同会话间切换 `frames` 和 `codec`：

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

### 实时流

`stream_frames` 接受任意同步或异步可迭代对象（每项为 `VideoFrame` 或 `(image, timestamp_sec)`），`stream_h264` 接受原始 H.264/MPEG-TS 字节流（需指定 `fps`、`width`、`height`），`stream_video` 也支持 `rtsp://` 地址。需要手动控制节奏时可调用 `send_frame`、`send_codec_window`、`send_question`，或使用 `AsyncStreamingEngine` 做全异步推流与事件读取。

完整可运行脚本见 [examples/stream_video.py](examples/stream_video.py) 和 [examples/stream_camera.py](examples/stream_camera.py)。

## 服务端模式

安装服务端依赖（`pip install "ov-streaming[serve] @ git+..."`）后启动：

```bash
ov-streaming serve lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct \
    --renderer ov25-window \
    --max-model-len 32768 \
    --mm-processor-kwargs '{"max_pixels":258048}'
```

默认监听 `127.0.0.1:8011`，支持 vLLM 引擎参数（`ov-streaming serve --help=all`）。`--renderer` 支持 `ov25-window`（默认）、`ov25`、`ov21`、`ov21-window`、`univia`。异步客户端调用见 [examples/async_client.py](examples/async_client.py)，接口列表见 [README.md](README.md#serving)。

## 更多文档

- [Codec 客户端与 WebSocket 网关](docs/codec.md)
- [提示词 Renderer 说明](docs/renderers.md)
- [语音交互示例（Pipecat + Qwen3-ASR / Qwen3-TTS）](apps/voice/README.md)

## 测试

```bash
pip install -e ".[serve,test]"
pytest
```

## 许可

Apache-2.0，详见 [LICENSE](LICENSE)。
