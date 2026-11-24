# onevision-streaming（`ov-streaming`）

[English](README.md) | 简体中文

`onevision-streaming`（Python 包名：`ov-streaming`）为 **LLaVA-OneVision-2.5**（同时兼容 OV2.1 与 UniVIA）提供单脚本与 HTTP/SSE 服务端两种实时流式视频推理方式：

- **vLLM 后端（`backend="vllm"`）**：基于官方 `vllm==0.25.1`，安装 `ov-streaming` 后通过插件自动生效，无需定制 vLLM 分支。支持并发流式推理——模型生成回答期间仍持续接收后续画面。
- **Transformers 后端（`backend="transformers"`）**：仅需 PyTorch 与 `transformers`（`>=5.7.0,<5.10`），无需安装 vLLM，适合轻量试跑与单路调试。
- **多种流式输入**：支持视频文件、网络流 URL（如 `rtsp://`）、外部逐帧图像流（摄像头/自定义采样）以及实时 H.264/MPEG-TS 字节流。
- **支持 `frames` 与 `codec`（I+P 画布）双模式**：同一引擎实例可在不同会话间切换普通抽帧与 `codec` 模式。
- **内置 HTTP/SSE 服务与异步客户端**：支持多会话并发、长视频滑窗压缩、持续旁白（`enable=["caption"]`）与客户端控制的回答回写。

## 支持的模型

| `--renderer` | 模型系列 | 说明 |
| :--- | :--- | :--- |
| `ov25-window`（默认） | **LLaVA-OneVision-2.5** | [`lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct`](https://huggingface.co/lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct)，支持滑窗压缩 |
| `ov25` | **LLaVA-OneVision-2.5** | 不压缩上下文格式 |
| `ov21` / `ov21-window` | OV2.1 | OV2.1 权重兼容别名 |
| `univia` | UniVIA | Qwen3-VL 流式权重 |

## 安装

### 方式 1：vLLM 后端（实时并发）

```bash
python3.12 -m venv .venv-ov-vllm   # Python 3.10 / 3.11 / 3.12 均可
source .venv-ov-vllm/bin/activate

# 确保 torch 与机器 CUDA 版本一致
pip install "vllm==0.25.1" "transformers>=5.7.0,<5.10" "qwen-vl-utils>=0.0.14"
pip install "ov-streaming @ git+https://github.com/Kagura-0001/onevision-streaming.git"
```

### 方式 2：仅使用 Transformers（不装 vLLM）

如需在同一环境中使用 `codec` 模式，建议使用 **Python 3.11**（`codec-video-prep==0.3.0` 目前提供 Linux x86_64 的 Python 3.10/3.11 安装包，且要求 `numpy<2`）：

```bash
python3.11 -m venv .venv-ov-tf
source .venv-ov-tf/bin/activate

pip install "torch>=2.4" torchvision "transformers>=5.7.0,<5.10" "accelerate>=1.1.0" \
    "numpy>=1.23,<2" "opencv-python-headless<4.12" "qwen-vl-utils>=0.0.14"
pip install "ov-streaming @ git+https://github.com/Kagura-0001/onevision-streaming.git"

# 可选：启用 codec 模式（目前托管在 TestPyPI）
pip install --no-deps --index-url https://test.pypi.org/simple/ "codec-video-prep==0.3.0"
```

> **在 vLLM 环境中使用 `codec` 模式**
>
> `vllm==0.25.1` 依赖 `numpy>=2`，而 `codec-video-prep==0.3.0` 要求 Python 3.10/3.11 与 `numpy<2`，两者无法装在同一环境里。要在 `backend="vllm"` 下开启 `codec` 模式，另建一个轻量的 Python 3.11 解码环境（安装 `codec-video-prep==0.3.0`、`numpy<2`、`pillow`、`httpx`），通过环境变量或 `codec_python` 指定该环境的 Python 路径即可：
>
> ```bash
> export OV_STREAMING_CODEC_PYTHON=/path/to/.venv-codec-py311/bin/python
> ```

## 快速上手

### 视频文件流式推理（`frames` / `codec` 模式）

```python
from ov_streaming import CodecConfig, StreamingEngine

MODEL = "lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct"

with StreamingEngine(
    MODEL,
    backend="vllm",            # 不装 vLLM 时改为 "transformers"
    renderer="ov25-window",    # 提示词格式："ov25-window"（默认滑窗）或 "ov25"
    max_model_len=32768,       # 上下文长度上限，长视频可设至 262144
    max_pixels=258048,         # 单帧 / 单画布像素上限（常用 100352 / 258048 / 313000）
) as engine:
    # 1) 普通抽帧模式
    with engine.session(
        enable=[],                     # 默认主动提醒/检测；需要持续旁白时改为 ["caption"]
        max_visual_frames=96,          # 滑窗上限；3 分钟以内短视频可设为 None 不压缩
        compact_to_visual_frames=64,   # 触发压缩后保留的最近视觉帧数
        max_response_tokens=128,       # 单次回复长度上限（提醒类 128，时序描述可用 256）
        max_responses=3,               # 允许同时生成中的回答数上限
    ) as session:
        for event in session.stream_video(
            "example.mp4",
            video_backend="frames",
            fps=1.0,
            questions=[(0.0, "Notify me when the door opens.")],
        ):
            if event.type == "response.output_text.done":
                print(f"[{event.timestamp_sec:.1f}s] ({event.control}) {event.text}")

    # 2) 同一引擎切换为 codec I+P 模式（需满足 max_pixels >= pixel_budget）
    codec_config = CodecConfig(strategy="rdo", pixel_budget=258048, p_canvases=1)
    with engine.session(
        enable=[],
        max_visual_frames=192,         # I+1P 每秒 2 张画布，192 -> 128 对应约 96 秒 -> 64 秒
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

### 外部实时流与逐条推送

同一个 `session` 也支持外部逐帧输入（`stream_frames`）、实时 H.264/MPEG-TS 字节流（`stream_h264`）、网络流 URL（`stream_video("rtsp://...")`）以及手动单步推送（`send_frame`、`send_codec_window`、`send_question`）：

```python
# 外部逐帧输入（同步或异步可迭代对象，每项为 VideoFrame 或 (image, timestamp_sec)）
for event in session.stream_frames(camera_frames(), questions=[(0.0, "What is happening?")]):
    if event.type == "response.output_text.done":
        print(f"[{event.timestamp_sec:.1f}s] ({event.control}) {event.text}")

# 实时 H.264/MPEG-TS 字节流（任意分片，需声明源帧率与原始分辨率）
for event in session.stream_h264(ts_chunks, fps=30.0, width=1920, height=1080, codec_config=codec_config):
    if event.type == "response.output_text.done":
        print(f"[{event.timestamp_sec:.1f}s] ({event.control}) {event.text}")
```

如需全异步并发（一个协程推流、另一个协程读取 `events()`），使用 `AsyncStreamingEngine` 与 `AsyncLocalSession`。更完整的摄像头示例和逐条推送示例见 [快速上手指南](docs/quickstart_ov25.md)。

### 常用参数参考

| 典型场景 | `video_backend` | `fps` / `p_canvases` | `max_pixels` / `pixel_budget` | `max_visual_frames` → `compact_to` | `max_response_tokens` | `enable` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 通用默认 | `frames` / `codec` | `1.0` fps / `1` P | `258048` | `96` → `64`（`I+1P` 用 `192` → `128`） | `128` | `[]` |
| 主动提醒（OmniMMI PA） | `frames` | `1.0` fps | `258048` | `None`（直播用 `96` → `64`） | `128` | `[]` |
| 时序事件检测（OVO-Timing） | `codec` | `p_canvases=2`（I+2P） | `100352` | `None`（直播按 `3×` 换算或设 `None`） | `256` | `[]` |
| 持续实时旁白 | `frames` / `codec` | `1.0` fps / `1` P | `258048` | `96` → `64`（`I+1P` 用 `192` → `128`） | `128` | `["caption"]` |

- `codec` 模式下 `max_visual_frames` / `compact_to_visual_frames` 按画布数计数（每秒 `1 + p_canvases` 张），保持与 1 fps 普通帧相同的历史时长时按 `1 + p_canvases` 倍数放大即可。
- 持续旁白通过 `enable=["caption"]` 开启，可选配合 `caption_prompt` 与 `max_caption_tokens`。

## HTTP/SSE 服务模式

安装服务端依赖（`pip install "ov-streaming[serve] @ git+..."`）后启动：

```bash
ov-streaming serve lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct \
    --renderer ov25-window \
    --max-model-len 32768 \
    --mm-processor-kwargs '{"max_pixels":258048}'
```

服务默认监听 `127.0.0.1:8011`，支持 vLLM 引擎参数（通过 `ov-streaming serve --help=all` 查看），可通过 `OVClient` 异步连接多路会话。接口详情见 [README.md](README.md#http-endpoints)。

## 更多文档与应用

- [详细快速上手指南（含摄像头与手动推送）](docs/quickstart_ov25.md)
- [Codec 推理与网关部署](docs/codec.md)
- [提示词格式与自定义 Renderer](docs/renderers.md)
- [触发二次校验分支](docs/verification.md)
- [Packed QA 缓存](docs/packed_qa.md) · [Raw Logits](docs/packed_raw_logits.md) · [Visual Diffusion](docs/visual_diffusion.md)
- [语音交互应用（Pipecat + Qwen3-ASR / Qwen3-TTS）](apps/voice/README.md)

## 测试

CPU 单元测试无需加载模型权重：

```bash
pip install -e ".[serve,test]"
pytest
```

## 许可协议

Apache-2.0，详见 [LICENSE](LICENSE)。
