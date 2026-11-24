# LLaVA-OneVision-2.5 流式推理快速上手

`ov-streaming` 为 `LLaVA-OneVision-2.5-4B-Instruct` 提供单脚本和服务端两种流式推理方式：

- **vLLM 后端（`backend="vllm"`）**：基于官方 `vllm==0.25.1`，安装 `ov-streaming` 后自动生效。支持并发流式推理——模型生成回答期间仍持续接收后续画面。
- **Transformers 后端（`backend="transformers"`）**：仅需 PyTorch 与 `transformers`（`>=5.7.0,<5.10`），无需安装 vLLM。适合轻量试跑、调试和单路演示（生成回答期间暂停接收新帧）。
- **多种输入方式**：支持视频文件、直播 URL（如 `rtsp://`）、外部逐帧图像流，以及实时 H.264/MPEG-TS 字节流；均可在普通抽帧（`frames`）与 `codec`（I+P 画布）两种模式间切换。

---

## 安装

### 方式 1：仅使用 Transformers（不装 vLLM）

如需在同一环境使用 `codec` 模式，建议使用 **Python 3.11**（`codec-video-prep==0.3.0` 目前仅提供 Linux x86_64 的 Python 3.10 / 3.11 安装包，且要求 `numpy<2`）：

```bash
python3.11 -m venv .venv-ov25-tf
source .venv-ov25-tf/bin/activate

pip install "torch>=2.4" torchvision "transformers>=5.7.0,<5.10" "accelerate>=1.1.0" \
    "numpy>=1.23,<2" "opencv-python-headless<4.12" "qwen-vl-utils>=0.0.14"
pip install "ov-streaming @ git+https://github.com/Kagura-0001/onevision-streaming.git"

# 可选：启用 codec 模式
pip install --no-deps --index-url https://test.pypi.org/simple/ "codec-video-prep==0.3.0"
```

### 方式 2：使用 vLLM（实时并发）

```bash
python3.12 -m venv .venv-ov25-vllm   # Python 3.10 / 3.11 / 3.12 均可
source .venv-ov25-vllm/bin/activate

# 确保 torch / torchcodec 与机器 CUDA 版本一致（例如 cu129）
pip install "vllm==0.25.1" "transformers>=5.7.0,<5.10" "qwen-vl-utils>=0.0.14"
pip install "ov-streaming @ git+https://github.com/Kagura-0001/onevision-streaming.git"
```

> **在 vLLM 环境中使用 `codec` 模式**
>
> `vllm==0.25.1` 依赖 `numpy>=2`，而 `codec-video-prep==0.3.0` 要求 Python 3.10/3.11 与 `numpy<2`，两者无法装在同一个环境里。如果要在 `backend="vllm"` 下开启 `codec` 模式，另建一个轻量的 Python 3.11 解码环境（安装 `codec-video-prep==0.3.0`、`numpy<2`、`pillow`、`httpx`），并通过环境变量或 `codec_python` 参数指定该环境的 Python 路径即可：
>
> ```bash
> export OV_STREAMING_CODEC_PYTHON=/path/to/.venv-codec-py311/bin/python
> ```
>
> 如果采用方式 1 在 Python 3.11 单环境中安装了 `codec-video-prep`，无需此配置。

---

## 使用示例

### 1. 视频文件流式推理（普通帧 / Codec）

下面的脚本把常用参数全部展开，方便按任务调整：

```python
from ov_streaming import CodecConfig, StreamingEngine

MODEL = "lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct"


def main():
    with StreamingEngine(
        MODEL,
        backend="vllm",          # 不装 vLLM 时改为 "transformers"
        renderer="ov25-window",  # 提示词格式："ov25-window"（默认滑窗）或 "ov25"
        max_model_len=32768,     # 上下文长度上限，长视频可设至 262144
        max_pixels=258048,       # 单帧/单画布像素上限（常用 100352 / 258048 / 313000）
    ) as engine:
        # 1) 普通抽帧模式
        with engine.session(
            enable=[],                     # 默认主动提醒/检测；需要持续旁白时改为 ["caption"]
            max_visual_frames=96,          # 滑窗上限；3 分钟以内短视频可设为 None 不压缩
            compact_to_visual_frames=64,   # 触发压缩后保留的最近视觉帧数
            max_response_tokens=128,       # 单次回复长度上限（提醒类 128，时序描述可用 256）
            max_responses=3,               # 允许同时生成中的回答数上限
        ) as session:
            for ev in session.stream_video(
                "demo.mp4",
                video_backend="frames",
                fps=1.0,
                questions=[(0.0, "Notify me when the door opens.")],
            ):
                if ev.type == "response.output_text.done":
                    print(f"[{ev.timestamp_sec:.1f}s] ({ev.control}) {ev.text}")

        # 2) 同一引擎上切换为 codec I+P 模式（需满足 max_pixels >= pixel_budget）
        # codec 按画布数计窗（I+1P 每秒 2 张），如需保留约 96s->64s 历史可设为 192 / 128
        codec_config = CodecConfig(strategy="rdo", pixel_budget=258048, p_canvases=1)
        with engine.session(
            enable=[],
            max_visual_frames=192,
            compact_to_visual_frames=128,
            max_response_tokens=128,
            max_responses=3,
        ) as session:
            for ev in session.stream_video(
                "demo.mp4",
                video_backend="codec",
                codec_config=codec_config,
                # codec_python="/path/to/.venv-codec-py311/bin/python",
                questions=[(0.0, "Notify me when the door opens.")],
            ):
                if ev.type == "response.output_text.done":
                    print(f"[{ev.timestamp_sec:.1f}s] ({ev.control}) {ev.text}")


if __name__ == "__main__":
    main()
```

### 2. 外部实时流（摄像头、直播流、自定义采样）

`stream_frames` 接受任意同步或异步可迭代对象，每项为 `VideoFrame` 或 `(image, timestamp_sec[, frame_index[, fps]])`。输入按需读取，不会将整段视频提前载入内存；同步迭代器的阻塞读取在后台线程执行，不会阻塞推理。

```python
import cv2
from ov_streaming import StreamingEngine, VideoFrame


def camera_frames(device=0, sample_fps=1.0):
    capture = cv2.VideoCapture(device)
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    index = 0
    next_tick = 0.0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                return
            t = index / source_fps
            if t + 1e-6 >= next_tick:
                yield VideoFrame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t, index, source_fps)
                next_tick += 1.0 / sample_fps
            index += 1
    finally:
        capture.release()


with StreamingEngine(MODEL, backend="vllm") as engine:
    with engine.session(
        enable=["caption"],
        max_visual_frames=96,
        compact_to_visual_frames=64,
    ) as session:
        for ev in session.stream_frames(camera_frames()):
            if ev.type == "response.output_text.done":
                print(f"[{ev.timestamp_sec:.1f}s] ({ev.control}) {ev.text}")
```

如果上游提供的是实时 H.264/MPEG-TS 字节流，可以使用 `stream_h264` 走 `codec` 模式。字节块切分位置任意，需显式提供源视频的恒定帧率与原始分辨率：

```python
def ts_chunks(stream):
    ...  # 每次 yield 一段 MPEG-TS bytes，例如 ffmpeg -i rtsp://... -c copy -f mpegts - 的输出

for ev in session.stream_h264(ts_chunks(src), fps=30.0, width=1920, height=1080):
    if ev.type == "response.output_text.done":
        print(f"[{ev.timestamp_sec:.1f}s] ({ev.control}) {ev.text}")
```

`stream_video` 的第一个参数同样支持 OpenCV / FFmpeg 可打开的网络流 URL（如 `rtsp://...`）。

### 3. 逐条手动推送

适合需要自行控制推帧与提问节奏的场景：

```python
from pathlib import Path
from PIL import Image
from ov_streaming import StreamingEngine

with StreamingEngine(MODEL, backend="transformers") as engine:
    with engine.session(enable=[]) as session:
        for sec, img_path in enumerate(sorted(Path("frames").glob("*.jpg"))):
            for ev in session.send_frame(Image.open(img_path), timestamp_sec=float(sec)):
                if ev.type == "response.output_text.done":
                    print(f"[{ev.timestamp_sec:.1f}s] {ev.text}")
        session.send_question("What just happened?", timestamp_sec=7.5)
        for ev in session.close():
            if ev.type == "response.output_text.done":
                print(f"[{ev.timestamp_sec:.1f}s] {ev.text}")
```

单条推送方法（`send_frame` / `send_codec_window` / `send_question` / `send_messages`）默认等待该条输入处理完成并返回期间产生的事件；传入 `wait=False` 可立即返回，事件会在后续调用或 `close()` 时一并返回。如需全异步并发（一个协程持续推流、另一个协程消费 `events()`），使用 `AsyncStreamingEngine` 与 `AsyncLocalSession`。

### 4. HTTP/SSE 服务模式

如需多客户端共享同一模型服务，可启动内置的 HTTP/SSE 服务：

```bash
ov-streaming serve lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct \
    --renderer ov25-window --max-model-len 32768 \
    --mm-processor-kwargs '{"max_pixels":258048}'
```

---

## 常用配置说明

### 提问时间戳（`questions=[(t, text), ...]`）

- 新会话开始、首个画面到来之前的问题（例如 `t=0.0` 的常驻指令）会随第一轮画面一起送入，并排在图像之前。
- 后续落在两个画面之间的问题，会在 `t` 时刻单独送入，模型只看得到 `t` 之前的画面；恰好落在某一画面时刻的问题，会与该画面同轮送入。
- 每条问题都会保留自身提问时间戳。
- 传入 `close_when_done=False` 时，本次流结束后会话保持打开，可继续送入下一段流或追问；仍在生成的回答会在后续读取事件时返回。

### 上下文滑窗与模式切换

- `max_visual_frames` / `compact_to_visual_frames` 默认不限制，长时间直播建议设置滑窗以防超出 `max_model_len`。
- 普通帧模式按帧数计数；`codec` 模式按画布数计数（每秒 `1 + p_canvases` 张）。保持与 1 fps 帧模式相同的历史时长时，将窗口按 `1 + p_canvases` 倍数放大即可（例如 `I+1P` 用 `192 / 128` 对应约 96 秒 → 64 秒）。
- 同一个 `StreamingEngine` 可在不同会话间切换 `frames` 和 `codec` 模式，只需初始化引擎时保证 `max_pixels >= CodecConfig.pixel_budget`。
- 如需开启连续旁白，在 `engine.session(...)` 中传入 `enable=["caption"]`；还可以通过 `caption_prompt` 自定义旁白提示，通过 `max_caption_tokens` 限制单次旁白长度。

---

## 两种后端对比

| 特性 | `backend="vllm"` | `backend="transformers"` |
| :--- | :--- | :--- |
| 依赖 | 官方 `vllm==0.25.1` + `ov-streaming` | PyTorch + `transformers>=5.7.0,<5.10` + `ov-streaming` |
| 执行方式 | 并发实时（回答生成期间持续接收新帧） | 串行单路（回答生成期间暂停接收新帧） |
| 输入支持 | `frames` / `codec` / 外部流（`codec` 在 Py3.12 下需指定 Py3.11 解码环境） | `frames` / `codec` / 外部流（Py3.11 单环境可用） |
| 适用场景 | 实时流式服务、高并发、长视频评测 | 快速试跑、无 vLLM 环境、轻量调试 |
