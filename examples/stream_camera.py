"""Stream live frames from a local camera or OpenCV capture source."""

import argparse
from collections.abc import Iterator

import cv2

from ov_streaming import StreamingEngine, VideoFrame


def camera_frames(source: int | str, sample_fps: float) -> Iterator[VideoFrame]:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
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
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                yield VideoFrame(rgb, t, index, source_fps)
                next_tick += 1.0 / sample_fps
            index += 1
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index (e.g. 0) or stream URL")
    parser.add_argument(
        "--model",
        default="lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct",
    )
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--question", default=None)
    args = parser.parse_args()

    source: int | str = int(args.source) if args.source.isdigit() else args.source
    questions = [(0.0, args.question)] if args.question else None

    with StreamingEngine(args.model, backend=args.backend) as engine:
        with engine.session(
            enable=[] if questions else ["caption"],
            max_visual_frames=96,
            compact_to_visual_frames=64,
        ) as session:
            for ev in session.stream_frames(camera_frames(source, args.fps), questions=questions):
                if ev.type == "response.output_text.done":
                    print(f"[{ev.timestamp_sec:.1f}s] ({ev.control}) {ev.text}")


if __name__ == "__main__":
    main()
