"""Stream a video file or RTSP URL with LLaVA-OneVision-2.5."""

import argparse

from ov_streaming import CodecConfig, StreamingEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to a local video file or RTSP URL")
    parser.add_argument(
        "--model",
        default="lmms-lab-encoder/LLaVA-OneVision-2.5-4B-Instruct",
    )
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--mode", choices=("frames", "codec"), default="frames")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--p-canvases", type=int, default=1)
    parser.add_argument("--max-pixels", type=int, default=258048)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-visual-frames", type=int, default=96)
    parser.add_argument("--compact-to", type=int, default=64)
    parser.add_argument("--max-response-tokens", type=int, default=128)
    parser.add_argument("--question", default=None)
    parser.add_argument("--caption", action="store_true")
    parser.add_argument("--codec-python", default=None)
    args = parser.parse_args()

    scale = (1 + args.p_canvases) if args.mode == "codec" else 1
    max_frames = args.max_visual_frames * scale if args.max_visual_frames > 0 else None
    compact_to = args.compact_to * scale if max_frames is not None else None

    codec_config = (
        CodecConfig(strategy="rdo", pixel_budget=args.max_pixels, p_canvases=args.p_canvases)
        if args.mode == "codec"
        else None
    )
    questions = [(0.0, args.question)] if args.question else None

    with StreamingEngine(
        args.model,
        backend=args.backend,
        max_model_len=args.max_model_len,
        max_pixels=args.max_pixels,
    ) as engine:
        with engine.session(
            enable=["caption"] if args.caption else [],
            max_visual_frames=max_frames,
            compact_to_visual_frames=compact_to,
            max_response_tokens=args.max_response_tokens,
        ) as session:
            for ev in session.stream_video(
                args.video,
                video_backend=args.mode,
                fps=args.fps,
                codec_config=codec_config,
                codec_python=args.codec_python,
                questions=questions,
            ):
                if ev.type == "response.output_text.done":
                    print(f"[{ev.timestamp_sec:.1f}s] ({ev.control}) {ev.text}")


if __name__ == "__main__":
    main()
