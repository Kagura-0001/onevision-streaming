"""Private isolated decoder process for codec I+P windows.

It reads H.264 (MPEG-TS for live input) from stdin with ``--fps`` and the
declared source size, or decodes ``--video PATH_OR_URL`` and probes it. Both
modes print the same JSON lines; offline evaluation caches them per video, the
codec gateway forwards them live, and the in-process engine consumes them
through ``ov_streaming.codec_process``.
"""

from __future__ import annotations

import argparse
import json

from .client import _encode_messages
from .codec import CODEC_WORKER_VERSION, CodecConfig, stream_codec_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=float, help="declared CFR rate of stdin input")
    parser.add_argument("--video", help="decode this file or URL instead of stdin; fps is probed")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="stop after this source time (0: to EOF)")
    parser.add_argument("--strategy", choices=["rdo", "pwindow"], default="rdo")
    parser.add_argument("--pixel-budget", type=int, default=258048,
                        help="R: per-canvas pixel budget; infer source aspect on the 48-pixel lattice")
    parser.add_argument("--p-canvases", type=int, choices=(1, 2), default=1,
                        help="P canvases per one-second window, in addition to one I canvas")
    parser.add_argument("--source-width", type=int, help="native source width for stdin")
    parser.add_argument("--source-height", type=int, help="native source height for stdin")
    parser.add_argument("--canvas-width", type=int, help="fixed canvas width (default: from pixel budget)")
    parser.add_argument("--canvas-height", type=int, help="fixed canvas height (default: from pixel budget)")
    parser.add_argument("--candidate-fps", type=float, default=24.0)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--decode-threads", type=int, default=1)
    parser.add_argument("--paced", action="store_true", help="decode a file at its real-time rate")
    args = parser.parse_args()
    if (args.video is None) == (args.fps is None):
        parser.error("give exactly one of --fps (stdin) or --video")
    if args.video is None and (args.source_width is None or args.source_height is None):
        parser.error("stdin requires --source-width and --source-height")

    config = CodecConfig(
        strategy=args.strategy,
        width=args.canvas_width,
        height=args.canvas_height,
        pixel_budget=args.pixel_budget,
        p_canvases=args.p_canvases,
        candidate_fps=args.candidate_fps,
        max_candidates=args.max_candidates,
        decode_threads=args.decode_threads,
        max_seconds=args.max_seconds,
        paced=args.paced,
    )

    def consume(window):
        messages = _encode_messages([{"role": "user", "content": window.content()}])
        print(
            json.dumps(
                dict(
                    type="window",
                    timestamp_sec=window.timestamp_sec,
                    messages=messages,
                    worker_version=CODEC_WORKER_VERSION,
                    p_canvases=args.p_canvases,
                )
            ),
            flush=True,
        )

    stats = stream_codec_video(
        args.video or "pipe:0",
        consume,
        config=config,
        source_fps=None if args.video else round(args.fps, 6),
        source_size=None if args.video else (args.source_width, args.source_height),
    )
    print(json.dumps(dict(type="done", worker_version=CODEC_WORKER_VERSION, stats=stats)), flush=True)


if __name__ == "__main__":
    main()
