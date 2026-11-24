"""Stream H.264 with codec processing on the client or server."""

import argparse
import asyncio
import json
import math
from urllib.parse import urlsplit


async def _run_client(args):
    # Keep optional decode dependencies out of upload-only devices.
    from ov_streaming import OVClient
    from ov_streaming.codec import CodecConfig, send_codec_video

    client = OVClient(args.url, timeout=600)
    async with client.session(
        enable=["caption"] if args.caption else None,
        max_visual_frames=96,
        compact_to_visual_frames=64,
    ) as session:

        async def events():
            async for event in session.events():
                print(json.dumps(event.to_dict(), ensure_ascii=False), flush=True)

        consumer = asyncio.create_task(events())
        try:
            stats = await send_codec_video(
                session,
                args.video,
                config=CodecConfig(
                    strategy=args.strategy,
                    max_seconds=args.max_seconds,
                    decode_threads=args.decode_threads or 1,
                    paced=args.paced,
                ),
            )
            print(json.dumps({"codec": stats}), flush=True)
            await session.close()
            await consumer
        finally:
            if not consumer.done():
                consumer.cancel()
                await asyncio.gather(consumer, return_exceptions=True)


async def main(args):
    if args.codec_location == "client":
        await _run_client(args)
    else:
        # The sibling uploader imports only the stdlib and websockets. No
        # OVStreaming/Pillow/NumPy/codec imports run in this branch.
        from upload_h264 import main as upload

        await upload(
            argparse.Namespace(
                url=args.url,
                fps=args.fps,
                width=args.width,
                height=args.height,
                strategy=args.strategy,
                caption=args.caption,
                replay=None if args.video == "-" else args.video,
                timeout=3600,
            )
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video",
        nargs="?",
        help="H.264 file; omit or use - for server-mode camera stdin",
    )
    parser.add_argument(
        "--codec-location",
        choices=["client", "server"],
        default="client",
        help="where decoding and I+P selection run (default: client)",
    )
    parser.add_argument(
        "--url",
        help="model HTTP URL for client mode; gateway WebSocket URL for server mode",
    )
    parser.add_argument(
        "--fps", type=float, help="camera CFR rate, required in server mode"
    )
    parser.add_argument("--width", type=int, help="camera native width, required in server mode")
    parser.add_argument("--height", type=int, help="camera native height, required in server mode")
    parser.add_argument("--strategy", choices=["rdo", "pwindow"], default="rdo")
    parser.add_argument("--max-seconds", type=float, default=0, help="client mode only")
    parser.add_argument(
        "--decode-threads",
        type=int,
        choices=range(1, 9),
        help="client mode only (default: 1)",
    )
    parser.add_argument(
        "--paced",
        action="store_true",
        help="pace client decoding; server file replay is always paced",
    )
    parser.add_argument("--caption", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_seconds) or args.max_seconds < 0:
        parser.error("--max-seconds must be finite and non-negative")
    if args.codec_location == "client":
        if not args.video or args.video == "-":
            parser.error("client codec processing requires a video file")
        if args.fps is not None or args.width is not None or args.height is not None:
            parser.error("--fps/--width/--height are for server mode; client mode probes the source")
        args.url = args.url or "http://127.0.0.1:8011"
        schemes = {"http", "https"}
    else:
        if args.fps is None or not math.isfinite(args.fps) or not 0 < args.fps <= 240:
            parser.error("server mode requires --fps in (0, 240]")
        if not args.width or not args.height or args.width <= 0 or args.height <= 0:
            parser.error("server mode requires the camera's native --width and --height")
        if args.max_seconds or args.decode_threads is not None:
            parser.error(
                "--max-seconds and --decode-threads are client decode options; server mode uses gateway settings and streams until EOF/end"
            )
        args.url = args.url or "ws://127.0.0.1:8012/v1/codec/stream"
        schemes = {"ws", "wss"}
    if urlsplit(args.url).scheme not in schemes or not urlsplit(args.url).netloc:
        parser.error(
            "--url must address the model HTTP service in client mode or the gateway WebSocket in server mode"
        )
    return args


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
