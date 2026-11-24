"""Upload H.264/MPEG-TS bytes and receive results; no codec wheel or decoding.

Read a camera's MPEG-TS stream from stdin, or use --replay VIDEO to simulate
capture with FFmpeg (-re -c:v copy: stream copy only, no video decoding).
"""

import argparse
import asyncio
import json
import sys

from websockets.asyncio.client import connect


async def main(args):
    process = None
    stdin_transport = None
    if args.replay:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-v",
            "error",
            "-re",
            "-i",
            args.replay,
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            "-bsf:v",
            "h264_mp4toannexb",
            "-mpegts_flags",
            "+resend_headers",
            "-muxdelay",
            "0",
            "-f",
            "mpegts",
            "pipe:1",
            stdout=asyncio.subprocess.PIPE,
        )
    try:
        if process:
            source = process.stdout
        else:
            # A blocking read in to_thread cannot be cancelled when an idle
            # camera stops sending. Use an async pipe so disconnect exits.
            source = asyncio.StreamReader()
            stdin_transport, _ = await asyncio.get_running_loop().connect_read_pipe(
                lambda: asyncio.StreamReaderProtocol(source), sys.stdin.buffer
            )
        async with connect(args.url, max_size=1024 * 1024, max_queue=4) as ws:
            await ws.send(
                json.dumps(
                    dict(
                        type="start",
                        format="mpegts",
                        fps=args.fps,
                        width=args.width,
                        height=args.height,
                        strategy=args.strategy,
                        caption=args.caption,
                    )
                )
            )
            ready = json.loads(await ws.recv())
            print(json.dumps(ready), flush=True)
            if ready.get("type") != "ready":
                raise RuntimeError("gateway rejected stream")
            ack = asyncio.Queue(maxsize=1)

            async def receive():
                async for raw in ws:
                    row = json.loads(raw)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    if row["type"] == "error":
                        raise RuntimeError(row["error"])
                    if row["type"] == "upload.ack":
                        await ack.put(row["bytes_received"])
                    if row["type"] == "stream.closed":
                        return
                raise RuntimeError("connection ended before stream.closed")

            async def upload():
                while True:
                    data = await source.read(32768)
                    if not data:
                        break
                    await ws.send(data)
                    await asyncio.wait_for(ack.get(), 120)
                if process and await process.wait():
                    raise RuntimeError("FFmpeg replay failed")
                await ws.send('{"type":"end"}')

            tasks = [asyncio.create_task(receive()), asyncio.create_task(upload())]
            try:
                await asyncio.wait_for(asyncio.gather(*tasks), args.timeout)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if stdin_transport:
            stdin_transport.close()
        if process and process.returncode is None:
            process.terminate()
            await process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8012/v1/codec/stream")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--width", type=int, required=True, help="camera native width")
    parser.add_argument("--height", type=int, required=True, help="camera native height")
    parser.add_argument("--strategy", choices=["rdo", "pwindow"], default="rdo")
    parser.add_argument("--caption", action="store_true")
    parser.add_argument("--replay")
    parser.add_argument("--timeout", type=float, default=3600)
    asyncio.run(main(parser.parse_args()))
