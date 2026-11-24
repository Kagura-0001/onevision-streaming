"""Drive an `ov-streaming serve` instance asynchronously with OVClient."""

import argparse
import asyncio
from pathlib import Path

from ov_streaming import OVClient


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frames_dir", type=Path, help="Directory of ordered .jpg frames")
    parser.add_argument("--url", default="http://127.0.0.1:8011")
    parser.add_argument("--question", default="Tell me when something notable happens.")
    parser.add_argument("--question-sec", type=int, default=0)
    args = parser.parse_args()

    client = OVClient(args.url)
    async with client.session(max_visual_frames=96, compact_to_visual_frames=64) as session:

        async def read_answers() -> None:
            async for ev in session.responses():
                print(f"[{ev.timestamp_sec:.1f}s] ({ev.control}) {ev.text}")

        reader = asyncio.create_task(read_answers())
        for sec, path in enumerate(sorted(args.frames_dir.glob("*.jpg"))):
            content: list[dict] = [
                {"type": "image_url", "image_url": {"url": path}, "timestamp_sec": float(sec)}
            ]
            if sec == args.question_sec:
                content.append({"type": "text", "text": args.question})
            await session.send_messages(messages=[{"role": "user", "content": content}])
        await session.close()
        await reader


if __name__ == "__main__":
    asyncio.run(main())
