# Codec inference

Codec processing can run on a capable client or in the server-side gateway.
Both paths decode the H.264 video once and send two canvases per second: a full anchor (I), followed by a packed canvas (P).
`CodecConfig()` uses seconds-local RDO (`absolute_gain`) by default;
`CodecConfig(strategy="pwindow")` uses PWindow on the same candidate frames.
These I/P names describe model inputs, not a re-encoding or a new video GOP.

Defaults: 384×672 RGB, patch16, contiguous 3×3 source blocks, up to 24
candidates per second, one decoder thread. Candidates use the production
policy: 24 inclusive target times per second, selecting the latest real
predecessor at each target and deduplicating frame IDs. PWindow caps selected
block groups at 45, as in production. The first pair closes at 1 second;
there is no duplicated initialization input at zero. The decoder flushes the
last partial second at EOF. Selection uses only the current second and its
predecessor anchor. A later frame can close a window but cannot enter its P.
The current adapter requires a CFR H.264 input; unsupported codecs and
frame/fps timeline drift fail explicitly.

RDO calls the wheel's existing bitcost scorer and `absolute_gain` packer with
a fixed one-P budget. Its selected blocks are materialized in source-time
order. This is a codec score objective, without a learned semantic distortion
model or cross-second budget allocation. PWindow uses the wheel's existing
planner/materializer. No FFmpeg, native decoder, or codec library source is
vendored in OVStreaming.

## Choose where codec processing runs

Use the same entry point with `--codec-location client|server`. Location is
chosen when a run starts; changing it starts a new session. The existing
script defaults to **client** for compatibility with earlier commands.
RDO remains the default strategy in both modes; add `--strategy pwindow`
to use PWindow.

| Mode | Uploading machine | Codec processing | Endpoint |
|---|---|---|---|
| `client` | Capable client with codec wheel and `.[codec]` | Local H.264 file -> I+P/JPEG/positions | Model HTTP service, default `http://127.0.0.1:8011` |
| `server` | `websockets` only; file replay or camera MPEG-TS stdin | Server gateway with wheel and `.[codec-server]` | Gateway WebSocket, default `ws://127.0.0.1:8012/v1/codec/stream` |

```bash
# Capable client: decode and select locally, send I+P directly to the model.
python scripts/stream_codec_video.py input.mp4 --codec-location client \
  --caption --paced --url http://127.0.0.1:8011

# Thin client: upload H.264; the gateway decodes and selects.
python scripts/stream_codec_video.py input.mp4 --codec-location server \
  --fps 59.94006 --width 1920 --height 1080 --caption \
  --url ws://127.0.0.1:8012/v1/codec/stream

# Camera already produces H.264/MPEG-TS; the uploader does no video processing.
camera-command | python scripts/stream_codec_video.py - \
  --codec-location server --fps 30 --width 1920 --height 1080 --caption
```

Client mode probes the source FPS with `ffprobe`; server mode requires the
camera's declared CFR `--fps` and native `--width`/`--height` (a live pipe
cannot be probed without consuming it). Server mode loads only the lightweight uploader,
without importing OVStreaming, Pillow, NumPy or codec modules. File replay in
server mode uses FFmpeg stream copy and is always paced; `--paced` controls
client decoding. `--max-seconds` and `--decode-threads` are client-only and
are rejected in server mode rather than ignored. Server input runs to
EOF/`end`; the gateway owns decoder settings. The standalone
`scripts/upload_h264.py` entry point remains available for upload-only devices.

## Wheel installation

Install the wheel where decoding runs: on the capable client in client mode,
or in the server gateway in server mode. Keep that environment separate from
the vLLM environment when their Python or
NumPy requirements differ. Use the validated **TestPyPI 0.3.0** wheel for
Linux x86_64 / CPython 3.11 (glibc 2.17 or newer):

- Release: [codec-video-prep 0.3.0 on TestPyPI](https://test.pypi.org/project/codec-video-prep/0.3.0/)
- File: `codec_video_prep-0.3.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`
- SHA-256: `b76c2b96f91bead7304f609c7e169368cb4f6b4bb22e951ef1fee448ae700931`

Run these commands from the OVStreaming checkout. The URL below pins both the
wheel and its checksum; pip verifies the hash before installing. This package
is installed without dependencies first, then the chosen extra resolves its declared
dependencies from the normal PyPI index. No TestPyPI extra index is needed.

```bash
python3.11 -m venv .venv-codec
source .venv-codec/bin/activate
python -m pip install --only-binary=:all: --no-deps \
  'https://test-files.pythonhosted.org/packages/21/f1/307dd90673d523d55a419174c5794d407700f0632226b372ac6e3e831e0a/codec_video_prep-0.3.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl#sha256=b76c2b96f91bead7304f609c7e169368cb4f6b4bb22e951ef1fee448ae700931'
python -m pip install --only-binary=:all: --index-url https://pypi.org/simple/ -e '.[codec-server]'
python -m pip check
python -c 'from codec_video_prep import cv_reader_fast; assert hasattr(cv_reader_fast, "read_video_stream")'
```

The commands above install the server gateway. For client-side decoding,
replace `-e '.[codec-server]'` with `-e '.[codec]'` in the final install command.
Use the same pinned wheel URL and checksum; the vLLM process never needs it.

Alternatively, replace the direct-URL command with the following TestPyPI
version lookup; the remaining commands stay the same. The direct URL above
is preferred when reproducing this validated artifact.

```bash
python -m pip install --only-binary=:all: --no-deps \
  --index-url https://test.pypi.org/simple/ codec-video-prep==0.3.0
```

`ffprobe` must be on PATH; for example, Ubuntu/Debian provides it in the
`ffmpeg` system package. The wheel bundles the patched FFmpeg shared libraries;
do not override them with an unrelated `LD_LIBRARY_PATH`. Do not rename a
cp311 wheel for another interpreter. The vLLM process does not install this
wheel; the selected codec processor does.
A plain `pip install codec-video-prep` from the normal index does not select
this TestPyPI release.

## Real-time glasses ingress

```text
Glasses: camera H.264 + MPEG-TS mux -> WebSocket binary upload
Server: codec gateway -> isolated decoder + RDO/PWindow -> HTTP -> vLLM
Glasses: <- model events and completed answers on the same WebSocket
```

The glasses do not decode video, select patches, create JPEGs or install the
codec wheel. The gateway and its worker run on the server in the Linux Python
3.11 environment above; the model process keeps its existing vLLM environment.
The lightweight uploader needs only `pip install websockets` (or a device-native
WebSocket implementation). The gateway extra includes FastAPI and Uvicorn with
WebSocket support; it does not install or replace vLLM.

Start the existing model server as below, then in the gateway environment:

```bash
ov-codec-gateway --upstream http://127.0.0.1:8011 --host 127.0.0.1 --port 8012
```

Expose this through the deployment's authenticated TLS/WebSocket proxy when
connecting a device. The gateway itself defaults to loopback and does not
provide authentication. Each connection owns one model session and one decoder
process; the default capacity is two connections (`--max-streams`).

Connect to `/v1/codec/stream` and send:

1. JSON text: `{"type":"start","format":"mpegts","fps":59.94006,"width":1920,"height":1080,"strategy":"rdo","caption":true}`.
   `fps` is the camera's CFR source rate, not upload rate; `width`/`height`
   are its native size (a live pipe cannot be probed without consuming it). Strategy defaults to
   RDO; `pwindow` is also supported. Wait for `ready`.
2. Binary messages containing continuous **H.264 in MPEG-TS**, 1..65536 bytes
   per message. Message boundaries may split TS packets and H.264 NAL units.
   Wait for each `upload.ack` before sending another chunk. It acknowledges
   accepted bytes, not completed inference.
3. Optional JSON `{"type":"question","text":"What is happening?"}` during
   upload. It enters the current model session, followed by `question.accepted`.
4. JSON `{"type":"end"}` closes the video input. Keep receiving until
   `stream.closed`; the server drains the decoder, flushes the last partial
   second and waits for pending model answers before closing the session.

The server returns regular OV events (`input.done`, `response.output_text.delta`,
`response.output_text.done`, etc.) on this same socket. `codec.input` reports
an accepted I+P pair, its source-relative `timestamp_sec`, and SHA-256 digests
of the patch positions/JPEGs. `codec.done` reports decoder statistics. Errors
terminate the connection and abort its owned model session and decoder; a
reconnect starts a new timeline/session, without joining unrelated streams.

MPEG-TS supplies PTS/DTS; transport muxing requires no video decoding or
re-encoding. Start with PAT/PMT, SPS/PPS and a decodable keyframe. Bare Annex-B
without timestamps, RTP/WebRTC and VFR are not supported by this first ingress.
The worker normalizes the first decoded PTS to zero and uses subsequent source
PTS, never network arrival time. Source frame/fps drift, nonmonotone PTS or a
source gap over one second fails the session. Do not reset timestamps or change
camera FPS mid-connection. A disconnected upload is aborted; graceful `end` is
required for the final partial-second flush.

Upload backpressure propagates through bounded WebSocket queues and OS pipes;
frames are not silently dropped or buffered as a whole recording. A blocked
native probe runs in a killable subprocess. An idle upload times out after
30 seconds; blocked processing/output times out after 120 seconds. The wheel's
initial stream probing and codec frame reordering can add startup latency;
this interface does not promise subsecond startup. Selection still uses only
source predecessors within each one-second window.

For a paced NBA/file replay from a device simulator, run **only the uploader**
on the uploading machine. FFmpeg here stream-copies H.264 and adds MPEG-TS;
it does not decode or select patches:

```bash
python scripts/upload_h264.py --replay input.mp4 --fps 59.94006 \
  --width 1920 --height 1080 --caption --url ws://127.0.0.1:8012/v1/codec/stream
python scripts/upload_h264.py --replay input.mp4 --fps 59.94006 \
  --width 1920 --height 1080 --caption --strategy pwindow --url ws://127.0.0.1:8012/v1/codec/stream
# A real camera already providing H.264/MPEG-TS writes to stdin instead:
# camera-command | python scripts/upload_h264.py --fps 30 --width 1920 --height 1080 --caption
```

## Serve and run

The server needs the existing OV2.1 vLLM fork and a patch16/merge3 checkpoint.
It does not need the codec wheel. Allow the exact 258048-pixel canvas:

```bash
ov-streaming serve /path/to/ov21-checkpoint --renderer ov21 \
  --max-model-len 32768 --mm-processor-kwargs '{"max_pixels":258048}'
```

For client-side decoding, run from the capable client's codec environment
(the same commands can also be used for offline server diagnostics):

```bash
python scripts/stream_codec_video.py input.mp4 --codec-location client --caption --max-seconds 10
python scripts/stream_codec_video.py input.mp4 --codec-location client --caption --strategy pwindow --max-seconds 10
```

Use `--paced` to replay packets by DTS. Without it, the file decodes as fast
as possible. No camera capture or raw-frame encoding is provided. Without
`--caption`, the model samples the ordinary question/response controls.
Questions can be sent with the regular client while `send_codec_video` runs.
Keep one task reading `session.events()` or `session.responses()`.

Each pair is one input and occupies two visual-frame slots; with the example's
96/64 frame limits, those correspond to approximately 48/32 seconds. Existing
session compaction and memory handling apply unchanged.

## Input contract

Each image part carries `codec={layout: "row_major", patch_size: 16,
patch_positions: [[source_frame, source_y, source_x], ...]}` plus real source
`fps` and the availability cutoff `timestamp_sec`. Positions refer to the
source raster before packing. The server preserves the canvas dimensions,
and the selected codec processor encodes each canvas once as JPEG quality 95 /
4:4:4, matching production training (optimize and progressive are both false).
The client or gateway sends those bytes unchanged. The server converts positions to merge-block
order and emits exactly one vision wrapper per canvas. Its timestamp is
`<t seconds>` when every patch has the same source frame, or
`<min_t to max_t seconds>` otherwise, with source frame indices divided by
real fps and rounded to one decimal. Distinct frames remain a range even
when the rounded endpoints coincide. Patch temporal coordinates are never
rounded or replaced by canvas timestamps. Token count and positions remain
aligned. This follows `canvas_patch_span_point_range_v1` in the frozen
training consumer; it does not implement the older per-source-frame-run
serializer. Incompatible checkpoint geometry, invalid coordinates,
future source frames, and mismatched merge blocks are rejected.

Ordinary images still use `render_frame`; UniVIA rejects codec inputs.

No vLLM processor override is needed: each complete canvas is one normal
image item, preserving the training vision attention window.

The raw-video candidate selection and the model input contract are separate:
the same candidates produce matching selection; the same decoded JPEG
canvases and positions produce matching consumer inputs. A checkpoint
trained with another timestamp schema needs that schema explicitly; this
first integration targets the frozen point/range consumer.

For file decoding, the CFR source clock uses ffprobe `r_frame_rate` rounded to six decimals,
matching production. It does not use container-average fps or renumber
selected candidates as a new 24fps timeline.
