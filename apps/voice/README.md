# OV Voice

`ov_voice` lets a user talk to an OV session from a browser and hear its
answers. [Pipecat](https://github.com/pipecat-ai/pipecat) carries the audio and
video: its SmallWebRTC transport and prebuilt page, Silero voice activity
detection, Smart Turn end-of-turn detection, interruptions and playback.
Speech recognition and speech synthesis run on OpenAI-compatible servers, and
OV runs on `ov-streaming serve`.

```text
browser mic, camera
  -> SmallWebRTC input -> Silero VAD -> TranscriptionService -> UserTurnProcessor -> OVProcessor -> SpeechService -> SmallWebRTC output
                                               |                (Smart Turn v3)          |               |
                                  vllm serve Qwen3-ASR                          ov-streaming serve   vLLM-Omni Qwen3-TTS
```

This is an application on top of OVStreaming, kept apart from the
`ov_streaming` package: it has its own `pyproject.toml`, needs Python 3.11 and
Pipecat, and reaches OV only through `ov_streaming.client` over the public HTTP
API. So `OVProcessor`, in `src/ov_voice/pipecat.py`, does not know where a
question goes in the prompt, how memories are written or which tokens the
model samples. Installed next to `ov-streaming`, the app could still import the
server, so `tests/test_pipecat.py` checks that importing it leaves
`ov_streaming.session`, `ov_streaming.backend`, `ov_streaming.renderers` and
`ov_streaming.serve` unloaded. Which answers are spoken and how much of each is
remembered is decided in `src/ov_voice/answers.py`, which knows nothing about
Pipecat.

## Deployment

Everything runs on the server; the user's machine only runs a browser, which
sends the microphone and camera and plays the answers. Voice is four processes
in three environments:

| Process | Environment | Port | GPU |
|---|---|---|---|
| `ov-streaming serve` | the vLLM fork | 8011 | yes |
| `vllm serve Qwen/Qwen3-ASR-1.7B` | the vLLM fork | 8002 | yes |
| `vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni` | vLLM-Omni | 8091 | yes |
| `python -m ov_voice` | Python 3.11 or newer with this package | 7860 | no |

Qwen3-ASR is an upstream vLLM model (`Qwen3ASRForConditionalGeneration`), so
the fork serves it on the plain `/v1/audio/transcriptions` route, which does
not touch the streaming input or branch code the fork adds. Install the fork's
audio dependencies and start both servers:

```bash
pip install av scipy soundfile soxr "mistral_common[audio]"
ov-streaming serve /path/to/univia --renderer univia --port 8011
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-ASR-1.7B --port 8002 --gpu-memory-utilization 0.08
```

Speech comes from Qwen3-TTS CustomVoice on
[vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/stable/serving/speech_api/),
which runs on its own vLLM release and so needs an environment apart from the
fork:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --omni --port 8091 --trust-remote-code --enforce-eager
```

Any server with an OpenAI-compatible `/v1/audio/speech` that streams 24 kHz
16-bit mono PCM will do instead; pass its URL, model and voice to the bot. The
CustomVoice checkpoint has the voices aiden, dylan, eric, ono_anna, ryan,
serena, sohee, uncle_fu and vivian, and `GET /v1/audio/voices` lists what a
running server offers.

The bot needs no GPU and no vLLM. Install it with the OVStreaming client, in a
Python 3.11 or newer environment of its own, from the repository root:

```bash
pip install -e . -e apps/voice
python -m ov_voice -t webrtc \
    --ov-url http://127.0.0.1:8011 \
    --asr-url http://127.0.0.1:8002/v1 --asr-model Qwen/Qwen3-ASR-1.7B \
    --tts-url http://127.0.0.1:8091/v1 --tts-model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --tts-voice vivian \
    --max-visual-frames 120 --compact-to-visual-frames 60
```

The URLs and models shown are the defaults. `--frame-interval` (1 s by
default) is the shortest time between two frames sent to OV, and
`--max-visual-frames` and `--compact-to-visual-frames` are the session options
described in the [OVStreaming README](../../README.md). `--turn-detection` is
described under [Turn detection](#turn-detection). The other flags belong to
Pipecat's development runner: `--host` and `--port` (localhost:7860 by
default), `-t webrtc` to serve only the WebRTC transport, and `--ice-servers`.

Open http://localhost:7860/live/ for this app's page. It shows the camera,
what OV is doing now (waiting, listening, thinking, speaking), and one
conversation: each of the user's turns walks through 说话 → 识别 → 发给 OV →
OV 决定 → OV 说 with OV's answer under it, and OV's unprompted answers sit
in between. OV always captions the camera (`--no-captions` turns that off);
the 画面解说 switch shows the captions written after it is turned on, and
captions are never spoken. Questions can also be typed. The page warns
when the browser hands it a silent microphone (on macOS, a browser without the
system's microphone permission) and when the microphone hears speech the
server never reports. A developer panel, off by default, shows OV's latency,
its decision on each frame, the connection (path, RTT, bitrates, loss) and the
RTVI event log. The caption switch opens the session with captioning on, and
push-to-talk keeps the microphone silent except while the talk button or the
space bar is held. The page asks the camera for 640x360 at 5 fps and caps it at
250 kbps, since OV takes one frame a second; `?vbr=<kbps>&vfps=<n>` override.

The page is a Vite + React + TypeScript app in `web/`, built with the
[Pipecat client](https://github.com/pipecat-ai/pipecat-client-web) and
[voice UI kit](https://voiceuikit.pipecat.ai/). Its build goes into
`src/ov_voice/web/`, which is committed, so running the bot needs no Node:

```bash
cd apps/voice/web
npm ci
npm test            # the session model in src/ov/session.ts
npm run build       # writes ../src/ov_voice/web
npm run dev         # a dev server that proxies /start, /sessions and /model to a bot on :7860
```

`src/ov/session.ts` turns the bot's messages into the page's state: RTVI events
for speech, transcripts and OV speaking, and OVProcessor's questions and OV
events, which it sends as RTVI server messages (`{"ov": "question"}` and
`{"ov": "event", ...}`). Typed questions go to the bot as the RTVI client
message `ov-text`. Pipecat's prebuilt page at /client/ still works too.

Each browser that connects gets its own pipeline and its own OV session, and
the session is aborted when the page disconnects. If OV refuses the session,
as it does with 503 when the engine has no free request slots, or the session
fails later, the bot reports the error and cancels the pipeline, which
disconnects the page.

## Connecting a browser

The page and the bot talk over two paths. The page, and the WebRTC offer and
answer, are plain HTTP on the bot's port. The audio and video are WebRTC
media, sent over UDP between the browser and the bot. Each needs its own way
through.

**The page.** Browsers give a page the microphone and the camera only over
HTTPS or on localhost. Either forward the port and open
http://localhost:7860:

```bash
ssh -L 7860:localhost:7860 gpu-host
```

or start the bot with `--host 0.0.0.0` behind an HTTPS reverse proxy (Caddy,
nginx) and open its https:// address.

**The media.** When the browser can reach the server over UDP, on the same
network or VPN, nothing more is needed. Otherwise, which is the usual case for
a GPU server behind a firewall or NAT, both sides need a TURN server they can
both reach, which relays the media. Give it to the bot in
`PIPECAT_ICE_SERVERS` (or `--ice-servers`); the bot uses it and hands the same
list to the page when it connects:

```bash
export PIPECAT_ICE_SERVERS='[
  {"urls": "stun:stun.l.google.com:19302"},
  {"urls": ["turn:turn.example.com:3478?transport=udp", "turns:turn.example.com:443?transport=tcp"],
   "username": "USER", "credential": "PASSWORD"}
]'
```

Any TURN server will do: a hosted service, or [coturn](https://github.com/coturn/coturn)
on a machine with a public address. A `turns:` URL on port 443 over TCP gets
through most firewalls that block UDP. If the page connects but no audio
arrives, or it stays connecting, the media found no path, and a TURN server is
what is missing. The page's connection line shows which path was taken; on a
slow path, such as TURN over TCP behind a VPN, the page caps the camera at
250 kbit/s and 5 frames a second (`?vbr=` and `?vfps=` change that) so that
the audio gets through.

## Turn detection

A question cannot be taken back once it is appended to the source, so the bot
sends it only when the user's turn is over. Two signals decide that:

- Silero VAD reports a pause after a short silence. Each pause
  ends a speech segment, and `TranscriptionService` transcribes each segment as
  one WAV file.
- With `--turn-detection smart-turn`, the default, Pipecat's `UserTurnProcessor`
  runs [Smart Turn v3](https://github.com/pipecat-ai/smart-turn), a small audio
  model on the CPU, at each pause and decides whether the turn is over. A pause
  it calls unfinished keeps the turn open; `OVProcessor` holds the segments'
  transcripts and joins them into one question when the turn ends. VAD stops
  after 0.2 s, Pipecat's default for use with Smart Turn.

`--turn-detection silence` is the behaviour before Smart Turn: no turn model,
VAD stops after 0.8 s, and every segment is a question of its own. It is kept
for comparison.

Pipecat's defaults bound how long a turn can wait. Smart Turn ends a turn it
still calls unfinished after 3 s of silence. When Smart Turn ends a turn, the
turn waits for the transcript of its last segment until 1 s after the speech
ended; a transcript that comes later is not part of that question, and goes in
with the user's next turn or as a question of its own. A finished question
then rides on the next camera frame, so it waits up to two frame intervals.

Each question sent to OV logs one line:

```text
OV question: speech_end->sent=439ms speech_end->turn_end=439ms turn_end->sent=0ms turns=1 segments=1 with_frame=False text='这锅水开了吗？'
```

`speech_end->turn_end` covers VAD, Smart Turn and ASR; `turn_end->sent` covers
the wait for a camera frame and the request to OV; `turns` and `segments` count
the turns and VAD segments joined into the question. The line gives timings,
not correctness: whether a question was cut in two shows only against what the
user actually said.

## GPU memory

Each GPU process takes a fixed share of its GPU, set by hand. The numbers below
are estimates for 80 GB GPUs, not measurements. Keep GPU 0 for OV at vLLM's
default `--gpu-memory-utilization 0.9`, and put ASR and TTS on GPU 1. Qwen3-ASR
1.7B holds about 4 GB of bf16 weights, so 0.08 of the GPU leaves room for its
cache. vLLM-Omni's default deploy config for Qwen3-TTS
(`vllm_omni/deploy/qwen3_tts.yaml`) gives each of its two stages, the talker
and the codec decoder, `gpu_memory_utilization: 0.3` of the first visible GPU,
so with `CUDA_VISIBLE_DEVICES=1` it takes 0.6 of GPU 1; the 1.7B talker needs
far less. To share a GPU, copy that file, lower both values, and pass the copy
with `--deploy-config`. On one GPU, an estimate is 0.7 for OV and 0.08 each for
ASR and the two TTS stages, started in that order so each server finds its
share free.

## A spoken question, step by step

The transport passes on up to 30 camera frames a second, and `OVProcessor`
sends the first that comes at least one frame interval after the last frame it
sent, about every 1.0 to 1.03 s. A frame's time is seconds on the pipeline
clock, which starts when the page connects, so it rarely falls on a whole
second; the times below are rounded. OV is serving UniVIA and has just sampled
`<silence>` for the frame at 13 s.

The user asks "这锅水开了吗？". Silero reports the start of speech, and since
no answer is playing there is nothing to interrupt. Silero reports the end of
speech after 0.2 s of silence, Smart Turn calls the turn finished, and
`TranscriptionService` posts the segment as one WAV file:

```text
POST http://127.0.0.1:8002/v1/audio/transcriptions
model=Qwen/Qwen3-ASR-1.7B  response_format=json  file=audio.wav
```

Recognition is not streamed. Partial text could not go to OV anyway, since
text appended to the source stays there, but streaming could still shorten the
wait for the final text; that has not been measured. The request also carries no
`language`. Pipecat's `OpenAISTTService` always sends one, English unless set,
and vLLM then ends Qwen3-ASR's prompt with `language English<asr_text>`, which
tells the model the speech is English whatever was said. Without it the model
writes `language Chinese<asr_text>这锅水开了吗？` and vLLM strips the prefix:

```json
{"text": "这锅水开了吗？"}
```

`TranscriptionService` asks for `json` because vLLM answers `verbose_json`
only for models with segment timestamps, which Qwen3-ASR does not have.

`OVProcessor` holds the transcript until the next frame, at 14 s, and sends
both as one input. The client turns the frame into a JPEG data URL:

```json
{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}, "timestamp_sec": 14.0},
  {"type": "text", "text": "这锅水开了吗？", "timestamp_sec": 14.0}
]}
```

The session reads this as one turn with a frame and a question. The UniVIA
renderer writes the question after the frame, as in training, and the append
starts with the control the source sampled last:

```text
<silence><|im_end|>
<|im_start|>user
<14.0 seconds><|vision_start|><|image_pad|><|vision_end|>
<question_memory>{"timestep": "00:00:14", "content": "这锅水开了吗？"}</question_memory><|im_end|>
<|im_start|>assistant
```

The source samples `</response>`, the session opens a branch, and the branch
writes the answer. `OVProcessor` reads the event stream and acts only on events
with `control: "response"`; captions are written back by the server and never
spoken.

```json
{"type": "response.output_text.done", "response_id": "ov-response-5f0c…-000000", "control": "response",
 "timestamp_sec": 14.0, "text": "At 14 seconds: 水已经开了。请把火关小一点。"}
```

The answer is cut into sentences with Pipecat's sentence splitter, and each
sentence goes to the TTS service as its own `TTSSpeakFrame`, so
`SpeechService` makes one request per sentence:

```json
POST http://127.0.0.1:8091/v1/audio/speech
{"model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "input": "At 14 seconds: 水已经开了。",
 "voice": "vivian", "response_format": "pcm", "stream_format": "audio"}
```

and then the same for "请把火关小一点。". The server streams 24 kHz PCM as it
decodes it. Pipecat's `OpenAITTSService` would refuse `vivian`, since it
accepts only OpenAI's voice names; `SpeechService` sends the voice as given.

The bot opens the session with `answer_commit="client"`, so the answer is not
written back when it finishes. Frames keep going in meanwhile; the append for
the frame at 15 s starts with `</response><|im_end|>` and carries no memory.

Pipecat tells the processor when a sentence has been played. The output
transport writes audio to the WebRTC track at playback speed and handles the
frames between audio chunks in order. Each TTS request ends with a
`TTSStoppedFrame` behind its audio, and when the transport reaches it, it sends
a `BotStoppedSpeakingFrame` upstream. `OVProcessor` sits upstream of the TTS
service, so it receives one `BotStoppedSpeakingFrame` per sentence, at the
moment that sentence has finished playing, and records the sentence end, here
character 21 of the answer.

During the second sentence the user says "好的". Silero reports the start of
speech, and `OVProcessor` stops the answer. It broadcasts an
`InterruptionFrame`, on which the TTS service drops the audio it has not sent
and the output transport drops the audio it has not played. It cancels every
response still generating with
`DELETE /v1/ov/sessions/{session_id}/responses/{response_id}`, which aborts
only the branch; the source request goes on unchanged. It drops the answers
waiting to be spoken by committing them as empty text, which writes nothing.
Before the next input it commits the sentences the user heard:

```text
POST /v1/ov/sessions/5f0c…/responses/ov-response-5f0c…-000000/commit
{"text": "At 14 seconds: 水已经开了。"}
```

and the session writes them at the start of the turn for the frame at 16 s.
The memory keeps the time OV chose to answer, not the time it was played:

```text
<silence><|im_end|>
<|im_start|>user
<answer_memory>{"timestep": "00:00:14", "content": "At 14 seconds: 水已经开了。"}</answer_memory><16.0 seconds><|vision_start|><|image_pad|><|vision_end|>
<|im_end|>
<|im_start|>assistant
```

"好的" then rides on the next frame as a question. Had the user let the answer
play to the end, the second `BotStoppedSpeakingFrame` would have committed the
whole answer, and the next answer waiting, if any, would have started.

With OV2.1 the same input renders in its own format (questions before frames,
millisecond timecodes, `<response>`), and the voice bot does not change: it
only reads the control name.

When no frame has come for two frame intervals, because the page has no
camera or the camera is off, a final transcript goes in alone as a text-only
turn. A transcript waiting for the next frame goes in alone the same way if no
frame comes within two intervals. UniVIA never saw such a turn in training.

## Tests

`tests/test_answers.py` covers which answers are spoken and what is
committed; it needs nothing beyond this package. `tests/test_pipecat.py`
checks the import boundary, drives `OVProcessor` with Pipecat's `run_test` and
a fake OV client (a question rides on the next frame or goes in alone when the
camera stops, a response is spoken sentence by sentence and a caption is not,
the user speaking cancels the answers still generating and commits only the
played sentences, a refused or failed session cancels the pipeline). With
`user_turns` it checks that the transcripts of a turn become one question that
no frame carries early, that an answer waits for the end of the turn and
does not start when the next turn has begun, and,
through the real `UserTurnProcessor` with a scripted turn model, that a turn
paused once still ends as one question after its last transcript. It also
checks what `TranscriptionService` sends. From `apps/voice`:

```bash
pip install -e "../..[test]" -e ".[test]"
pytest
```
