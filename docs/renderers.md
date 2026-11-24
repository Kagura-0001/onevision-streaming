# Renderers

A renderer is the prompt format of one checkpoint family. It writes a session's
turns as ChatML text, encodes frames, and names the control tokens the model
samples. It keeps no state and imports nothing from vLLM. The session, the
backend and the HTTP service never write a token or a tag, so a format change
stays in one file under `src/ov_streaming/renderers/`.

`ov-streaming serve --renderer` takes `univia`, `ov21`, or `module:Class` for a
renderer of your own. The names are looked up in `RENDERERS` in
`renderers/__init__.py`, and a module is imported only when its renderer is
picked.

## Turns

The session reads every input as one `Turn`:

```python
@dataclass(frozen=True)
class VisualFrame:
    text: str        # what the frame writes into the prompt
    data: Any        # what the backend sends to vLLM as the image

@dataclass(frozen=True)
class Memory:
    kind: str        # "question", or the control whose branch wrote it
    timestamp_sec: float
    text: str

@dataclass
class Turn:
    timestamp_sec: float
    frames: tuple[VisualFrame, ...] = ()
    questions: tuple[Memory, ...] = ()
    memories: tuple[Memory, ...] = ()   # written back with this turn, in completion order
    control: str | None = None          # what the source sampled, by name
```

Each image part of the input goes through `render_frame`, and each text part
becomes a question. Answers that finished since the previous input are attached
as the turn's memories, in the order they finished. An answer's time is the
time of the turn whose control opened its branch, so it is dated when the model
chose to speak, not when it was written back.

## The source and its branches

The first input opens the source request with `render_prompt([turn])`. Every
later input is one more streaming input on that request, `render_append(turn,
previous)`, so nothing before it is prefilled again. The source samples one
token per input, restricted to the renderer's control tokens.

When the next input arrives, vLLM drops the token the source sampled: it was
never fed back, so it has no KV yet. `render_append` therefore starts with that
token:

```text
{controls[previous]}<|im_end|>
<|im_start|>user
{render_user(turn)}<|im_end|>
<|im_start|>assistant
```

With that, the source always holds exactly the rendered conversation:
`render_prompt(turns[:n]) + render_append(turns[n], turns[n - 1].control)`
equals `render_prompt(turns[:n + 1])`. `tests/test_renderers.py` checks this for
both renderers, as text and as tokens.

A control other than `silence` opens a branch. The branch's prompt is the
source's computed tokens followed by that control token, so it shares the
source's KV cache up to the control and generates the answer on top of it while
the source takes the next inputs. The session awaits the branch's submission
before it sends the next input, so the engine copies the source's state before
it changes. The history the source keeps is the bare control and `<|im_end|>`;
the answer comes back only as a memory.

## UniVIA

`UniVIARenderer` writes what the ms-swift template `qwen3_vl_ov_source_policy`
wrote in training: no system prompt, a frame as its timestamp with one decimal
followed by one image placeholder, questions after the frame, and the controls
`<silence>` and `</response>`. The times below come from the training episode
in `tests/fixtures/univia_cooking.jsonl`, and the first three blocks appear in
its training text character for character. A system message sent with the
first input is still written; training had none.

The first input opens the source:

```text
<|im_start|>user
<1293.0 seconds><|vision_start|><|image_pad|><|vision_end|>
<|im_end|>
<|im_start|>assistant
```

The source samples `<silence>`. The next input carries a frame and a question,
and its append starts with that `<silence>`:

```text
<silence><|im_end|>
<|im_start|>user
<1294.0 seconds><|vision_start|><|image_pad|><|vision_end|>
<question_memory>{"timestep": "00:21:34", "content": "请继续观察该男子接下来的烹饪准备工作，等这段画面结束后，告诉我他主要在做什么。"}</question_memory><|im_end|>
<|im_start|>assistant
```

At 1368 s the source samples `</response>` after this append:

```text
<silence><|im_end|>
<|im_start|>user
<1368.0 seconds><|vision_start|><|image_pad|><|vision_end|>
<|im_end|>
<|im_start|>assistant
```

The session forks a branch with `suffix_token_ids=[id of </response>]`. Its
prompt is everything above plus `</response>`. In the training episode the
answer that follows is ` At 1368 seconds: 他主要在使用保鲜膜将面团塑形包裹。<|im_end|>`.
The answer event carries the text without surrounding whitespace.

The source does not wait for the branch. Once the answer has finished, the next
append writes it back as an `<answer_memory>` ahead of the frame, which is
where the UniVIA paper puts committed answers (Section 3.2):

```text
</response><|im_end|>
<|im_start|>user
<answer_memory>{"timestep": "00:22:48", "content": "At 1368 seconds: 他主要在使用保鲜膜将面团塑形包裹。"}</answer_memory><1369.0 seconds><|vision_start|><|image_pad|><|vision_end|>
<|im_end|>
<|im_start|>assistant
```

Training episodes end at the first answer, so none shows a written-back
answer. The block above is a default: `timestep` is the time the source chose
to answer, written as `HH:MM:SS`, and `content` is the answer as the event
reported it.

When a new input would bring the source past `max_visual_frames`, or leave no
room in the context for an answer, the session opens a new source from the
newest turns. The questions and memories of
the dropped turns go to the start of the first kept user turn, in the order the
dropped turns showed them, one block each. Here the turns before 1398 s were
dropped:

```text
<|im_start|>user
<question_memory>{"timestep": "00:21:34", "content": "请继续观察该男子接下来的烹饪准备工作，等这段画面结束后，告诉我他主要在做什么。"}</question_memory><answer_memory>{"timestep": "00:22:48", "content": "At 1368 seconds: 他主要在使用保鲜膜将面团塑形包裹。"}</answer_memory><1398.0 seconds><|vision_start|><|image_pad|><|vision_end|>
<|im_end|>
<|im_start|>assistant
<silence><|im_end|>
<|im_start|>user
<1399.0 seconds><|vision_start|><|image_pad|><|vision_end|>
<|im_end|>
<|im_start|>assistant
```

No training window starts with an archive either; the blocks go where OV2.1
puts them. A turn without a frame writes only its blocks, for example
`<question_memory>{"timestep": "00:23:20", "content": "锅里的水开了吗？"}</question_memory>`.
Training has no such turn.

`render_frame` resizes the image with `qwen_vl_utils.fetch_image` between 4096
and 131072 pixels, the function and bounds training used. The renderer passes
the same bounds to vLLM through `mm_processor_kwargs`, so vLLM's Qwen3-VL
processor keeps the size and expands the single `<|image_pad|>` to the frame's
patches. Captioning is off; `"caption": "<captioning>"` in `controls` turns it
on for a checkpoint that has that token.

## OV2.1

`OV21Renderer` keeps the format OV2.1 was served with before renderers existed.
`tests/fixtures/ov21_session.txt` holds a recorded session, and both the
renderer and the session reproduce it character for character. The prompt has
a system prompt, questions before the frame, millisecond timecodes, one block
per line, and the controls `<silence>`, `<response>` and `<captioning>`.
Captioning is a switch a session turns on; see "Captioning" below.

OV2.1 frames reach vLLM already processed: the renderer runs the checkpoint's
`Qwen2VLImageProcessor` and its `build_patch_positions`, and the text carries one
`<|image_pad|>` per merged patch. The dumps write that run as `<|image_pad|>×N`.

```text
<|im_start|>system
You are a real-time streaming video assistant.<|im_end|>
<|im_start|>user
<0.0 seconds><|vision_start|><|image_pad|>×N<|vision_end|>
<|im_end|>
<|im_start|>assistant
```

An input with a question:

```text
<silence><|im_end|>
<|im_start|>user
<question_memory>{"timestep": "00:00:01", "content": "Tell me when the door opens."}</question_memory>
<1.0 seconds><|vision_start|><|image_pad|>×N<|vision_end|>
<|im_end|>
<|im_start|>assistant
```

The source samples `<response>`, and the branch's prompt is the text above
followed by `<response>`. On the next input, at 2 s, it samples `<captioning>`,
which opens a second branch the same way. Both finish before the input at 3 s,
the caption first. OV2.1 writes captions and answers back as `<answer_memory>`,
after the frame, in the order they finished:

```text
<captioning><|im_end|>
<|im_start|>user
<3.0 seconds><|vision_start|><|image_pad|>×N<|vision_end|>
<answer_memory>{"timestep": "00:00:02", "content": "A man in a red apron walks in."}</answer_memory>
<answer_memory>{"timestep": "00:00:01", "content": "The door is still closed."}</answer_memory>
<|im_end|>
<|im_start|>assistant
```

A turn without a frame writes its time with three decimals in place of the
frame:

```text
<silence><|im_end|>
<|im_start|>user
<question_memory>{"timestep": "00:00:03.5", "content": "What is he holding?"}</question_memory>
<3.500 seconds>
<|im_end|>
<|im_start|>assistant
```

With `max_visual_frames=4` and `compact_to_visual_frames=1`, the frame at 4 s
is one too many, and the new source keeps the turns from 3.5 s on.
Archived blocks are joined by newlines, with none after the last, so the first
kept turn follows directly:

```text
<|im_start|>system
You are a real-time streaming video assistant.<|im_end|>
<|im_start|>user
<question_memory>{"timestep": "00:00:01", "content": "Tell me when the door opens."}</question_memory>
<answer_memory>{"timestep": "00:00:02", "content": "A man in a red apron walks in."}</answer_memory>
<answer_memory>{"timestep": "00:00:01", "content": "The door is still closed."}</answer_memory><question_memory>{"timestep": "00:00:03.5", "content": "What is he holding?"}</question_memory>
<3.500 seconds>
<|im_end|>
<|im_start|>assistant
<silence><|im_end|>
<|im_start|>user
<4.0 seconds><|vision_start|><|image_pad|>×N<|vision_end|>
<|im_end|>
<|im_start|>assistant
```

The patch positions carry the frame's index in the source video. Pass
`frame_index` and `fps` with an image to set it; the frame's timestamp in the
text is then `frame_index / fps`. Without them the index counts source frames
at 30 fps, which is a fallback, not a substitute: training writes the real
index, so a 25 fps clip has `t` running 0, 25, 50, ... and a clip of 270
sampled seconds reaches 6725. An index that advances faster than the source
fps walks the temporal axis of M-RoPE straight out of the trained range, and
nothing in the rendered text shows it, because the timestamp is recovered as
`frame_index / fps`.

### Captioning

Captioning is a switch, off by default. A session that turns it on opens with
`<caption_begin>`, once, ahead of the first frame, which is where the packed
training data puts it, exactly once per sample. The source may then sample
`<captioning>` as well as `<silence>` and `<response>`, and the session still
takes a question at any time and still answers it. With the switch off,
`<captioning>` stays out of `allowed_token_ids`, so the model is never offered
a token the session has not asked for.

`optional_controls` maps each switchable control to the text that opens a
session using it; every other control is always available:

```python
optional_controls = {"caption": "<caption_begin>\n"}
```

`POST /v1/ov/sessions` takes `enable: ["caption"]`, and so does
`client.session(enable=["caption"])`. Compaction keeps the opening text ahead
of the archive, so the switch survives once the turn that carried it is
dropped. Some OV2.1 checkpoints were trained without captioning; when
`<captioning>` is not a single token the renderer drops that control, and
`enable=["caption"]` then fails naming what the checkpoint can switch on.

Captioning and question answering are not exclusive. In the packed
`streaming_caption_qa` data, counted over 20 objects, `<caption_begin>` appears
60 times, `<captioning>` 213, `<response>` 46 and `question_memory` 60, all in
the same samples: one session captions on its own schedule and answers
questions as they arrive.

## What a renderer defines

```python
class Renderer:
    engine_args: ClassVar[dict[str, Any]] = {}  # defaults for vLLM's CLI flags
    controls: dict[str, str]                    # name -> token; all but "silence" branch
    optional_controls: ClassVar[dict[str, str]] = {}   # switchable control -> opening text
    memory_tags: ClassVar[dict[str, str]] = {
        "question": "question_memory",
        "response": "answer_memory",
        "caption": "caption_memory",
    }
    system_prompt: ClassVar[str | None] = None
    task_prefix: ClassVar[str] = ""             # raw prefix before archives and frames
    timestep_millis: ClassVar[bool] = False     # milliseconds in memory timesteps
    max_response_tokens: ClassVar[int] = 256

    def render_frame(self, image, timestamp_sec, *, frame_index=None, fps=None) -> VisualFrame: ...
    def render_user(self, turn) -> str: ...
    def render_memory(self, memory) -> str: ...              # <tag>{"timestep", "content"}</tag>
    def archive_turn(self, turn) -> tuple[Memory, ...]: ...  # a dropped turn's blocks, in prompt order
    def render_archive(self, archive) -> str: ...
```

A new family usually writes `render_frame` and `render_user` and sets the class
attributes. `render_prompt`, `render_append` and `check_text` are shared and
should stay as they are.

At startup the renderer looks up the id of every control in the checkpoint's
tokenizer and refuses to load if one is not a single token. `check_text`
refuses user text that contains any token the tokenizer added, which covers the
ChatML markers, the controls and, for UniVIA, the memory tags. It runs on
questions, on a system message and on committed answers.

`serve` passes `engine_args` to vLLM's argument parser as defaults, so a flag
on the command line still wins; a dict flag adds its keys to the renderer's
dict.

## Changing a format

A layout change, such as the order of frames and questions, a separator or the
precision of a timestamp, is a change to `render_user` or `render_frame`. A
change to the memory blocks, such as other JSON fields or the write-back time,
is a change to `render_memory` or `render_user`; the turn's own time is
`turn.timestamp_sec`. The archive layout is `render_archive`, and the order of
a dropped turn's blocks is `archive_turn`. Change the golden test with it:
`tests/test_renderers.py` compares UniVIA against the training text and OV2.1
against the recorded session, and its contract test checks that appending
equals rendering from scratch.

A control whose branch is written back like an answer is two lines: its token
in `controls` and its tag in `memory_tags`. Turning on UniVIA's captioning
needs only the first, since `caption_memory` is already a tag:

```python
class UniVIACaptionRenderer(UniVIARenderer):
    controls = {**UniVIARenderer.controls, "caption": "<captioning>"}
```

The session handles every control the same way: it opens a branch, reports the
text with the control's name in `response.output_text.done`, and writes it back
when the branch finishes. Only `response` waits for `commit_answer` when a
session uses `answer_commit="client"`. A control that should do something
else, such as calling another service, needs code in the session.

For checkpoints that require a raw task marker, a renderer can set
`task_prefix = "<caption_begin>\n"`. It appears once at the start of the first
user turn, before archived questions, memories and retained frames, including
after every compaction. Incremental appends do not repeat it. The default is
empty; later questions do not switch the task automatically. Define this in
the renderer rather than sending the marker as user text, which is checked for
protocol tokens. This carries forward the former replay harness's optional
task prefix; it does not restore that removed Python API. The tests check
prompt preservation, not model accuracy after removing visual history.

A renderer of your own does not need to live in this repository. Put it in any
importable module and name it on the command line. For example, a checkpoint
trained with questions before frames:

```python
# my_pkg/renderers.py
from ov_streaming.renderers.univia import UniVIARenderer


class QuestionFirstRenderer(UniVIARenderer):
    def render_user(self, turn):
        return (
            "".join(map(self.render_memory, turn.memories))
            + "".join(map(self.render_memory, turn.questions))
            + "".join(frame.text for frame in turn.frames)
        )
```

```bash
ov-streaming serve /path/to/checkpoint --renderer my_pkg.renderers:QuestionFirstRenderer
python scripts/check_branching.py --renderer my_pkg.renderers:QuestionFirstRenderer --model /path/to/checkpoint
```

If vLLM has no class for the model, register one from a vLLM plugin (an entry
point in the `vllm.general_plugins` group that calls
`ModelRegistry.register_model`).
