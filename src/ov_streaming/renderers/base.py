from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import pairwise
from typing import Any, ClassVar

from ..types import Memory, Turn, VisualFrame


def format_timecode(seconds: float, *, millis: bool = False) -> str:
    total = round(seconds * 1000) if millis else round(seconds) * 1000
    whole, ms = divmod(total, 1000)
    hours, whole = divmod(whole, 3600)
    minutes, whole = divmod(whole, 60)
    fraction = f".{ms:03d}".rstrip("0") if ms else ""
    return f"{hours:02d}:{minutes:02d}:{whole:02d}{fraction}"


class Renderer:
    """ChatML text of a streaming session for one checkpoint family.

    The session owns the turns; a renderer only writes them out, so the
    source prompt after turn n is ``render_prompt(turns[:n + 1])`` and the
    text appended for turn n is ``render_append(turns[n], turns[n - 1].control)``.
    """

    engine_args: ClassVar[dict[str, Any]] = {}
    # Control name -> token. Every control except silence opens a branch.
    controls: dict[str, str]
    # Explicit enable selects optional controls and their opening text.
    optional_controls: ClassVar[dict[str, str]] = {}
    memory_tags: ClassVar[dict[str, str]] = {
        "question": "question_memory",
        "response": "answer_memory",
        "caption": "caption_memory",
    }
    system_prompt: ClassVar[str | None] = None
    # Raw protocol text at the start of every initial or compacted context.
    task_prefix: ClassVar[str] = ""
    timestep_millis: ClassVar[bool] = False
    max_response_tokens: ClassVar[int] = 256

    def __init__(self, tokenizer: Any, model_config: Any) -> None:
        self.control_ids = {}
        for name, token in self.controls.items():
            ids = tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"the tokenizer must encode {token} as one token")
            self.control_ids[name] = ids[0]
        # Added tokens, special or not, are what the tokenizer splits out of
        # plain text; the special-token list depends on how a checkpoint was saved.
        self.added_tokens = tuple(tokenizer.get_added_vocab())
        # A checkpoint whose tokenizer lacks the token cannot switch it on.
        self.optional_controls = {
            name: prefix
            for name, prefix in self.optional_controls.items()
            if name in self.control_ids
        }
        self.base_controls = tuple(
            name for name in self.control_ids if name not in self.optional_controls
        )

    def session_controls(
        self, enable: Sequence[str] | None
    ) -> tuple[tuple[str, ...], str]:
        """The controls a session may sample, and the text that opens it."""
        # Omission preserves the original renderer controls and opening text.
        if enable is None:
            return tuple(self.control_ids), ""
        switches = tuple(dict.fromkeys(enable))
        unknown = [name for name in switches if name not in self.optional_controls]
        if unknown:
            available = ", ".join(self.optional_controls) or "nothing"
            raise ValueError(
                f"this checkpoint cannot switch on {', '.join(unknown)}; "
                f"it can switch on {available}"
            )
        prefix = "".join(self.optional_controls[name] for name in switches)
        return (*self.base_controls, *switches), prefix

    def render_frame(
        self,
        image: Any,
        timestamp_sec: float,
        *,
        frame_index: int | None = None,
        fps: float | None = None,
    ) -> VisualFrame:
        raise NotImplementedError

    def render_user(self, turn: Turn) -> str:
        raise NotImplementedError

    def render_codec_frame(self, image, timestamp_sec, *, codec, **kwargs):
        raise ValueError("codec canvases require the ov21 renderer")

    def render_memory(self, memory: Memory) -> str:
        tag = self.memory_tags[memory.kind]
        timestep = format_timecode(memory.timestamp_sec, millis=self.timestep_millis)
        item = {"timestep": timestep, "content": memory.text}
        return f"<{tag}>{json.dumps(item, ensure_ascii=False)}</{tag}>"

    def archive_turn(self, turn: Turn) -> tuple[Memory, ...]:
        # A dropped turn's questions and memories, in the order render_user shows them.
        return (*turn.questions, *turn.memories)

    def render_archive(self, archive: Sequence[Memory]) -> str:
        return "".join(map(self.render_memory, archive))

    def render_prompt(
        self,
        turns: Sequence[Turn],
        archive: Sequence[Memory] = (),
        system: str | None = None,
        prefix: str = "",
    ) -> str:
        system = system or self.system_prompt
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
        user = (
            self.task_prefix
            + prefix
            + self.render_archive(archive)
            + self.render_user(turns[0])
        )
        prompt += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        for previous, turn in pairwise(turns):
            prompt += self.render_append(turn, previous.control)
        return prompt

    def render_append(self, turn: Turn, previous: str) -> str:
        # The engine drops the sampled control when the next input arrives,
        # so the append writes it back as its first token.
        return (
            f"{self.controls[previous]}<|im_end|>\n"
            f"<|im_start|>user\n{self.render_user(turn)}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def render_verification_suffix(self, instruction: str) -> str:
        # The computed prefix ends at the assistant header, before its sampled
        # control. Close that empty turn; do not force acceptance on the branch.
        return (
            "<|im_end|>\n<|im_start|>user\n"
            f"{instruction}<|im_end|>\n<|im_start|>assistant\n"
        )

    def check_text(self, text: str) -> None:
        for token in self.added_tokens:
            if token in text:
                raise ValueError(f"text must not contain the token {token}")
