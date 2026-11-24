"""Prompt format and output semantics of the OV Qwen3.5 streaming checkpoints."""

import json
import re


def escape_control_text(text):
    """Keep quoted control spellings in user text from becoming OV control tokens."""
    for token in (
        "<silence>",
        "<response>",
        "<captioning>",
        "<caption_begin>",
        "<caption_end>",
        "<|im_start|>",
        "<|im_end|>",
    ):
        text = text.replace(token, "< " + token[1:-1] + " >")
    return text


def parse(raw):
    if raw == "<silence>":
        return {"kind": "silence", "text": "", "raw": raw}
    if raw.startswith(("<caption>", "<captioning>")):
        return {
            "kind": "caption",
            "text": raw.removeprefix("<captioning>").removeprefix("<caption>"),
            "raw": raw,
        }
    text = raw.removeprefix("<response>")
    kind = "response" if text.strip() else "empty"
    # OV's training format is <response>At {timestamp} seconds: {answer}.
    # Keep raw for replay, but give every benchmark the answer body. Scoring
    # still uses the controller's observation time, never a generated timestamp.
    text = re.sub(r"^At\s+\d+(?:\.\d+)?\s+seconds:\s*", "", text)
    return {"kind": kind, "text": text, "raw": raw}


def render(window, settings):
    """Question memory before the oldest frame, one user turn per frame, past outputs as assistant turns."""
    memory = "".join(
        f"<question_memory>{json.dumps({'timestep': _clock(m['at_sec']), 'content': escape_control_text(m['text'])}, ensure_ascii=False)}</question_memory>\n"
        for m in window.messages
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": settings["system"]}]}
    ]
    last = len(window.turns) - 1
    for index, turn in enumerate(window.turns):
        image = turn["image"]
        prefix = (memory if index == 0 else "") + f"<{turn['timestamp']:.1f} seconds>"
        content = [{"type": "text", "text": prefix}]
        if image is not None:
            content.append({"type": "image", "url": image})
        content.append({"type": "text", "text": "\n"})
        messages.append({"role": "user", "content": content})
        if (
            index < last
            and turn["output"] is not None
            and not (
                settings["drop_silence_history"] and turn["output"]["kind"] == "silence"
            )
        ):
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": turn["output"]["raw"]}],
                }
            )
    return messages


def _clock(seconds):
    whole = int(seconds)
    clock = f"{whole // 3600:02}:{whole // 60 % 60:02}:{whole % 60:02}"
    return (
        clock if seconds == whole else clock + f"{seconds - whole:.6f}"[1:].rstrip("0")
    )


def answer_text(raw):
    text = raw.strip()
    text = re.sub(r"^<response>\s*", "", text)
    text = re.sub(r"^At\s+\d+(?:\.\d+)?\s+seconds:\s*", "", text)
    text = re.sub(r"<\|im_end\|>$", "", text).strip()
    # A generated control-only answer is a model miss, not a transport failure.
    return text or raw.strip()
