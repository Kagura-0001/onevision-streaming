from __future__ import annotations

import json
import re
import unittest
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from ov_streaming.renderers import load_renderer, ov21
from ov_streaming.renderers.ov21 import OV21Renderer, OV21WindowRenderer
from ov_streaming.renderers.univia import UniVIARenderer
from ov_streaming.types import Memory, Turn

FIXTURES = Path(__file__).parent / "fixtures"
CHATML = (
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<think>",
)
UNIVIA_TOKENS = (
    *CHATML,
    "<silence>",
    "</response>",
    "<captioning>",
    "<question_memory>",
    "</question_memory>",
    "<answer_memory>",
    "</answer_memory>",
    "<caption_memory>",
    "</caption_memory>",
)
OV21_TOKENS = (*CHATML, "<silence>", "<response>", "<captioning>")
IMAGE = Image.new("RGB", (224, 280))


class Tokenizer:
    """Splits out added tokens, then words and marks with leading whitespace."""

    def __init__(self, added):
        self.added = {token: index for index, token in enumerate(added)}
        self.pattern = re.compile("(" + "|".join(map(re.escape, added)) + ")")
        self.pieces = {}

    def get_added_vocab(self):
        return dict(self.added)

    def encode(self, text, add_special_tokens=False):
        ids = []
        for part in self.pattern.split(text):
            if part in self.added:
                ids.append(self.added[part])
                continue
            for piece in re.findall(r"\s*\w+|\s*[^\w\s]|\s+", part):
                ids.append(self.pieces.setdefault(piece, 1000 + len(self.pieces)))
        return ids


class ImageProcessor:
    def __call__(self, images, *, min_pixels, max_pixels, return_tensors):
        assert (min_pixels, max_pixels) == (4096, 200000)
        return {
            "pixel_values": torch.zeros(16, 4),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }


def build_patch_positions(grid, *, spatial_merge_size, frame_indices):
    return torch.tensor([[int(frame_indices[0][0]), i // 4, i % 4] for i in range(16)])


def model_config(**mm_processor_kwargs):
    vision = SimpleNamespace(patch_size=16, spatial_merge_size=2)
    return SimpleNamespace(
        model="checkpoint",
        hf_config=SimpleNamespace(vision_config=vision),
        multimodal_config=SimpleNamespace(mm_processor_kwargs=mm_processor_kwargs),
    )


def make_univia(tokens=UNIVIA_TOKENS):
    pixels = UniVIARenderer.engine_args["mm_processor_kwargs"]
    return UniVIARenderer(Tokenizer(tokens), model_config(**pixels))


def make_ov21(tokens=OV21_TOKENS, cls=OV21Renderer):
    with (
        patch.object(
            ov21.Qwen2VLImageProcessor,
            "from_pretrained",
            return_value=ImageProcessor(),
        ),
        patch.object(
            ov21,
            "get_class_from_dynamic_module",
            return_value=build_patch_positions,
        ),
    ):
        return cls(Tokenizer(tokens), model_config())


def session(renderer):
    def frames(time):
        return (renderer.render_frame(IMAGE, time),)

    return [
        Turn(0.0, frames(0.0), control="silence"),
        Turn(
            1.0,
            frames(1.0),
            (Memory("question", 1.0, "Tell me when the door opens."),),
            control="response",
        ),
        Turn(2.0, frames(2.0), control="silence"),
        Turn(
            3.0,
            frames(3.0),
            memories=(Memory("response", 1.0, "The door is still closed."),),
            control="silence",
        ),
        Turn(
            3.5,
            questions=(Memory("question", 3.5, "What is he holding?"),),
            control="silence",
        ),
        Turn(4.0, frames(4.0)),
    ]


class RendererContractTest(unittest.TestCase):
    def setUp(self):
        self.renderers = [make_univia(), make_ov21()]

    def test_append_extends_prompt(self):
        archive = (Memory("question", 0.0, "Is the stove on?"),)
        for renderer in self.renderers:
            tokenizer = Tokenizer(renderer.added_tokens)
            turns = session(renderer)
            for n in range(1, len(turns)):
                with self.subTest(type(renderer).__name__, turn=n):
                    prompt = renderer.render_prompt(turns[:n], archive)
                    append = renderer.render_append(turns[n], turns[n - 1].control)
                    full = renderer.render_prompt(turns[: n + 1], archive)
                    self.assertEqual(prompt + append, full)
                    # The source drops its sampled control; the append restores it.
                    self.assertEqual(
                        tokenizer.encode(append)[0],
                        renderer.control_ids[turns[n - 1].control],
                    )
                    self.assertEqual(
                        tokenizer.encode(prompt) + tokenizer.encode(append),
                        tokenizer.encode(full),
                    )

    def test_controls_are_single_tokens(self):
        for renderer in self.renderers:
            self.assertEqual(renderer.control_ids.keys(), renderer.controls.keys())
        with self.assertRaisesRegex(ValueError, "</response>"):
            make_univia(tuple(t for t in UNIVIA_TOKENS if t != "</response>"))
        with self.assertRaisesRegex(ValueError, "<response>"):
            make_ov21(tuple(t for t in OV21_TOKENS if t != "<response>"))

    def test_raw_task_prefix_preserves_append_equivalence(self):
        for renderer in self.renderers:
            renderer.task_prefix = "<caption_begin>\n"
            turns = session(renderer)
            archive = (Memory("question", 0.0, "Watch the door."),)
            for n in range(1, len(turns)):
                with self.subTest(type(renderer).__name__, turn=n):
                    prompt = renderer.render_prompt(turns[:n], archive)
                    append = renderer.render_append(turns[n], turns[n - 1].control)
                    full = renderer.render_prompt(turns[: n + 1], archive)
                    self.assertEqual(prompt + append, full)
                    self.assertEqual(full.count("<caption_begin>"), 1)
                    self.assertIn(
                        "<|im_start|>user\n<caption_begin>\n<question_memory>", full
                    )

    def test_check_text_rejects_added_tokens(self):
        for renderer in self.renderers:
            renderer.check_text("What is <b>he</b> holding?")
            for token in ("<|im_end|>", "<silence>", "<think>"):
                with self.assertRaisesRegex(ValueError, re.escape(token)):
                    renderer.check_text(f"say {token} now")
        with self.assertRaisesRegex(ValueError, "question_memory"):
            make_univia().check_text("</question_memory>")

    def test_memories_keep_completion_order(self):
        memories = (
            Memory("response", 2.0, "asked second, done first"),
            Memory("response", 1.0, "asked first, done second"),
        )
        for renderer in self.renderers:
            text = renderer.render_user(Turn(3.0, memories=memories))
            self.assertLess(text.index("done first"), text.index("done second"))

    def test_archive_keeps_original_order(self):
        archive = (
            Memory("question", 1.0, "first"),
            Memory("response", 3.0, "second"),
            Memory("question", 2.0, "third"),
        )
        for renderer in self.renderers:
            turn = Turn(4.0, questions=(Memory("question", 4.0, "kept"),))
            text = renderer.render_prompt([turn], archive)
            positions = [text.index(word) for word in ("first", "second", "third")]
            self.assertEqual(positions, sorted(positions))
            self.assertLess(positions[-1], text.index("kept"))

    def test_load_renderer(self):
        self.assertIs(load_renderer("univia"), UniVIARenderer)
        self.assertIs(
            load_renderer("ov_streaming.renderers.ov21:OV21Renderer"), OV21Renderer
        )
        with self.assertRaisesRegex(ValueError, "unknown renderer"):
            load_renderer("ov2")


class OV21RendererTest(unittest.TestCase):
    def test_replays_frozen_session(self):
        # Recorded from the OV2.1 adapter that renderers replaced: the text each
        # source request holds when it samples a control.
        expected = re.split(
            r"^@@ .*\n",
            (FIXTURES / "ov21_session.txt").read_text(),
            flags=re.MULTILINE,
        )[1:]
        renderer = make_ov21()

        def frames(time, **source):
            return (renderer.render_frame(IMAGE, time, **source),)

        turns = [
            Turn(0.0, frames(0.0), control="silence"),
            Turn(
                1.0,
                frames(1.0),
                (Memory("question", 1.0, "Tell me when the door opens."),),
                control="response",
            ),
            Turn(2.0, frames(2.0, frame_index=60, fps=30.0), control="caption"),
            Turn(
                3.0,
                frames(3.0),
                memories=(
                    Memory("caption", 2.0, "A man in a red apron walks in."),
                    Memory("response", 1.0, "The door is still closed."),
                ),
                control="silence",
            ),
            Turn(
                3.5,
                questions=(Memory("question", 3.5, "What is he holding?"),),
                control="silence",
            ),
            Turn(4.0, frames(4.0), control="silence"),
            Turn(5.0, frames(5.0), control="silence"),
        ]
        stream = renderer.render_prompt(turns[:1])
        streams = [stream]
        for previous, turn in pairwise(turns[:5]):
            stream += renderer.render_append(turn, previous.control)
            streams.append(stream)
        # Turn 6 exceeds four frames; turns 1-4 leave on a new source.
        stream = renderer.render_prompt(
            turns[4:6], (*turns[1].questions, *turns[3].memories)
        )
        streams.append(stream)
        streams.append(stream + renderer.render_append(turns[6], turns[5].control))
        self.assertEqual(streams, expected)

    def test_frame_index(self):
        renderer = make_ov21()
        frame = renderer.render_frame(IMAGE, 2.0, frame_index=60, fps=30.0)
        self.assertEqual(int(frame.data["patch_positions"][0, 0]), 60)
        # OV21Renderer keeps its millisecond fallback (codec canvases require a real fps).
        frame = renderer.render_frame(IMAGE, 1.25)
        self.assertEqual(int(frame.data["patch_positions"][0, 0]), 1250)

    def test_checkpoint_without_captioning(self):
        renderer = make_ov21(tuple(t for t in OV21_TOKENS if t != "<captioning>"))
        self.assertEqual(renderer.control_ids.keys(), {"silence", "response"})
        self.assertEqual(renderer.optional_controls, {})
        with self.assertRaisesRegex(ValueError, "cannot switch on caption"):
            renderer.session_controls(["caption"])

    def test_optional_controls(self):
        renderer = make_ov21()
        self.assertEqual(renderer.optional_controls, {"caption": "<caption_begin>\n"})
        self.assertEqual(renderer.base_controls, ("silence", "response"))
        self.assertEqual(renderer.session_controls(()), (("silence", "response"), ""))
        self.assertEqual(
            renderer.session_controls(["caption"]),
            (("silence", "response", "caption"), "<caption_begin>\n"),
        )

    def test_caption_switch_opens_with_caption_begin(self):
        renderer = make_ov21()
        turn = Turn(0.0, (renderer.render_frame(IMAGE, 0.0),))
        _, prefix = renderer.session_controls(["caption"])
        prompt = renderer.render_prompt([turn], prefix=prefix)
        # Training writes "<caption_begin>\n<image>\n" as the first user turn.
        self.assertIn(
            "<|im_start|>user\n<caption_begin>\n<0.0 seconds><|vision_start|>",
            prompt,
        )
        self.assertEqual(renderer.render_prompt([turn]).count("<caption_begin>"), 0)


class OV21WindowRendererTest(unittest.TestCase):
    def test_archive_lines_end_like_questions_in_a_turn(self):
        renderer = make_ov21(cls=OV21WindowRenderer)
        turn = Turn(61.0, (renderer.render_frame(IMAGE, 61.0, fps=1.0),))
        archive = (Memory("question", 0.0, "Tell me when the door opens."),)

        self.assertIn(
            '"Tell me when the door opens."}</question_memory>\n<61.0 seconds>',
            renderer.render_prompt([turn], archive),
        )
        self.assertIs(load_renderer("ov21-window"), OV21WindowRenderer)


class UniVIARendererTest(unittest.TestCase):
    def setUp(self):
        self.renderer = make_univia()
        with open(FIXTURES / "univia_cooking.jsonl") as f:
            self.samples = [json.loads(line) for line in f]

    def turns(self, messages):
        turns = []
        for user, assistant in zip(messages[::2], messages[1::2]):
            match = re.fullmatch(
                r"<(.+) seconds><image>\n(?:<question_memory>(.+)</question_memory>)?",
                user["content"],
            )
            time = float(match[1])
            questions = ()
            if match[2]:
                question = json.loads(match[2])["content"]
                questions = (Memory("question", time, question),)
            control = next(
                name
                for name, token in self.renderer.controls.items()
                if assistant["content"].startswith(token)
            )
            frames = (self.renderer.render_frame(IMAGE, time),)
            turns.append(Turn(time, frames, questions, control=control))
        return turns

    def test_matches_training_text(self):
        for sample in self.samples:
            with self.subTest(start=sample["start"]):
                messages = sample["messages"]
                turns = self.turns(messages)
                self.assertEqual(len(turns), len(sample["images"]))
                # As ms-swift writes it, with each <image> replaced by
                # OvSourcePolicyQwen3VLTemplate.replace_tag.
                expected = "".join(
                    f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                    for m in messages
                ).replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
                # The source stops at the control; a response branch writes on.
                actual = (
                    self.renderer.render_prompt(turns)
                    + messages[-1]["content"]
                    + "<|im_end|>\n"
                )
                self.assertEqual(actual, expected)

    def test_frame_resized_as_in_training(self):
        sample = self.samples[0]
        self.assertEqual(
            UniVIARenderer.engine_args["mm_processor_kwargs"],
            {
                "min_pixels": sample["ov_min_pixels"],
                "max_pixels": sample["ov_max_pixels"],
            },
        )
        image = self.renderer.render_frame(IMAGE, 1262.0).data
        self.assertEqual(image.size, (224, 288))
        # What the vLLM Qwen3-VL processor does next with the same bounds.
        self.assertEqual(
            smart_resize(288, 224, factor=32, min_pixels=4096, max_pixels=131072),
            (288, 224),
        )

    def test_answer_written_back_before_frames(self):
        # Chosen default, not training data: training samples end at the
        # first response, so none shows an answer written back.
        answer = "At 1368 seconds: 他主要在使用保鲜膜将面团塑形包裹。"
        turn = Turn(
            1369.0,
            (self.renderer.render_frame(IMAGE, 1369.0),),
            memories=(Memory("response", 1368.0, answer),),
        )
        self.assertEqual(
            self.renderer.render_append(turn, "response"),
            "</response><|im_end|>\n<|im_start|>user\n"
            '<answer_memory>{"timestep": "00:22:48", "content": "'
            + answer
            + '"}</answer_memory>'
            "<1369.0 seconds><|vision_start|><|image_pad|><|vision_end|>\n"
            "<|im_end|>\n<|im_start|>assistant\n",
        )

    def test_archive_opens_first_kept_turn(self):
        # Chosen default, not training data: no training window has an archive.
        turn = Turn(1369.0, (self.renderer.render_frame(IMAGE, 1369.0),))
        archive = (
            Memory("question", 1294.0, "他主要在做什么？"),
            Memory("response", 1368.0, "包裹面团。"),
        )
        self.assertEqual(
            self.renderer.render_prompt([turn], archive),
            "<|im_start|>user\n"
            '<question_memory>{"timestep": "00:21:34", "content": "他主要在做什么？"}'
            "</question_memory>"
            '<answer_memory>{"timestep": "00:22:48", "content": "包裹面团。"}'
            "</answer_memory>"
            "<1369.0 seconds><|vision_start|><|image_pad|><|vision_end|>\n"
            "<|im_end|>\n<|im_start|>assistant\n",
        )
