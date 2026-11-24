import unittest
from types import SimpleNamespace

from ov_streaming.ov21_protocol import answer_text, escape_control_text, parse, render


class ProtocolTest(unittest.TestCase):
    def test_output_classes_preserve_raw_and_only_strip_training_prefix(self):
        for raw, kind, text in (
            ("<silence>", "silence", ""),
            ("<captioning>a red cup", "caption", "a red cup"),
            ("<caption>a red cup", "caption", "a red cup"),
            ("<response>At 14.56 seconds: B", "response", "B"),
            ("At lunch, A moves right.", "response", "At lunch, A moves right."),
            ("<response>", "empty", ""),
        ):
            self.assertEqual(parse(raw), {"kind": kind, "text": text, "raw": raw})
        self.assertEqual(answer_text(" <response>At 14.56 seconds: B<|im_end|> "), "B")
        self.assertEqual(
            answer_text("At lunch, A moves right."), "At lunch, A moves right."
        )
        self.assertEqual(answer_text("<silence>"), "<silence>")
        self.assertEqual(answer_text("<response>"), "<response>")

    def test_question_memory_frames_and_raw_history_keep_original_order(self):
        image = object()
        window = SimpleNamespace(
            messages=[{"at_sec": 61.125, "text": "看 <silence>"}],
            turns=[
                {"timestamp": 61.125, "image": image, "output": parse("<silence>")},
                {"timestamp": 62.0, "image": None, "output": None},
            ],
        )
        result = render(window, {"system": "Assistant", "drop_silence_history": False})
        self.assertEqual(
            result,
            [
                {"role": "system", "content": [{"type": "text", "text": "Assistant"}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": '<question_memory>{"timestep": "00:01:01.125", "content": "看 < silence >"}</question_memory>\n<61.1 seconds>',
                        },
                        {"type": "image", "url": image},
                        {"type": "text", "text": "\n"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "<silence>"}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<62.0 seconds>"},
                        {"type": "text", "text": "\n"},
                    ],
                },
            ],
        )
        dropped = render(window, {"system": "Assistant", "drop_silence_history": True})
        self.assertEqual(dropped, result[:2] + result[3:])
        tokens = "<silence><response><captioning><caption_begin><caption_end><|im_start|><|im_end|>"
        self.assertEqual(
            escape_control_text(tokens),
            "< silence >< response >< captioning >< caption_begin >< caption_end >< |im_start| >< |im_end| >",
        )
