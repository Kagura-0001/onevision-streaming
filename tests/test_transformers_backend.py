from __future__ import annotations

import asyncio
import re
import unittest
from types import SimpleNamespace

import torch
from PIL import Image

from ov_streaming.renderers.base import Renderer
from ov_streaming.session import OVSession
from ov_streaming.transformers_backend import (
    TransformersBackend,
    clone_past_key_values,
)
from ov_streaming.types import VisualFrame


class _FakeLayer:
    def __init__(self, *, is_linear: bool):
        if is_linear:
            self.conv_states = torch.zeros(1, 4, 3)
            self.recurrent_states = torch.zeros(1, 2, 4, 4)
            self.has_previous_state = True
        else:
            self.keys = torch.zeros(1, 2, 0, 4)
            self.values = torch.zeros(1, 2, 0, 4)


class _FakeHybridCache:
    def __init__(self):
        self.layers = [_FakeLayer(is_linear=True), _FakeLayer(is_linear=False)]


class _FakeTokenizer:
    def __init__(self):
        self.vocab = {
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "<|vision_start|>": 3,
            "<|vision_end|>": 4,
            "<|image_pad|>": 5,
            "<silence>": 6,
            "<response>": 7,
            "<captioning>": 8,
            "Water": 10,
            " boils": 11,
            ".": 12,
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.eos_token_id = 2
        self._next_id = 50
        self._pattern = re.compile(
            "(" + "|".join(re.escape(k) for k in self.vocab) + ")"
        )

    def get_added_vocab(self):
        return {
            k: v
            for k, v in self.vocab.items()
            if k.startswith("<") and k.endswith(">")
        }

    def encode(self, text, add_special_tokens=False):
        ids = []
        for part in self._pattern.split(text):
            if not part:
                continue
            if part in self.vocab:
                ids.append(self.vocab[part])
                continue
            for word in re.findall(r"\S+|\s+", part):
                if word not in self.vocab:
                    self.vocab[word] = self._next_id
                    self.id_to_token[self._next_id] = word
                    self._next_id += 1
                ids.append(self.vocab[word])
        return ids

    def decode(self, ids, skip_special_tokens=True):
        pieces = []
        for tid in ids:
            tok = self.id_to_token.get(int(tid), "")
            if skip_special_tokens and tok.startswith("<") and tok.endswith(">"):
                continue
            pieces.append(tok)
        return "".join(pieces)


class _FakeRenderer(Renderer):
    controls = {"silence": "<silence>", "response": "<response>", "caption": "<captioning>"}
    optional_controls = {"caption": "<caption_begin>\n"}
    memory_tags = {**Renderer.memory_tags, "caption": "answer_memory"}
    system_prompt = "You are a real-time streaming video assistant."

    def render_frame(self, image, timestamp_sec, *, frame_index=None, fps=None):
        fidx = 0 if frame_index is None else int(frame_index)
        return VisualFrame(
            f"<{timestamp_sec:.1f} seconds><|vision_start|><|image_pad|><|vision_end|>\n",
            {
                "pixel_values": torch.zeros(9, 3),
                "image_grid_thw": torch.tensor([[1, 3, 3]]),
                "patch_positions": torch.tensor([[fidx, i // 3, i % 3] for i in range(9)]),
            },
        )

    def render_user(self, turn):
        frames = "".join(f.text for f in turn.frames)
        return (
            "".join(f"{self.render_memory(m)}\n" for m in turn.questions)
            + (frames or f"<{turn.timestamp_sec:.3f} seconds>\n")
            + "".join(f"{self.render_memory(m)}\n" for m in turn.memories)
        )


class _FakeModel:
    def __init__(self, control_sequence, branch_tokens):
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(
            eos_token_id=2,
            image_token_id=5,
            video_token_id=99,
            max_position_embeddings=4096,
            vision_config=SimpleNamespace(spatial_merge_size=3),
        )
        self.generation_config = SimpleNamespace(eos_token_id=2)
        self._controls = list(control_sequence)
        self._branch_tokens = list(branch_tokens)
        self.calls = []

    def __call__(
        self,
        *,
        input_ids,
        position_ids,
        cache_position,
        past_key_values=None,
        use_cache=True,
        **kwargs,
    ):
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "position_ids": position_ids.clone(),
                "cache_position": cache_position.clone(),
                "has_pixels": "pixel_values" in kwargs,
                "cache": past_key_values,
            }
        )
        cache = past_key_values if past_key_values is not None else _FakeHybridCache()
        # Mutate both GDN and attention layers in-place to verify branch isolation.
        step_len = input_ids.shape[1]
        cache.layers[0].conv_states.add_(float(step_len))
        cache.layers[0].recurrent_states.add_(1.0)
        new_k = torch.ones(1, 2, step_len, 4) * float(cache_position[0].item())
        cache.layers[1].keys = torch.cat([cache.layers[1].keys, new_k], dim=2)
        cache.layers[1].values = torch.cat([cache.layers[1].values, new_k], dim=2)

        vocab_size = 200
        logits = torch.full((1, step_len, vocab_size), -10.0)
        # Branch decode calls pass [7] (<response>) or single generated tokens without pixels.
        if not kwargs and step_len == 1 and (
            int(input_ids[0, 0].item()) in (7, 10, 11, 12)
        ):
            nxt = self._branch_tokens.pop(0)
        else:
            nxt = self._controls.pop(0)
        logits[0, -1, nxt] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=cache)


def _fake_mrope(input_ids, attention_mask, patch_positions, merge_size, image_id, video_id):
    seq_len = input_ids.shape[1]
    base = torch.arange(seq_len, dtype=torch.long).view(1, 1, -1).expand(3, 1, -1).clone()
    return base, torch.tensor([[0]], dtype=torch.long)


class TransformersBackendTest(unittest.TestCase):
    def test_clone_past_key_values_isolates_hybrid_tensors(self):
        cache = _FakeHybridCache()
        cache.layers[0].conv_states.fill_(3.0)
        cache.layers[1].keys = torch.ones(1, 2, 4, 4)

        cloned = clone_past_key_values(cache)
        cloned.layers[0].conv_states.fill_(99.0)
        cloned.layers[1].keys.zero_()

        self.assertTrue(torch.all(cache.layers[0].conv_states == 3.0))
        self.assertTrue(torch.all(cache.layers[1].keys == 1.0))

    def test_session_fork_and_commit_without_vllm(self):
        async def run():
            tokenizer = _FakeTokenizer()
            renderer = _FakeRenderer(tokenizer, SimpleNamespace())
            # Turn 0 -> silence(6); Turn 1 -> response(7) -> branch yields "Water boils." + eos(2); Turn 2 -> silence(6)
            model = _FakeModel(
                control_sequence=[6, 7, 6],
                branch_tokens=[10, 11, 12, 2],
            )
            backend = TransformersBackend(
                model, tokenizer, max_model_len=4096, mrope_positions_fn=_fake_mrope
            )
            session = OVSession(backend, renderer, "sess-tf", enable=[])
            events = []

            async def reader():
                async for ev in session.events():
                    events.append(ev)

            task = asyncio.create_task(reader())
            img = Image.new("RGB", (48, 48))
            await session.send_messages(
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "image_pil", "image_pil": img, "timestamp_sec": 0.0}],
                    }
                ]
            )
            await session.send_messages(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_pil", "image_pil": img, "timestamp_sec": 1.0},
                            {"type": "text", "text": "Is the water boiling?"},
                        ],
                    }
                ]
            )
            # Let the branch finish so Turn 2 commits <answer_memory>.
            await session.drain()
            await asyncio.sleep(0.05)
            await session.send_messages(
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "image_pil", "image_pil": img, "timestamp_sec": 2.0}],
                    }
                ]
            )
            await session.close()
            await task
            return events, model, tokenizer

        events, model, tokenizer = asyncio.run(run())
        controls = [e.control for e in events if e.type == "input.done"]
        self.assertEqual(controls, ["silence", "response", "silence"])
        done_texts = [e.text for e in events if e.type == "response.output_text.done"]
        self.assertEqual(done_texts, ["Water boils."])

        # Verify every call received 3D M-RoPE position_ids of shape (3, 1, chunk_len).
        for call in model.calls:
            self.assertEqual(call["position_ids"].dim(), 3)
            self.assertEqual(call["position_ids"].shape[0], 3)

        # Verify Turn 2's prompt contains the committed <answer_memory> and
        # its cache_position starts immediately after Turn 1's source tokens
        # (not polluted by branch decode tokens!).
        source_calls = [c for c in model.calls if c["has_pixels"]]
        self.assertEqual(len(source_calls), 3)
        t0_len = source_calls[0]["input_ids"].shape[1]
        t1_len = source_calls[1]["input_ids"].shape[1]
        self.assertEqual(int(source_calls[1]["cache_position"][0].item()), t0_len)
        self.assertEqual(int(source_calls[2]["cache_position"][0].item()), t0_len + t1_len)
        turn2_text = tokenizer.decode(
            source_calls[2]["input_ids"][0].tolist(), skip_special_tokens=False
        )
        self.assertIn("<answer_memory>", turn2_text)
        self.assertIn("Water boils.", turn2_text)


if __name__ == "__main__":
    unittest.main()
