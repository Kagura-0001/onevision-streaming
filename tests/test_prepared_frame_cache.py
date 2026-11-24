import base64
import io
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch
from PIL import Image

from ov_streaming import qa
from ov_streaming.image_cache import PreparedFrameCache, _resident_bytes
from ov_streaming.types import VisualFrame


def image_part(color="red", **extra):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buffer, "PNG")
    return dict(type="image", data=base64.b64encode(buffer.getvalue()).decode(),
                timestamp=1., frame_index=25, fps=25., source_size=[16, 16], **extra)


class Renderer:
    system_prompt = "Video assistant"
    preserve_source_pixels = True
    image_processor = SimpleNamespace(patch_size=16, merge_size=3)

    def __init__(self):
        self.calls = 0

    def render_frame(self, image, at, *, frame_index, fps):
        self.calls += 1
        return VisualFrame(
            f"<{frame_index / fps}><|vision_start|>pixels<|vision_end|>\n",
            {"pixel_values": torch.tensor([list(image.getpixel((0, 0)))], dtype=torch.float32),
             "patch_positions": torch.tensor([[frame_index, image.width, image.height]])},
        )


def render(renderer, part, cache=None):
    return qa.render_packed(renderer, [{"role": "user", "content": [part]}],
                            prepared_frame_cache=cache)


class PreparedFrameCacheTest(unittest.TestCase):
    def test_content_time_fps_and_source_crop_remain_distinct(self):
        renderer, cache = Renderer(), PreparedFrameCache(1024 * 1024)
        base = image_part()
        cases = [base, image_part("blue"), dict(base, timestamp=2., frame_index=50),
                 dict(base, fps=50.), dict(base, source_size=[8, 8])]
        for part in cases:
            expected = render(renderer, part)
            first = render(renderer, part, cache)
            second = render(renderer, part, cache)
            self.assertEqual(expected[0], first[0])
            self.assertEqual(first[0], second[0])
            for key, value in expected[1][0].data.items():
                self.assertTrue(torch.equal(value, second[1][0].data[key]))
        self.assertEqual(renderer.calls, 2 * len(cases))
        self.assertEqual(cache.stats()["hits"], len(cases))
        with self.assertRaises(ValueError):
            render(renderer, dict(base, source_size=[17, 16]), cache)

    def test_timestamp_text_is_per_request_not_cached(self):
        renderer, cache = Renderer(), PreparedFrameCache(1024 * 1024)
        base = image_part()
        first = render(renderer, dict(base, timestamp_text="<first>"), cache)
        second = render(renderer, dict(base, timestamp_text="<second>"), cache)
        self.assertIn("<first><|vision_start|>", first[0])
        self.assertIn("<second><|vision_start|>", second[0])
        self.assertEqual(renderer.calls, 1)

    def test_request_tensor_mutation_does_not_change_cached_frame(self):
        renderer, cache = Renderer(), PreparedFrameCache(1024 * 1024)
        prompt, frames = render(renderer, image_part(), cache)
        backend = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **k: [1]))
        first = qa.engine_prompt(backend, prompt, frames)
        first["multi_modal_data"]["image"]["pixel_values"].zero_()
        second = qa.engine_prompt(backend, prompt, frames)
        self.assertEqual(second["multi_modal_data"]["image"]["pixel_values"][0, 0], 255)
        self.assertEqual(frames[0].data["pixel_values"][0, 0], 255)

    def test_single_flight_but_distinct_frames_prepare_in_parallel(self):
        cache = PreparedFrameCache(1024 * 1024)
        lock = threading.Lock()
        calls = active = maximum = 0

        def prepare():
            nonlocal calls, active, maximum
            with lock:
                calls += 1
                active += 1
                maximum = max(maximum, active)
            time.sleep(.02)
            with lock:
                active -= 1
            return VisualFrame("frame", {"x": torch.ones(16)})

        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda _: cache.get_or_prepare("same", prepare), range(8)))
        self.assertEqual(calls, 1)
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(lambda i: cache.get_or_prepare(i, prepare), range(4)))
        self.assertGreater(maximum, 1)
        self.assertGreater(cache.stats()["coalesced"], 0)

    def test_bytes_include_full_key_and_backing_storage(self):
        backing = torch.ones(10000)
        small_view = backing[:1]
        self.assertGreaterEqual(_resident_bytes(small_view), 40000)
        key = "x" * 50000
        frame = VisualFrame("f", {"view": small_view})
        cache = PreparedFrameCache(60000)
        cache.get_or_prepare(key, lambda: frame)
        self.assertEqual(cache.stats()["entries"], 0)
        self.assertEqual(cache.stats()["oversized_bypasses"], 1)
        unit = _resident_bytes(("a", frame)) + PreparedFrameCache.ENTRY_OVERHEAD
        cache = PreparedFrameCache(unit * 2)
        for key in ("a", "b", "c"):
            cache.get_or_prepare(key, lambda: frame)
        self.assertEqual(cache.stats()["entries"], 2)
        self.assertEqual(cache.stats()["evictions"], 1)
        self.assertLessEqual(cache.stats()["retained_bytes"], unit * 2)

    def test_exception_is_not_cached_and_disabled_path_is_uncached(self):
        cache = PreparedFrameCache(10000)
        with self.assertRaisesRegex(ValueError, "bad image"):
            cache.get_or_prepare("a", lambda: (_ for _ in ()).throw(ValueError("bad image")))
        frame = VisualFrame("f", {"x": torch.ones(1)})
        self.assertIs(cache.get_or_prepare("a", lambda: frame), frame)
        self.assertEqual(cache.stats()["misses"], 2)
        self.assertEqual(cache.stats()["hits"], 0)
        disabled = PreparedFrameCache(0)
        disabled.get_or_prepare("a", lambda: frame)
        disabled.get_or_prepare("a", lambda: frame)
        self.assertEqual(disabled.stats()["entries"], 0)


if __name__ == "__main__":
    unittest.main()
