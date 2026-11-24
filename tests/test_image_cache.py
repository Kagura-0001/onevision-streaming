import base64
import io
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from ov_streaming import qa
from ov_streaming.image_cache import DecodedImageCache, decode_rgb


def encoded(color, size=(16, 16)):
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, "PNG")
    return base64.b64encode(out.getvalue()).decode()


class ImageCacheTest(unittest.TestCase):
    def test_pixels_independent_and_lru_byte_limit(self):
        size = 16 * 16 * 3 + DecodedImageCache.ENTRY_OVERHEAD
        cache = DecodedImageCache(2 * size)
        a, b, c = [encoded(color) for color in ("red", "green", "blue")]
        first = cache.decode(a)
        first.putpixel((0, 0), (0, 0, 0))
        cache.decode(b)
        self.assertEqual(cache.decode(a).getpixel((0, 0)), (255, 0, 0))
        cache.decode(c)
        self.assertEqual(cache.stats()["evictions"], 1)
        cache.decode(b)
        self.assertEqual(cache.stats()["misses"], 4)
        self.assertLessEqual(cache.stats()["retained_bytes"], 2 * size)

    def test_concurrent_single_decode_and_oversized_bypass(self):
        cache = DecodedImageCache(4096)
        data = encoded("red")
        active = maximum = calls = 0
        lock = threading.Lock()
        def slow(value):
            nonlocal active, maximum, calls
            with lock:
                calls += 1
                active += 1
                maximum = max(maximum, active)
            time.sleep(.01)
            result = decode_rgb(value)
            with lock:
                active -= 1
            return result
        with patch("ov_streaming.image_cache.decode_rgb", side_effect=slow):
            with ThreadPoolExecutor(8) as pool:
                outputs = list(pool.map(cache.decode, [data] * 8))
        self.assertEqual((calls, maximum), (1, 1))
        self.assertEqual(len({id(x) for x in outputs}), 8)
        small = DecodedImageCache(100)
        with patch("ov_streaming.image_cache.decode_rgb", side_effect=slow):
            with ThreadPoolExecutor(4) as pool:
                list(pool.map(small.decode, [data] * 4))
        self.assertEqual(maximum, 1)
        self.assertEqual(small.stats()["retained_bytes"], 0)
        self.assertEqual(small.stats()["oversized_bypasses"], 4)

    def test_disabled_and_invalid_input(self):
        cache = DecodedImageCache(0)
        data = encoded("red")
        with patch("ov_streaming.image_cache.decode_rgb", wraps=decode_rgb) as fn:
            cache.decode(data)
            cache.decode(data)
            self.assertEqual(fn.call_count, 2)
        with self.assertRaises(ValueError):
            cache.decode("not-base64!")
        with self.assertRaises(ValueError):
            DecodedImageCache(-1)

    def test_render_metadata_stays_per_request(self):
        cache = DecodedImageCache(4096)
        seen = []
        class Renderer:
            system_prompt = "s"
            preserve_source_pixels = True
            image_processor = SimpleNamespace(patch_size=16, merge_size=3)
            def render_frame(self, image, at, *, frame_index, fps):
                seen.append((image.size, image.tobytes(), at, frame_index, fps))
                return SimpleNamespace(text=f"<{at}><|vision_start|>v", data=None)
        image = dict(type="image", data=encoded("red"), source_size=[16, 16],
                     timestamp=1., frame_index=25, fps=25., timestamp_text="")
        for at, question in [(1., "one"), (2., "two")]:
            image = dict(image, timestamp=at, frame_index=int(at * 25))
            messages = [dict(role="user", content=[image, dict(type="text", text=question)])]
            plain = qa.render_packed(Renderer(), messages)[0]
            cached = qa.render_packed(Renderer(), messages, image_cache=cache)[0]
            self.assertEqual(plain, cached)
            self.assertEqual(seen[-1], seen[-2])
        self.assertEqual(cache.stats()["hits"], 1)
        self.assertNotEqual(seen[0][2:], seen[-1][2:])
