"""Byte-bounded cache for packed QA's pure base64-to-RGB decode."""
import base64
import hashlib
import io
import sys
import threading
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import Future

from PIL import Image


def decode_rgb(data):
    encoded = base64.b64decode(data, validate=True)
    with Image.open(io.BytesIO(encoded)) as image:
        return image.convert("RGB")


class DecodedImageCache:
    # Covers digest, tuple, bytes header and OrderedDict entry conservatively.
    ENTRY_OVERHEAD = 512

    def __init__(self, max_bytes=0):
        if max_bytes < 0:
            raise ValueError("decoded image cache size must be nonnegative")
        self.max_bytes = int(max_bytes)
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self._hits = self._misses = self._evictions = self._oversized = 0

    def decode(self, data):
        if not self.max_bytes:
            return decode_rgb(data)
        # RGB conversion is fixed; key the complete image, never request/frame id.
        key = hashlib.sha256(data.encode("ascii")).digest()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                self._hits += 1
                size, pixels = entry
                return Image.frombytes("RGB", size, pixels)
            self._misses += 1
            # Serialize misses, including oversized bypasses: at most one extra
            # decode in flight per cache, rather than one per concurrent request.
            image = decode_rgb(data)
            cost = image.width * image.height * 3 + self.ENTRY_OVERHEAD
            if cost > self.max_bytes:
                self._oversized += 1
                return image
            while self._entries and self._bytes + cost > self.max_bytes:
                _, (_, pixels) = self._entries.popitem(last=False)
                self._bytes -= len(pixels) + self.ENTRY_OVERHEAD
                self._evictions += 1
            self._entries[key] = (image.size, image.tobytes())
            self._bytes += cost
            return image

    def stats(self):
        with self._lock:
            return dict(max_bytes=self.max_bytes, retained_bytes=self._bytes,
                        entries=len(self._entries), hits=self._hits,
                        misses=self._misses, evictions=self._evictions,
                        oversized_bypasses=self._oversized)


def _resident_bytes(value):
    """Count retained CPU tensor storage, input strings and Python containers."""
    seen, storages = set(), set()

    def size(item):
        if id(item) in seen:
            return 0
        seen.add(id(item))
        total = sys.getsizeof(item)
        if hasattr(item, "untyped_storage"):
            if item.device.type != "cpu":
                raise ValueError("Prepared frame cache only accepts CPU tensors")
            storage = item.untyped_storage()
            identity = (storage.data_ptr(), storage.nbytes())
            if identity not in storages:
                storages.add(identity)
                total += storage.nbytes()
        elif isinstance(item, Mapping):
            total += sum(size(k) + size(v) for k, v in item.items())
        elif isinstance(item, (tuple, list)):
            total += sum(map(size, item))
        elif hasattr(item, "text") and hasattr(item, "data"):
            total += size(item.text) + size(item.data)
        return total

    return size(value)


class PreparedFrameCache:
    """Bounded LRU of immutable, fully rendered CPU frames.

    The complete encoded input and rendering metadata form the key; no digest
    or caller-supplied frame ID substitutes for image content. Cached tensors
    are owned by the cache and read-only. QA's engine_prompt concatenates them
    into fresh, request-owned tensors before the backend sees them.
    """

    ENTRY_OVERHEAD = 512

    def __init__(self, max_bytes=0):
        if max_bytes < 0:
            raise ValueError("prepared frame cache size must be nonnegative")
        self.max_bytes = int(max_bytes)
        self._entries, self._pending = OrderedDict(), {}
        self._bytes = 0
        self._lock = threading.Lock()
        self._hits = self._misses = self._evictions = 0
        self._oversized = self._coalesced = 0

    def get_or_prepare(self, key, prepare):
        if not self.max_bytes:
            return prepare()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                self._hits += 1
                return entry[1]
            pending = self._pending.get(key)
            owner = pending is None
            if owner:
                pending = self._pending[key] = Future()
                self._misses += 1
            else:
                self._coalesced += 1
        if not owner:
            return pending.result()
        try:
            frame = prepare()
            cost = _resident_bytes((key, frame)) + self.ENTRY_OVERHEAD
            with self._lock:
                if cost > self.max_bytes:
                    self._oversized += 1
                else:
                    while self._entries and self._bytes + cost > self.max_bytes:
                        _, (old_cost, _) = self._entries.popitem(last=False)
                        self._bytes -= old_cost
                        self._evictions += 1
                    self._entries[key] = (cost, frame)
                    self._bytes += cost
                del self._pending[key]
            pending.set_result(frame)
            return frame
        except BaseException as error:
            with self._lock:
                self._pending.pop(key, None)
            pending.set_exception(error)
            raise

    def stats(self):
        with self._lock:
            return dict(max_bytes=self.max_bytes, retained_bytes=self._bytes,
                        entries=len(self._entries), hits=self._hits,
                        misses=self._misses, coalesced=self._coalesced,
                        evictions=self._evictions,
                        oversized_bypasses=self._oversized)
