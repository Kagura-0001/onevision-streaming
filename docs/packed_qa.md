# Packed QA Caches

`python -m ov_streaming.qa` provides two optional, opt-in LRU caches for packed QA services. Both default to `0` (disabled).

## Decoded image cache

Enable with `--decoded-image-cache-mib 2048`. Each service retains at most the configured MiB of immutable RGB pixels plus a conservative 512-byte allowance per entry, evicting least recently used entries.

This is not a model, prompt, or tensor cache: cropping, padding, timestamps, frame positions, and image preprocessing still run for every request. Entries are keyed by a hash of the complete base64 image content; a hit returns an independent RGB image. Only RGB pixels and dimensions are retained, never image metadata.

A lock serializes cache misses and oversized bypasses so concurrent requests cannot trigger duplicate decodes for the same image. Images larger than the cache budget remain supported without insertion. The limit bounds retained cache storage, not incoming HTTP bodies, the in-flight decode, returned images, or renderer tensors; existing preprocessing concurrency limits still apply.

## Prepared CPU frame cache

Enable with `--prepared-frame-cache-mib 2048` (usable with or without the decoded image cache). It additionally retains deterministic rendered CPU frames, including normalized image patches and frame positions.

Entries are keyed by the complete encoded image, source crop dimensions, timestamp/index/fps, and renderer identity; per-request timestamp-label overrides still run after lookup. No KV state, visual encoder output, or model response is cached.

The LRU budget accounts for full input strings, Python containers, and CPU tensor backing storage. Different frames prepare concurrently while simultaneous requests for the same frame share one preparation. Cached tensors are read-only; the QA backend receives fresh concatenated tensors. As with the decoded cache, the budget excludes incoming HTTP bodies and tensors owned by active requests.
