import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image, JpegImagePlugin
from transformers import Qwen2VLImageProcessor

from ov_streaming.client import _encode_content_part
from ov_streaming.codec import CodecConfig, CodecWindow, _source_fps, stream_codec_video
from ov_streaming.renderers.base import Renderer
from ov_streaming.renderers.ov21 import OV21Renderer
from ov_streaming.serve import _decode_content_part


def renderer():
    obj = object.__new__(OV21Renderer)
    obj.patch_size, obj.merge_size, obj.max_pixels = 16, 3, 258048
    obj.image_processor = Qwen2VLImageProcessor(
        patch_size=16,
        merge_size=3,
        temporal_patch_size=1,
    )
    return obj


def canvas():
    import numpy as np

    yy, xx = np.indices((6, 6))
    # Two source-time runs; all source blocks remain spatially contiguous.
    tt = np.where(yy < 3, 0, 15)
    pos = np.stack([tt, yy, xx], axis=-1)
    return CodecWindow(
        1.0, 30.0, (Image.new("RGB", (96, 96)),) * 2, np.stack([pos] * 2), 16, "rdo"
    )


def test_codec_http_pixels_positions_and_canvas_span():
    import numpy as np

    window = canvas()
    pixels = np.random.default_rng(123).integers(0, 256, (96, 96, 3), dtype=np.uint8)
    window.images[0].paste(Image.fromarray(pixels))
    encoded = _encode_content_part(window.content()[0])
    decoded = _decode_content_part(encoded)
    reference = io.BytesIO()
    window.images[0].save(
        reference,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=False,
        progressive=False,
    )
    expected = Image.open(io.BytesIO(reference.getvalue()))
    assert JpegImagePlugin.get_sampling(expected) == 0
    assert np.array_equal(np.asarray(decoded["image_pil"]), np.asarray(expected))
    frame = renderer().render_codec_frame(
        decoded["image_pil"],
        decoded["timestamp_sec"],
        codec=decoded["codec"],
        fps=decoded["fps"],
    )
    assert frame.data["image_grid_thw"].tolist() == [[1, 6, 6]]
    assert frame.data["pixel_values"].shape[0] == 36
    assert frame.data["patch_positions"].shape == (36, 3)
    assert frame.text.count("<|image_pad|>") == 4
    assert frame.text == (
        "<0.0 to 0.5 seconds><|vision_start|>"
        + "<|image_pad|>" * 4
        + "<|vision_end|>\n"
    )
    # First merge block: raster source coordinates, no dense regenerated t.
    assert frame.data["patch_positions"][:9].tolist() == [
        [0, y, x] for y in range(3) for x in range(3)
    ]


def test_source_fps_matches_production_clock():
    probe = {
        "streams": [
            {
                "codec_name": "h264",
                "r_frame_rate": "60000/1001",
                "avg_frame_rate": "59940/1000",
            }
        ]
    }
    with patch(
        "ov_streaming.codec.subprocess.check_output", return_value=json.dumps(probe)
    ):
        assert _source_fps("unused") == 59.94006


def test_point_canvas_keeps_one_point_label():
    window = canvas()
    window.positions[..., 0] = 3
    part = window.content()[0]
    frame = renderer().render_codec_frame(
        window.images[0], 1.0, codec=part["codec"], fps=30.0
    )
    assert frame.text.startswith("<0.1 seconds><|vision_start|>")
    assert frame.text.count("<|vision_start|>") == 1


def test_different_frames_keep_range_when_rounded_times_coincide():
    window = canvas()
    window.positions[..., 0] = window.positions[..., 0] // 15
    part = window.content()[0]
    frame = renderer().render_codec_frame(
        window.images[0], 1.0, codec=part["codec"], fps=30.0
    )
    assert frame.text.startswith("<0.0 to 0.0 seconds>")


def test_source_coordinates_survive_canvas_repacking():

    window = canvas()
    source = window.positions[0].copy()
    window.positions[0, :3, :3] = source[3:, 3:]
    part = window.content()[0]
    frame = renderer().render_codec_frame(
        window.images[0], 1.0, codec=part["codec"], fps=30.0
    )
    assert frame.data["patch_positions"][:9].tolist() == [
        [15, y, x] for y in range(3, 6) for x in range(3, 6)
    ]


def test_sampling_uses_inclusive_predecessors_at_max_seconds():
    seen = []

    def read(video, callback, **kwargs):
        assert kwargs["sample_fps"] == 240.0
        assert kwargs["max_seconds"] > 1.0
        for i in range(33):
            if not callback({"frame_idx": i, "source_time_s": i / 30}):
                break
        return {"last_decoded_time_s": i / 30}

    def prepare(records, cutoff, fps, config, deps):
        seen.append((cutoff, [r["frame_idx"] for r in records]))

    with (
        patch(
            "ov_streaming.codec._dependencies",
            return_value=(SimpleNamespace(read_video_stream=read),),
        ),
        patch("ov_streaming.codec._source_fps", return_value=30.0),
        patch("ov_streaming.codec._prepare", side_effect=prepare),
    ):
        stream_codec_video(
            "unused", lambda _: None, config=CodecConfig(max_seconds=1.0)
        )
    assert seen == [
        (
            1.0,
            [
                0,
                1,
                2,
                3,
                5,
                6,
                7,
                9,
                10,
                11,
                13,
                14,
                15,
                16,
                18,
                19,
                20,
                22,
                23,
                24,
                26,
                27,
                28,
                30,
            ],
        )
    ]


@pytest.mark.parametrize(
    "fault", ["length", "float", "future", "grid", "merge", "patch", "fps", "resize"]
)
def test_codec_rejects_incompatible_inputs(fault):
    window = canvas()
    part = window.content()[0]
    r = renderer()
    if fault == "length":
        part["codec"]["patch_positions"].pop()
    elif fault == "float":
        part["codec"]["patch_positions"][0][0] = 0.5
    elif fault == "future":
        part["codec"]["patch_positions"][0][0] = 31
    elif fault == "grid":
        part["codec"]["patch_positions"][0][1] = 6
    elif fault == "merge":
        part["codec"]["patch_positions"][0][0] = 1
    elif fault == "patch":
        part["codec"]["patch_size"] = 14
    elif fault == "fps":
        part["fps"] = float("nan")
    else:
        r.max_pixels = 100
    with pytest.raises(ValueError):
        r.render_codec_frame(
            window.images[0], 1.0, codec=part["codec"], fps=part["fps"]
        )


def test_other_renderer_rejects_codec():
    with pytest.raises(ValueError, match="ov21"):
        Renderer.render_codec_frame(object(), None, 1.0, codec={})


def test_causal_window_cutoff_endpoint_and_eof():
    records = [{"frame_idx": i, "source_time_s": i / 4} for i in range(10)]
    seen = []

    def read(video, callback, **kwargs):
        for record in records:
            if not callback(record):
                break
        return {"frames": len(records)}

    def prepare(values, cutoff, fps, config, dependencies):
        seen.append((cutoff, [r["source_time_s"] for r in values]))
        return seen[-1]

    with (
        patch(
            "ov_streaming.codec._dependencies",
            return_value=(SimpleNamespace(read_video_stream=read),),
        ),
        patch("ov_streaming.codec._source_fps", return_value=4.0),
        patch("ov_streaming.codec._prepare", side_effect=prepare),
    ):
        stats = stream_codec_video("unused", lambda x: None)
    assert stats["windows"] == 3
    assert seen == [
        (1.0, [0.0, 0.25, 0.5, 0.75, 1.0]),
        (2.0, [1.0, 1.25, 1.5, 1.75, 2.0]),
        (2.5, [2.0, 2.25]),
    ]
    assert all(max(ts) <= cutoff for cutoff, ts in seen)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": "other"},
        {"width": 640},
        {"max_candidates": 0},
        {"candidate_fps": float("nan")},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        CodecConfig(**kwargs)


def test_live_pipe_uses_declared_fps_without_consuming_a_probe():
    seen = []

    def read(video, callback, **kwargs):
        assert video == "pipe:0"
        for i in range(5):
            callback({"frame_idx": i, "source_time_s": i / 4})
        return {"last_decoded_time_s": 1.0}

    with (
        patch(
            "ov_streaming.codec._dependencies",
            return_value=(SimpleNamespace(read_video_stream=read),),
        ),
        patch(
            "ov_streaming.codec._source_fps",
            side_effect=AssertionError("consumes pipe"),
        ),
        patch(
            "ov_streaming.codec._prepare", side_effect=lambda rows, t, fps, *_: (t, fps)
        ),
    ):
        stats = stream_codec_video("pipe:0", seen.append, source_fps=4)
    assert stats["windows"] == 1
    assert seen == [(1.0, 4)]


def test_packed_qa_dispatches_codec_parts_to_codec_renderer():
    import base64

    from ov_streaming import qa

    window = canvas()
    part = window.content()[0]
    jpeg = part["image_url"]["url"]
    wire = {
        "type": "image",
        "data": base64.b64encode(jpeg).decode(),
        "timestamp": part["timestamp_sec"],
        "fps": part["fps"],
        "codec": part["codec"],
    }
    prompt, frames = qa.render_packed(
        renderer_with_prompt(),
        [{"role": "user", "content": [wire, {"type": "text", "text": "Q?"}]}],
    )
    assert len(frames) == 1
    assert frames[0].data["patch_positions"].shape == (36, 3)
    assert "<0.0 to 0.5 seconds><|vision_start|>" in prompt
    assert prompt.endswith("Q?<|im_end|>\n<|im_start|>assistant\n")


def renderer_with_prompt():
    obj = renderer()
    obj.system_prompt = "Video assistant"
    return obj


def test_codec_frame_at_the_cutoff_is_not_future_under_rounded_fps():
    import numpy as np

    # tt0063442: r_frame_rate 13978/583 is serialized as 23.975986; frame 13978 is at 583 s.
    fps = round(13978 / 583, 6)
    yy, xx = np.indices((6, 6))
    pos = np.stack([np.full((6, 6), 13978), yy, xx], axis=-1)
    codec = {"layout": "row_major", "patch_size": 16, "patch_positions": pos.reshape(-1, 3).tolist()}
    frame = renderer().render_codec_frame(Image.new("RGB", (96, 96)), 583.0, codec=codec, fps=fps)
    assert frame.data["patch_positions"].shape == (36, 3)
    pos[..., 0] = 13979
    codec["patch_positions"] = pos.reshape(-1, 3).tolist()
    with pytest.raises(ValueError, match="future source frame"):
        renderer().render_codec_frame(Image.new("RGB", (96, 96)), 583.0, codec=codec, fps=fps)
