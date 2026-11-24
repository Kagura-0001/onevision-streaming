"""vLLM model and config registration for LLaVA-OneVision-2.5 & OV2.1 Qwen3.5."""

from __future__ import annotations

import functools
from typing import Any

import torch
from torch import nn
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLImageProcessor
from transformers.configuration_utils import PretrainedConfig

from .._checkpoint import (
    is_local_checkpoint,
    load_dynamic_ref,
    load_mrope_positions_fn,
    read_auto_map,
    read_json_from_checkpoint,
)

_MODELS_REGISTERED = False


@functools.lru_cache(maxsize=8)
def _load_ov25_processor(
    model_ref: str,
    revision: str | None,
    trust_remote_code: bool,
    auto_map_items: tuple[tuple[str, str], ...],
    **kwargs: Any,
) -> Any:
    """Assemble and cache the OV2.5/OV2.1 remote-code processor."""
    local_only = is_local_checkpoint(model_ref)
    auto_map = dict(auto_map_items)
    proc_refs = [
        *(
            [auto_map["AutoProcessor"]]
            if isinstance(auto_map.get("AutoProcessor"), str)
            else []
        ),
        "processing_llava_onevision2_5.LlavaOnevision2_5Processor",
        "processing_llava_onevision2.LlavaOnevision2Processor",
    ]
    vid_refs = [
        *(
            [auto_map["AutoVideoProcessor"]]
            if isinstance(auto_map.get("AutoVideoProcessor"), str)
            else []
        ),
        "video_processing_llava_onevision2_5.LlavaOnevision2_5VideoProcessor",
        "video_processing_llava_onevision2.LlavaOnevision2VideoProcessor",
    ]
    processor_cls = load_dynamic_ref(
        model_ref, proc_refs, revision=revision, trust_remote_code=trust_remote_code
    )
    video_processor_cls = load_dynamic_ref(
        model_ref, vid_refs, revision=revision, trust_remote_code=trust_remote_code
    )
    image_processor = Qwen2VLImageProcessor.from_pretrained(
        model_ref, revision=revision, local_files_only=local_only, **kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        revision=revision,
        trust_remote_code=trust_remote_code,
        local_files_only=local_only,
        **kwargs,
    )
    video_processor = video_processor_cls(
        image_processor=image_processor,
        min_pixels=getattr(image_processor, "min_pixels", 256 * 28 * 28),
        max_pixels=getattr(image_processor, "max_pixels", 1605632),
        patch_size=getattr(image_processor, "patch_size", 14),
        spatial_merge_size=getattr(image_processor, "merge_size", 2),
    )
    codec_config = (
        read_json_from_checkpoint(
            model_ref, "preprocessor_config.json", revision=revision
        ).get("codec")
        or {}
    )
    return processor_cls(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        codec_config=codec_config,
    )


def _build_config_classes() -> tuple[type[PretrainedConfig], type[PretrainedConfig]]:
    from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig

    class LlavaOnevision2Qwen35Config(PretrainedConfig):
        model_type = "llava_onevision2_qwen35"
        sub_configs = {
            "text_config": Qwen3_5TextConfig,
            "vision_config": PretrainedConfig,
        }

        def __init__(self, text_config=None, vision_config=None, **kwargs):
            self.text_config = Qwen3_5TextConfig(**(text_config or {}))
            self.vision_config = PretrainedConfig(**(vision_config or {}))
            super().__init__(**kwargs)

    class LlavaOnevision2_5Config(LlavaOnevision2Qwen35Config):
        model_type = "llava_onevision2_5"

    return LlavaOnevision2Qwen35Config, LlavaOnevision2_5Config


def _patch_llava_onevision2_parser() -> None:
    from vllm.model_executor.models import llava_onevision2 as lov2
    from vllm.multimodal.parse import DictEmbeddingItems

    parser_cls = lov2.LlavaOnevision2MultiModalDataParser
    if getattr(parser_cls, "_ov_streaming_pixel_values_patched", False):
        return

    orig_parse = parser_cls._parse_image_data

    def _parse_image_data(self, data):
        if isinstance(data, dict):
            required_fields = (
                {"pixel_values", "image_grid_thw", "patch_positions"}
                if "pixel_values" in data
                else {"image_embeds", "image_grid_thw"}
            )
            return DictEmbeddingItems(
                data,
                modality="image",
                required_fields=required_fields,
                fields_factory=lov2._create_field_factory(self._spatial_merge_size),
            )
        return orig_parse(self, data)

    parser_cls._parse_image_data = _parse_image_data
    parser_cls._ov_streaming_pixel_values_patched = True


def _patch_incremental_mrope_flags() -> None:
    from vllm.model_executor.models.interfaces import SupportsMRoPE

    if not hasattr(SupportsMRoPE, "supports_incremental_mrope"):
        SupportsMRoPE.supports_incremental_mrope = False

    try:
        from vllm.model_executor.models.qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )

        Qwen3VLForConditionalGeneration.supports_incremental_mrope = True
    except Exception:
        pass


def _build_model_class():
    from vllm.config import VllmConfig
    from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
    from vllm.model_executor.models.llava_onevision2 import (
        LlavaOnevision2DummyInputsBuilder,
        LlavaOnevision2ForConditionalGeneration,
        LlavaOnevision2MultiModalProcessor,
        LlavaOnevision2ProcessingInfo,
        LlavaOnevision2VisionTower,
    )
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration,
    )
    from vllm.model_executor.models.utils import maybe_prefix
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.multimodal.parse import DictEmbeddingItems
    from vllm.transformers_utils.processor import _merge_mm_kwargs

    class LlavaOnevision2_5ProcessingInfo(LlavaOnevision2ProcessingInfo):
        """Resolve and cache OV2.5/OV2.1 processor classes from ``auto_map``."""

        def get_hf_processor(self, **kwargs: object):
            model_cfg = self.ctx.model_config
            merged = _merge_mm_kwargs(model_cfg, AutoProcessor, **kwargs)
            merged.setdefault("use_fast", True)
            auto_map = read_auto_map(str(model_cfg.model), self.get_hf_config())
            auto_map_items = tuple(
                sorted((k, v) for k, v in auto_map.items() if isinstance(v, str))
            )
            return _load_ov25_processor(
                str(model_cfg.model),
                model_cfg.revision,
                bool(model_cfg.trust_remote_code),
                auto_map_items,
                **merged,
            )

    class LlavaOnevision2Qwen35MultiModalProcessor(LlavaOnevision2MultiModalProcessor):
        def _maybe_apply_prompt_updates(
            self,
            mm_items,
            prompt_ids,
            mm_kwargs,
            mm_prompt_updates,
            is_update_applied,
        ):
            # Token prompts may already expand each image placeholder.
            images = mm_items.get("image")
            if isinstance(images, DictEmbeddingItems):
                config = self.info.get_hf_config()
                merge_size = config.vision_config.spatial_merge_size
                grid = images.data["image_grid_thw"]
                if prompt_ids.count(config.image_token_id) == (
                    int(grid.prod(-1).sum()) // merge_size**2
                ):
                    is_update_applied = True
            return super()._maybe_apply_prompt_updates(
                mm_items, prompt_ids, mm_kwargs, mm_prompt_updates, is_update_applied
            )

    @MULTIMODAL_REGISTRY.register_processor(
        LlavaOnevision2Qwen35MultiModalProcessor,
        info=LlavaOnevision2_5ProcessingInfo,
        dummy_inputs=LlavaOnevision2DummyInputsBuilder,
    )
    class LlavaOnevision2_5ForConditionalGeneration(
        LlavaOnevision2ForConditionalGeneration, IsHybrid, SupportsMRoPE
    ):
        packed_modules_mapping = Qwen3_5ForConditionalGeneration.packed_modules_mapping

        def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
            nn.Module.__init__(self)
            config = vllm_config.model_config.hf_config
            self.config = config
            self.multimodal_config = vllm_config.model_config.multimodal_config

            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = LlavaOnevision2VisionTower(
                    config.vision_config,
                    text_hidden_size=config.text_config.hidden_size,
                    norm_eps=getattr(config.vision_config, "layer_norm_eps", 1e-6),
                    quant_config=None,
                    prefix=maybe_prefix(prefix, "visual"),
                )
            with self._mark_language_model(vllm_config):
                self.language_model = Qwen3_5ForCausalLM(
                    vllm_config=vllm_config,
                    prefix=maybe_prefix(prefix, "language_model"),
                )
            self.make_empty_intermediate_tensors = (
                self.language_model.make_empty_intermediate_tensors
            )

            # Reuse the checkpoint's SFT position convention, including the text
            # cursor after each visual block.
            self._get_mrope_positions = load_mrope_positions_fn(
                str(vllm_config.model_config.model),
                config,
            )

        def get_mrope_input_positions(
            self,
            input_tokens: list[int],
            mm_features: list[MultiModalFeatureSpec],
        ) -> tuple[torch.Tensor, int]:
            patch_positions = [
                f.data[
                    "patch_positions_videos"
                    if f.modality == "video"
                    else "patch_positions"
                ].data
                for f in sorted(mm_features, key=lambda f: f.mm_position.offset)
            ]
            if not patch_positions:
                return torch.arange(len(input_tokens)).expand(3, -1), 0
            positions, delta = self._get_mrope_positions(
                torch.tensor([input_tokens]),
                None,
                torch.cat(patch_positions),
                self.config.vision_config.spatial_merge_size,
                self.config.image_token_id,
                self.config.video_token_id,
            )
            return positions[:, 0], int(delta.item())

        @classmethod
        def get_mamba_state_dtype_from_config(cls, vllm_config):
            return Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(
                vllm_config
            )

        @classmethod
        def get_mamba_state_shape_from_config(cls, vllm_config):
            return Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(
                vllm_config
            )

        @classmethod
        def get_mamba_state_copy_func(cls):
            return Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()

    return LlavaOnevision2_5ForConditionalGeneration


def register_ov25_models() -> None:
    """Register OV2.5 and OV2.1-Qwen3.5 configs and model architectures in vLLM."""
    global _MODELS_REGISTERED
    if _MODELS_REGISTERED:
        return

    from vllm.model_executor.models.config import (
        MODELS_CONFIG_MAP,
        Qwen3_5ForConditionalGenerationConfig,
    )
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    _patch_incremental_mrope_flags()
    _patch_llava_onevision2_parser()

    ov21_cfg, ov25_cfg = _build_config_classes()
    _CONFIG_REGISTRY["llava_onevision2_qwen35"] = ov21_cfg
    _CONFIG_REGISTRY["llava_onevision2_5"] = ov25_cfg

    model_cls = _build_model_class()
    for arch in (
        "LlavaOnevision2_5ForConditionalGeneration",
        "LlavaOnevision2Qwen35ForConditionalGeneration",
    ):
        MODELS_CONFIG_MAP[arch] = Qwen3_5ForConditionalGenerationConfig
        ModelRegistry.register_model(arch, model_cls)

    _MODELS_REGISTERED = True
