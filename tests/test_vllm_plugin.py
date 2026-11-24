from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ov_streaming.vllm_plugin import _is_supported_vllm_version
from ov_streaming.vllm_plugin.branch_patch import (
    _BRANCH_FROM_EXTRA_KEY,
    _extract_branch_from_params,
)
from ov_streaming._checkpoint import modeling_module_candidates


class VLLMPluginHelpersTest(unittest.TestCase):
    def test_supported_version_matching(self):
        self.assertTrue(_is_supported_vllm_version("0.25.1"))
        self.assertTrue(_is_supported_vllm_version("0.25.1+cu129"))
        self.assertTrue(_is_supported_vllm_version("0.25.1.post1"))
        self.assertFalse(_is_supported_vllm_version("0.24.0"))
        self.assertFalse(_is_supported_vllm_version("0.26.0"))

    def test_modeling_module_candidates(self):
        cfg = SimpleNamespace(
            model_type="llava_onevision2_5",
            auto_map={
                "AutoModelForCausalLM": "modeling_llava_onevision2_5.LlavaOnevision2_5ForConditionalGeneration"
            },
        )
        self.assertEqual(
            modeling_module_candidates("unused-remote-repo", cfg),
            ["modeling_llava_onevision2_5", "modeling_llava_onevision2"],
        )

    def test_extract_branch_from_params_peek_and_pop(self):
        params = SimpleNamespace(
            extra_args={_BRANCH_FROM_EXTRA_KEY: "src-123", "other": 1}
        )
        self.assertEqual(
            _extract_branch_from_params(params, pop=False),
            "src-123",
        )
        self.assertIn(_BRANCH_FROM_EXTRA_KEY, params.extra_args)

        self.assertEqual(
            _extract_branch_from_params(params, pop=True),
            "src-123",
        )
        self.assertEqual(params.extra_args, {"other": 1})

        only_branch = SimpleNamespace(extra_args={_BRANCH_FROM_EXTRA_KEY: "src-456"})
        self.assertEqual(_extract_branch_from_params(only_branch, pop=True), "src-456")
        self.assertIsNone(only_branch.extra_args)

    @unittest.skipIf(importlib.util.find_spec("vllm") is None, "vllm not installed")
    def test_register_is_idempotent_and_exposes_ov25_and_branching(self):
        from ov_streaming.vllm_plugin import register
        from vllm.model_executor.models.config import MODELS_CONFIG_MAP
        from vllm.model_executor.models.registry import ModelRegistry
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.engine.async_llm import AsyncLLM

        register()
        register()  # idempotent

        self.assertIn("llava_onevision2_5", _CONFIG_REGISTRY)
        self.assertIn("llava_onevision2_qwen35", _CONFIG_REGISTRY)
        self.assertIn("LlavaOnevision2_5ForConditionalGeneration", MODELS_CONFIG_MAP)
        self.assertIn(
            "LlavaOnevision2_5ForConditionalGeneration",
            ModelRegistry.get_supported_archs(),
        )
        self.assertTrue(hasattr(AsyncLLM, "add_branch_request"))
        self.assertTrue(hasattr(Scheduler, "_prepare_branch_request"))

    def test_load_ov25_processor_is_cached(self):
        from ov_streaming.vllm_plugin import ov25_model

        class DummyProcessor:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class DummyVideoProcessor:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        ov25_model._load_ov25_processor.cache_clear()
        try:
            with (
                patch.object(
                    ov25_model,
                    "load_dynamic_ref",
                    side_effect=[DummyProcessor, DummyVideoProcessor],
                ) as mock_dyn,
                patch.object(
                    ov25_model.Qwen2VLImageProcessor,
                    "from_pretrained",
                    return_value=SimpleNamespace(max_pixels=258048),
                ) as mock_img,
                patch.object(
                    ov25_model.AutoTokenizer,
                    "from_pretrained",
                    return_value="tok",
                ) as mock_tok,
                patch.object(
                    ov25_model,
                    "read_json_from_checkpoint",
                    return_value={"codec": {"strategy": "rdo"}},
                ) as mock_json,
            ):
                first = ov25_model._load_ov25_processor(
                    "unused-remote-repo",
                    None,
                    True,
                    (("AutoProcessor", "processing.Cls"),),
                    max_pixels=258048,
                    use_fast=True,
                )
                second = ov25_model._load_ov25_processor(
                    "unused-remote-repo",
                    None,
                    True,
                    (("AutoProcessor", "processing.Cls"),),
                    max_pixels=258048,
                    use_fast=True,
                )
            self.assertIs(first, second)
            self.assertEqual(mock_dyn.call_count, 2)
            self.assertEqual(mock_img.call_count, 1)
            self.assertEqual(mock_tok.call_count, 1)
            self.assertEqual(mock_json.call_count, 1)
        finally:
            ov25_model._load_ov25_processor.cache_clear()


if __name__ == "__main__":
    unittest.main()
