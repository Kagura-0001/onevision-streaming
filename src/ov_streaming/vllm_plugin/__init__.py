"""vLLM general plugin for LLaVA-OneVision-2.5/2.1 streaming inference.

Registered via the ``vllm.general_plugins`` entry point so vLLM loads it in
Process 0 (AsyncLLM frontend), the EngineCore process, and every Worker process.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

SUPPORTED_VLLM_VERSION = "0.25.1"

_REGISTERED = False


def _is_supported_vllm_version(version_str: str) -> bool:
    if version_str == SUPPORTED_VLLM_VERSION:
        return True
    if version_str.startswith(f"{SUPPORTED_VLLM_VERSION}+"):
        return True
    if version_str.startswith(f"{SUPPORTED_VLLM_VERSION}."):
        return True
    return False


def _has_native_branching() -> bool:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.async_llm import AsyncLLM

    return hasattr(AsyncLLM, "add_branch_request") and hasattr(
        Scheduler, "_prepare_branch_request"
    )


def register() -> None:
    """Register OV2.5/OV2.1 models and Fork-and-Commit branching on vLLM 0.25.1."""
    global _REGISTERED
    if _REGISTERED:
        return

    import vllm

    from .branch_patch import apply_branch_patches
    from .ov25_model import register_ov25_models

    # Always register OV2.5 / OV2.1 Qwen3.5 model & config classes so both
    # unmodified vLLM 0.25.1 and existing forks can load OV2.5 HF repos directly.
    register_ov25_models()

    if _has_native_branching():
        logger.debug(
            "vLLM already provides native streaming branching; skipping branch patch."
        )
        _REGISTERED = True
        return

    vllm_version = getattr(vllm, "__version__", "")
    allow_unverified = os.environ.get("OV_STREAMING_ALLOW_UNVERIFIED_VLLM") == "1"
    if not _is_supported_vllm_version(vllm_version) and not allow_unverified:
        raise RuntimeError(
            f"ov-streaming vLLM plugin targets vllm=={SUPPORTED_VLLM_VERSION}, "
            f"but found vllm=={vllm_version!r}. Install vllm=={SUPPORTED_VLLM_VERSION} "
            "(or set OV_STREAMING_ALLOW_UNVERIFIED_VLLM=1 to bypass this check)."
        )

    apply_branch_patches()
    _REGISTERED = True
    logger.debug("Registered ov-streaming vLLM plugin for vllm==%s", vllm_version)


__all__ = ["SUPPORTED_VLLM_VERSION", "register"]
