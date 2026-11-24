"""Shared helpers for resolving local/HF checkpoints and dynamic modules."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from transformers.dynamic_module_utils import get_class_from_dynamic_module


def is_local_checkpoint(model_ref: Any) -> bool:
    return isinstance(model_ref, (str, Path)) and Path(str(model_ref)).exists()


def read_auto_map(model_ref: str | Path, hf_config: Any = None) -> dict[str, str]:
    auto_map = getattr(hf_config, "auto_map", None)
    if isinstance(auto_map, dict) and auto_map:
        return dict(auto_map)
    if is_local_checkpoint(model_ref):
        cfg_path = Path(str(model_ref)) / "config.json"
        if cfg_path.is_file():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
                if isinstance(data.get("auto_map"), dict):
                    return dict(data["auto_map"])
            except Exception:
                pass
    return {}


def module_candidates_from_auto_map(
    model_ref: str | Path,
    *,
    auto_map_keys: Sequence[str],
    fallback_modules: Sequence[str],
    hf_config: Any = None,
) -> list[str]:
    auto_map = read_auto_map(model_ref, hf_config)
    candidates: list[str] = []
    for key in auto_map_keys:
        target = auto_map.get(key)
        if isinstance(target, str) and "." in target:
            mod = target.rsplit(".", 1)[0]
            if mod and mod not in candidates:
                candidates.append(mod)
    for mod in fallback_modules:
        if mod not in candidates:
            candidates.append(mod)
    return candidates


def load_dynamic_ref(
    model_ref: str | Path,
    refs: Sequence[str],
    *,
    revision: str | None = None,
    trust_remote_code: bool = True,
    loader: Any = None,
) -> Any:
    """Resolve the first available ``module.Symbol`` from a checkpoint."""
    resolve = loader or get_class_from_dynamic_module
    local_only = is_local_checkpoint(model_ref)
    kwargs: dict[str, Any] = {"local_files_only": local_only}
    if revision is not None:
        kwargs["revision"] = revision
    if resolve is get_class_from_dynamic_module:
        kwargs["trust_remote_code"] = trust_remote_code

    last_error: Exception | None = None
    for ref in dict.fromkeys(refs):
        try:
            return resolve(ref, str(model_ref), **kwargs)
        except ModuleNotFoundError as exc:
            # Missing third-party dependency inside an existing module (e.g. cv2)
            # should surface directly instead of being masked by fallback names.
            target_mod = ref.split(".", 1)[0]
            if exc.name and target_mod not in exc.name:
                raise
            last_error = exc
        except (AttributeError, ImportError, OSError, ValueError) as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def read_json_from_checkpoint(
    model_ref: str | Path, filename: str, *, revision: str | None = None
) -> dict[str, Any]:
    path = str(model_ref)
    local_file = os.path.join(path, filename)
    if os.path.isfile(local_file):
        with open(local_file, encoding="utf-8") as fh:
            return json.load(fh)
    if is_local_checkpoint(path):
        return {}
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(path, filename, revision=revision or "main")
    with open(downloaded, encoding="utf-8") as fh:
        return json.load(fh)


def modeling_module_candidates(model_ref: str | Path, hf_config: Any = None) -> list[str]:
    """Remote-code modeling modules of an OV2.5/OV2.1 checkpoint, most likely first."""
    if getattr(hf_config, "model_type", "") == "llava_onevision2_5":
        fallbacks = ("modeling_llava_onevision2_5", "modeling_llava_onevision2")
    else:
        fallbacks = ("modeling_llava_onevision2", "modeling_llava_onevision2_5")
    return module_candidates_from_auto_map(
        model_ref,
        auto_map_keys=("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"),
        fallback_modules=fallbacks,
        hf_config=hf_config,
    )


def load_mrope_positions_fn(model_ref: str | Path, hf_config: Any = None) -> Any:
    """The checkpoint's own ``_mrope_positions_from_patch_positions``."""
    refs = [
        f"{module}._mrope_positions_from_patch_positions"
        for module in modeling_module_candidates(model_ref, hf_config)
    ]
    return load_dynamic_ref(model_ref, refs)
