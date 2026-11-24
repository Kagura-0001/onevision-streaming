"""Opt-in diffusion noise on already processed packed-QA pixel tensors."""

import hashlib
from collections.abc import Mapping


def validate_visual_diffusion(value):
    """Keep absence/null disabled; require explicit, bounded integer controls."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"noise_step", "seed"}:
        raise ValueError("visual_diffusion requires exactly noise_step and seed")
    step, seed = value["noise_step"], value["seed"]
    if type(step) is not int or not 0 <= step < 1000:
        raise ValueError("visual_diffusion.noise_step must be an integer in [0, 999]")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("visual_diffusion.seed must be an integer in [0, 2**63)")
    return {"noise_step": step, "seed": seed}


def _tensor_sha256(tensor):
    import torch

    raw = tensor.detach().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def apply_visual_diffusion(payload, settings):
    """Copy only the payload containers/pixel tensor; never alter clean caches.

    The disabled path returns the original object without importing torch,
    inspecting pixels, hashing, or drawing any random values.
    """
    settings = validate_visual_diffusion(settings)
    if settings is None:
        return payload, None

    import torch

    modalities = payload.get("multi_modal_data")
    images = modalities.get("image") if isinstance(modalities, Mapping) else None
    pixels = images.get("pixel_values") if isinstance(images, Mapping) else None
    if not isinstance(pixels, torch.Tensor):
        raise ValueError("visual_diffusion requires processed image.pixel_values")
    if pixels.device.type != "cpu" or not pixels.is_floating_point():
        raise ValueError("visual_diffusion requires a floating-point CPU pixel tensor")
    if not pixels.numel() or not torch.isfinite(pixels).all().item():
        raise ValueError("visual_diffusion requires nonempty finite pixels")

    # DAMO-NLP-SG/VCD@d6568ff81b8fd306a49e630df44f2db5c2300191,
    # vcd_utils/vcd_add_noise.py. Schedule stays float32 regardless of input dtype.
    # A private CPU generator replaces the reference's global sequential RNG.
    with torch.no_grad():
        betas = torch.linspace(-6, 6, 1000, dtype=torch.float32, device="cpu")
        betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
        alpha = torch.cumprod(1 - betas, dim=0)[settings["noise_step"]]
        generator = torch.Generator(device="cpu").manual_seed(settings["seed"])
        noise = torch.randn(
            pixels.shape, dtype=pixels.dtype, device="cpu", generator=generator
        )
        noisy = torch.sqrt(alpha) * pixels + torch.sqrt(1 - alpha) * noise
    if not torch.isfinite(noisy).all().item():
        raise ValueError("visual_diffusion produced nonfinite pixels")
    metadata = {
        **settings,
        "dtype": str(pixels.dtype),
        "shape": list(pixels.shape),
        "clean_tensor_sha256": _tensor_sha256(pixels),
        "noisy_tensor_sha256": _tensor_sha256(noisy),
    }
    replaced = {
        **payload,
        "multi_modal_data": {
            **modalities,
            "image": {**images, "pixel_values": noisy},
        },
    }
    return replaced, metadata
