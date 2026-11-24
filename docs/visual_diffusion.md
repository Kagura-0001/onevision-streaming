# Optional processed-pixel diffusion for packed QA

`POST /v1/qa` accepts an optional `visual_diffusion` object:

```json
{"noise_step": 500, "seed": 42}
```

Omission or null retains the previous prompt, pixels, sampling and response.
Both keys are required when enabled; no extra keys are accepted. `noise_step`
is an integer in [0, 999], and `seed` is an integer in [0, 2^63). Booleans,
fractional values and string coercion are rejected. Invalid inputs return 422.
The fixed evaluation condition uses step 500, not a step search.

The operation occurs after the existing renderer/image processor and
`engine_prompt` concatenate frame tensors. Only
`multi_modal_data.image.pixel_values` is replaced with a new CPU tensor.
Prompt token IDs, image grid, patch positions, frame order, source pixels and
cached decoded RGB images are unchanged. The clean request has no extra
metadata or tensor hashing. There is no control prefill, token bias, output
restriction, PNG noise, clamp, diffusion model, or vLLM fork change.

The formula comes from DAMO-NLP-SG/VCD commit
`d6568ff81b8fd306a49e630df44f2db5c2300191`,
`vcd_utils/vcd_add_noise.py` (SHA256
`adb28efe0f89874148f4cf02761bf53b447baf101e75ebde1ca5d54802adef33`):

- `beta = sigmoid(linspace(-6, 6, 1000)) * (0.005 - 0.00001) + 0.00001`;
- `alpha = cumprod(1 - beta)[noise_step]`;
- `noisy = sqrt(alpha) * clean + sqrt(1 - alpha) * epsilon`.

The schedule is explicitly CPU float32. Epsilon is a single normal draw in
input tensor dtype/shape using a private CPU `torch.Generator`; global RNG
state is untouched. No clean tensor is modified in place. Nonempty finite
floating CPU pixels are required; missing/unsupported/nonfinite pixels fail
explicitly, with no clean fallback. Noise is applied to the normalized OV
patch layout, which differs from the reference image-layout draw order. This
is a local video/packed adaptation, not an exact reproduction of VCD sampling.
The caller determines the seed. The fixed evaluation uses the first eight
big-endian bytes of SHA256(UTF8(`42:` + public_episode_uid)), modulo 2^63,
instead of the reference global sequential seed stream.

Only noisy responses include `visual_diffusion`, with these exact keys:
`noise_step`, `seed`, `dtype` (for example `torch.float32`), `shape` (integer
list), `clean_tensor_sha256`, and `noisy_tensor_sha256`. Hashes cover raw bytes
of the corresponding contiguous CPU tensors; dtype and shape are recorded
separately. Existing `first_token_logprobs`, text, finish reason and usage keep
their original meanings. The forced input perturbation adds no generated token.
This API only exposes the noisy branch; contrastive arithmetic and scoring
remain the evaluation caller's responsibility.

The current image-content hashing in vLLM distinguishes clean/noisy tensors;
this feature does not provide multimodal UUID overrides or alter prefix-cache
configuration. Service manifests must identify the actual deployed revision,
independent of the canonical source checkout.
