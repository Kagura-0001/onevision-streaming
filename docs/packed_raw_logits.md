# Selected next-token raw logits for packed QA

`python -m ov_streaming.qa` defaults to `--score-output-mode raw_logprobs`.
The existing `logprob_token_ids` / `first_token_logprobs` interface and ordinary
requests keep their existing behavior. To start a separate raw-logits endpoint,
use `--score-output-mode raw_logits`. This is an engine-wide vLLM setting, passed
as `AsyncEngineArgs(logprobs_mode="raw_logits")`; it is not a per-request switch.
The vLLM model, kernel, rendering, preprocessing, and generation path are unchanged.

On that endpoint, a request may include:

```json
{"messages": [], "max_tokens": 1, "logits_token_ids": [32, 33, 34, 35]}
```

`messages` must contain the ordinary packed input; the empty example above only
illustrates the score fields. The response adds:

```json
{"first_token_logits": {"32": 18.5, "33": -3.25, "34": 0.0, "35": 2.75},
 "value_semantics": "pre_sampling_logits"}
```

These are the requested token IDs' **first next-token raw logits before sampling
processors**, not log-probabilities or probabilities. No softmax, candidate-only
normalization, bias, allowed-token mask, or zero filling is applied. The generated
token may be outside the requested set. The existing vLLM selected-token channel
(`SamplingParams.logprob_token_ids`) and `Logprob.logprob` carrier transport the
raw floats when the engine runs in `raw_logits` mode. The external response uses
only the logits field and does not label these floats as log-probabilities.
The implementation does not expose the carrier's cumulative-logprob quantity.

The raw endpoint rejects the presence of `logprob_token_ids`, including null,
with HTTP 422. The default endpoint rejects the presence of `logits_token_ids`,
including null, with HTTP 422. Supplying both fields is also rejected. These
checks run before rendering or generation. `logits_token_ids`, when present,
must be a list of 1–128 unique, nonnegative integer IDs; booleans, coercion and
null are rejected. The legacy default `logprob_token_ids: null` behavior remains
unchanged. Missing requested output scores, invalid numbers and nonfinite floats
produce HTTP 502, never a partial score map or fallback. An out-of-vocabulary ID
is subject to the existing engine validation.

Omit both score fields for ordinary generation on either service: sampling
parameters and response fields remain the same. No score values or semantics
metadata are then added. `/v1/models` adds
`data[0].score_output_mode = "raw_logits"` only on the new raw endpoint; the default
models response is unchanged. Service manifests should record the actual source
revision, this CLI option, and the deployed vLLM revision separately.

The supported use is the existing V1, non-speculative packed path. For single
next-token scoring set `max_tokens=1`. If more output tokens are requested, this
API still reports position zero only; it is not a continuation-likelihood API.
Comparing raw logits across different inputs and projecting a method onto a
candidate subset are evaluation-method choices outside this serving interface.

This interface uses the existing vLLM `ModelConfig.logprobs_mode`, bound in
`v1/worker/gpu_model_runner.py`, and the raw branch in `v1/sample/sampler.py`.
No vLLM source changes are needed. CPU tests exercise the actual installed
`AsyncEngineArgs` and `SamplingParams`, mocked engine outputs, channel isolation,
default responses, and failure paths; they do not load a model or establish GPU
numerical parity.
