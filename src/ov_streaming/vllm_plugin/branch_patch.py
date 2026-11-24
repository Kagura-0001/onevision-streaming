"""Runtime patches that add Fork-and-Commit streaming branching to vLLM 0.25.1.

These patches add pure-Python streaming branch support across vLLM's three
process roles:
- Frontend (Process 0): ``AsyncLLM``, ``AsyncMPClient``, ``OutputProcessor``, ``RequestState``
- EngineCore: ``EngineCore``, ``EngineCoreProc``, ``Request``, ``Scheduler``,
  ``KVCacheManager``, ``KVCacheCoordinator``
- Worker: ``GPUModelRunner``, ``InputBatch``

To avoid mutating the ``msgspec.Struct`` layout of ``EngineCoreRequest``, the
branch parent ID is transported inside ``SamplingParams.extra_args`` under
``_BRANCH_FROM_EXTRA_KEY`` and popped when constructing ``Request`` in EngineCore.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

import torch

logger = logging.getLogger(__name__)

_BRANCH_FROM_EXTRA_KEY = "_ov_branch_from_request_id"
_PATCHED = False


def _extract_branch_from_params(sampling_params: Any, *, pop: bool) -> str | None:
    if sampling_params is None:
        return None
    extra_args = getattr(sampling_params, "extra_args", None)
    if not isinstance(extra_args, dict) or _BRANCH_FROM_EXTRA_KEY not in extra_args:
        return None
    if pop:
        value = extra_args.pop(_BRANCH_FROM_EXTRA_KEY, None)
        if not extra_args:
            sampling_params.extra_args = None
        return value
    return extra_args.get(_BRANCH_FROM_EXTRA_KEY)


def _patch_request() -> None:
    from vllm.v1.request import Request

    if getattr(Request, "_ov_streaming_branch_patched", False):
        return

    orig_init = Request.__init__
    orig_from_core = Request.from_engine_core_request.__func__

    def __init__(self, *args, branch_from_request_id: str | None = None, **kwargs):
        orig_init(self, *args, **kwargs)
        self.branch_from_request_id = branch_from_request_id

    @classmethod
    def from_engine_core_request(cls, request, *args, **kwargs):
        branch_from = getattr(request, "branch_from_request_id", None)
        if branch_from is None:
            branch_from = _extract_branch_from_params(
                getattr(request, "sampling_params", None), pop=True
            )
        req = orig_from_core(cls, request, *args, **kwargs)
        req.branch_from_request_id = branch_from
        return req

    Request.__init__ = __init__
    Request.from_engine_core_request = from_engine_core_request
    Request._ov_streaming_branch_patched = True


def _patch_output_processor() -> None:
    from vllm.v1.engine.output_processor import OutputProcessor, RequestState

    if not getattr(RequestState, "_ov_streaming_branch_patched", False):
        orig_apply = RequestState.apply_streaming_update

        def apply_streaming_update(self, update):
            # Clone list before upstream extend() so caller's prompt list is untouched.
            if self.prompt_token_ids is not None:
                self.prompt_token_ids = list(self.prompt_token_ids)
            return orig_apply(self, update)

        RequestState.apply_streaming_update = apply_streaming_update
        RequestState._ov_streaming_branch_patched = True

    if not hasattr(OutputProcessor, "get_request_state"):

        def get_request_state(self, external_request_id: str) -> RequestState:
            """Resolve one active external request ID to its frontend state."""
            internal_ids = self.external_req_ids.get(external_request_id, [])
            if not internal_ids:
                raise ValueError(f"request {external_request_id!r} is not active")
            if len(internal_ids) != 1:
                raise ValueError(
                    "request branching requires exactly one source sequence"
                )
            return self.request_states[internal_ids[0]]

        OutputProcessor.get_request_state = get_request_state


def _patch_core_client() -> None:
    from vllm.v1.engine.core_client import EngineCoreClient

    if "get_branch_snapshot_async" not in EngineCoreClient.__dict__:

        async def get_branch_snapshot_async(
            self, source_request_id: str
        ) -> tuple[list[int], str | None]:
            if hasattr(self, "call_utility_async"):
                return await self.call_utility_async(
                    "get_branch_snapshot", source_request_id
                )
            if hasattr(self, "engine_core"):
                return self.engine_core.get_branch_snapshot(source_request_id)
            raise NotImplementedError

        EngineCoreClient.get_branch_snapshot_async = get_branch_snapshot_async


def _patch_async_llm() -> None:
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.exceptions import EngineDeadError
    from vllm.v1.engine.output_processor import RequestOutputCollector

    if hasattr(AsyncLLM, "add_branch_request"):
        return

    async def add_branch_request(
        self,
        source_request_id: str,
        request_id: str,
        sampling_params: SamplingParams,
        *,
        suffix_token_ids: Sequence[int] = (),
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
    ) -> RequestOutputCollector:
        """Create a request from a paused streaming request's KV state."""
        if self.errored:
            raise EngineDeadError()
        if sampling_params.n != 1:
            raise ValueError("request branching requires sampling_params.n == 1")
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            raise NotImplementedError(
                "request branching currently requires data_parallel_size == 1"
            )
        source_state = self.output_processor.get_request_state(source_request_id)
        source_internal_id = source_state.request_id
        prompt_token_ids, cache_salt = await self.engine_core.get_branch_snapshot_async(
            source_internal_id
        )
        prompt = TokensPrompt(
            prompt_token_ids=[*prompt_token_ids, *suffix_token_ids],
            cache_salt=cache_salt,
        )
        supported_tasks = await self.get_supported_tasks()
        # Check after the last await; _add_request registers synchronously.
        if request_id in self.output_processor.external_req_ids:
            raise ValueError(f"request {request_id!r} is already active")

        branch_params = sampling_params.clone()
        branch_params.extra_args = {
            **(branch_params.extra_args or {}),
            _BRANCH_FROM_EXTRA_KEY: source_internal_id,
        }
        request = self.input_processor.process_inputs(
            request_id,
            prompt,
            branch_params,
            supported_tasks=supported_tasks,
            lora_request=source_state.lora_request,
            trace_headers=trace_headers,
            priority=priority,
        )
        self.input_processor.assign_request_id(request)

        self._run_output_handler()
        queue = RequestOutputCollector(request.params.output_kind, request.request_id)
        await self._add_request(request, None, None, 0, queue)
        return queue

    AsyncLLM.add_branch_request = add_branch_request


def _patch_engine_core() -> None:
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.core import EngineCore, EngineCoreProc
    from vllm.v1.request import RequestStatus

    if not hasattr(EngineCore, "get_branch_snapshot"):

        def get_branch_snapshot(
            self, source_request_id: str
        ) -> tuple[list[int], str | None]:
            """Return the computed prefix of a paused streaming request."""
            source = self.scheduler.requests.get(source_request_id)
            if source is None:
                raise ValueError(f"branch source {source_request_id!r} does not exist")
            if (
                not source.resumable
                or source.status != RequestStatus.WAITING_FOR_STREAMING_REQ
            ):
                raise RuntimeError(
                    "branch source must be paused between streaming inputs"
                )
            if source.prompt_embeds is not None or source.prompt_token_ids is None:
                raise NotImplementedError("request branching requires token inputs")
            if source.num_computed_tokens <= 0:
                raise RuntimeError("branch source has no computed prefix")

            return (
                source.all_token_ids[: source.num_computed_tokens],
                source.cache_salt,
            )

        EngineCore.get_branch_snapshot = get_branch_snapshot

    if not getattr(EngineCoreProc, "_ov_streaming_branch_patched", False):
        orig_handle = EngineCoreProc._handle_client_request

        def _handle_client_request(self, request_type, request):
            if request_type == EngineCoreRequestType.ADD:
                req, request_wave = request
                if self._reject_add_in_shutdown(req):
                    return
                is_branch = (
                    getattr(req, "branch_from_request_id", None) is not None
                    or _extract_branch_from_params(
                        getattr(req, "sampling_params", None), pop=False
                    )
                    is not None
                )
                try:
                    self.add_request(req, request_wave)
                except (ValueError, RuntimeError):
                    if not is_branch:
                        raise
                    logger.exception("Error handling branch request %s", req.request_id)
                    self._send_error_outputs_to_client(
                        [req.request_id], req.client_index
                    )
                return
            return orig_handle(self, request_type, request)

        EngineCoreProc._handle_client_request = _handle_client_request
        EngineCoreProc._ov_streaming_branch_patched = True


def _patch_kv_cache() -> None:
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    if not hasattr(KVCacheCoordinator, "share_blocks"):

        def share_blocks(
            self,
            source_request_id: str,
            branch_request_id: str,
            num_shared_tokens: int,
        ) -> None:
            """Retain a source request's full blocks for a branch request."""
            if any(
                branch_request_id in manager.req_to_blocks
                or branch_request_id in manager.num_cached_block
                for manager in self.single_type_managers
            ):
                raise ValueError(
                    f"request {branch_request_id!r} already has KV blocks"
                )

            shared_blocks_by_group = []
            for manager in self.single_type_managers:
                num_blocks = num_shared_tokens // manager.block_size
                source_blocks = manager.req_to_blocks.get(source_request_id)
                if source_blocks is None or len(source_blocks) < num_blocks:
                    raise ValueError(
                        f"source {source_request_id!r} does not own {num_blocks} "
                        "shareable KV blocks"
                    )
                shared_blocks = source_blocks[:num_blocks]
                if any(block.is_null for block in shared_blocks):
                    raise NotImplementedError(
                        "request branching does not support recycled/null KV blocks"
                    )
                shared_blocks_by_group.append(shared_blocks)

            # Validate before changing refcounts to avoid a partial attachment.
            for manager, shared_blocks in zip(
                self.single_type_managers, shared_blocks_by_group, strict=True
            ):
                self.block_pool.touch(shared_blocks)
                manager.req_to_blocks[branch_request_id].extend(shared_blocks)
                manager.num_cached_block[branch_request_id] = len(shared_blocks)

        KVCacheCoordinator.share_blocks = share_blocks

    if not hasattr(KVCacheCoordinator, "fork_blocks"):

        def fork_blocks(
            self, source_request_id: str, branch_request_id: str, num_tokens: int
        ) -> list[tuple[int, int, int]]:
            """Give a branch the source's KV cache for its first num_tokens tokens."""
            forks = []
            for manager in self.single_type_managers:
                source_blocks = manager.req_to_blocks[source_request_id]
                if isinstance(manager.kv_cache_spec, MambaSpec):
                    # As after a prefix-cache hit, only the state block is kept.
                    state_idx = (num_tokens - 1) // manager.block_size
                    null_blocks = [self.block_pool.null_block] * state_idx
                    forks.append((null_blocks, source_blocks[state_idx]))
                elif isinstance(manager.kv_cache_spec, FullAttentionSpec):
                    num_blocks, num_tail_tokens = divmod(
                        num_tokens, manager.block_size
                    )
                    tail = source_blocks[num_blocks] if num_tail_tokens else None
                    forks.append((source_blocks[:num_blocks], tail))
                else:
                    raise NotImplementedError(
                        "request branching supports full attention and Mamba layers only"
                    )

            # Allocate first so running out of blocks leaves no partial fork.
            new_blocks = self.block_pool.get_new_blocks(
                sum(tail is not None for _, tail in forks)
            )
            copies: list[tuple[int, int, int]] = []
            for manager, (shared, tail) in zip(
                self.single_type_managers, forks, strict=True
            ):
                self.block_pool.touch(shared)
                branch_blocks = manager.req_to_blocks[branch_request_id]
                branch_blocks.extend(shared)
                manager.num_cached_block[branch_request_id] = (
                    num_tokens // manager.block_size
                )
                if tail is not None:
                    new_block = new_blocks.pop()
                    branch_blocks.append(new_block)
                    # Hold both ends until the step that runs the copy is processed.
                    self.block_pool.touch([tail, new_block])
                    copies.append(
                        (manager.kv_cache_group_id, tail.block_id, new_block.block_id)
                    )
            return copies

        KVCacheCoordinator.fork_blocks = fork_blocks

    if not getattr(KVCacheManager, "_ov_streaming_branch_patched", False):
        orig_mgr_init = KVCacheManager.__init__

        def __init__(self, *args, **kwargs):
            orig_mgr_init(self, *args, **kwargs)
            if not hasattr(self, "scheduler_block_size"):
                if "scheduler_block_size" in kwargs:
                    self.scheduler_block_size = kwargs["scheduler_block_size"]
                elif len(args) >= 3:
                    self.scheduler_block_size = args[2]
                else:
                    self.scheduler_block_size = self.block_size

        def share_prefix(self, source, branch) -> int:
            """Attach a branch to the source's immutable full KV blocks."""
            if self.use_eagle:
                raise NotImplementedError("request branching does not support EAGLE")
            if not all(
                isinstance(group.kv_cache_spec, FullAttentionSpec)
                for group in self.kv_cache_config.kv_cache_groups
            ):
                raise NotImplementedError(
                    "request branching currently supports full attention only"
                )

            max_shared_tokens = min(source.num_computed_tokens, branch.num_tokens - 1)
            shared_tokens = (
                max(0, max_shared_tokens) // self.scheduler_block_size
            ) * self.scheduler_block_size
            self.coordinator.share_blocks(
                source.request_id,
                branch.request_id,
                shared_tokens,
            )
            return shared_tokens

        KVCacheManager.__init__ = __init__
        KVCacheManager.share_prefix = share_prefix
        KVCacheManager._ov_streaming_branch_patched = True


def _patch_scheduler() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    if getattr(Scheduler, "_ov_streaming_branch_patched", False):
        return

    orig_init = Scheduler.__init__
    orig_schedule = Scheduler.schedule
    orig_update_session = Scheduler._update_request_as_session
    orig_update_from_output = Scheduler.update_from_output
    orig_add_request = Scheduler.add_request
    orig_free_request = Scheduler._free_request

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.pending_branch_copies: dict[str, list[tuple[int, int, int]]] = {}

    def schedule(self, *args, **kwargs):
        scheduler_output = orig_schedule(self, *args, **kwargs)
        branch_block_copies = None
        if scheduler_output.total_num_scheduled_tokens and self.pending_branch_copies:
            branch_block_copies = [
                copy_spec
                for copies in self.pending_branch_copies.values()
                for copy_spec in copies
            ]
            self.pending_branch_copies.clear()
        scheduler_output.branch_block_copies = branch_block_copies
        return scheduler_output

    def _update_request_as_session(self, session, update):
        # The discarded sampled token may already have completed a hash block.
        num_computed_tokens = session.num_computed_tokens
        hash_block_size = self.kv_cache_manager.block_pool.hash_block_size
        del session.block_hashes[num_computed_tokens // hash_block_size :]
        return orig_update_session(self, session, update)

    def update_from_output(self, scheduler_output, model_runner_output):
        self._release_branch_copy_blocks(
            getattr(scheduler_output, "branch_block_copies", None) or []
        )
        return orig_update_from_output(self, scheduler_output, model_runner_output)

    def add_request(self, request):
        if (
            request.request_id not in self.requests
            and getattr(request, "branch_from_request_id", None) is not None
        ):
            self._prepare_branch_request(request)
        return orig_add_request(self, request)

    def _prepare_branch_request(self, branch) -> None:
        """Prepare a branch using a paused source's full KV blocks."""
        source_id = branch.branch_from_request_id
        assert source_id is not None
        source = self.requests.get(source_id)
        if source is None:
            raise ValueError(f"branch source {source_id!r} does not exist")
        if (
            not source.resumable
            or source.status != RequestStatus.WAITING_FOR_STREAMING_REQ
        ):
            raise RuntimeError("branch source must be paused between streaming inputs")
        if branch.resumable:
            raise ValueError("a branch cannot be a resumable streaming request")
        if branch.prompt_embeds is not None or source.prompt_embeds is not None:
            raise NotImplementedError(
                "request branching does not support prompt_embeds"
            )
        num_prefix_tokens = source.num_computed_tokens
        if (
            branch.all_token_ids[:num_prefix_tokens]
            != source.all_token_ids[:num_prefix_tokens]
        ):
            raise RuntimeError("branch tokens do not start with the source's tokens")
        if branch.lora_request != source.lora_request:
            raise ValueError("branch and source must use the same LoRA")
        if branch.cache_salt != source.cache_salt:
            raise ValueError("branch and source must use the same cache salt")
        if self.connector is not None or self.ec_connector is not None:
            raise NotImplementedError(
                "request branching does not support KV or encoder connectors"
            )
        if self.is_encoder_decoder:
            raise NotImplementedError("request branching requires a decoder-only model")

        branch.mm_features = list(source.mm_features)
        branch.block_hashes.clear()
        branch.update_block_hashes()

        if branch.num_tokens == num_prefix_tokens:
            shared_tokens = self.kv_cache_manager.share_prefix(source, branch)
        else:
            if (
                self.use_v2_model_runner
                or self.num_spec_tokens
                or self.parallel_config.pipeline_parallel_size != 1
                or self.dcp_world_size != 1
                or self.pcp_world_size != 1
            ):
                raise NotImplementedError(
                    "branching with suffix_token_ids requires the V1 runner "
                    "without speculative decoding or pipeline/context parallelism"
                )
            self.pending_branch_copies[branch.request_id] = (
                self.kv_cache_manager.coordinator.fork_blocks(
                    source.request_id, branch.request_id, num_prefix_tokens
                )
            )
            shared_tokens = num_prefix_tokens
        branch.num_computed_tokens = shared_tokens
        branch.branch_from_request_id = None
        if branch.prefill_stats is not None:
            branch.prefill_stats.set(
                num_prompt_tokens=branch.num_prompt_tokens,
                num_local_cached_tokens=shared_tokens,
                num_external_cached_tokens=0,
            )

    def _free_request(self, request, delay_free_blocks: bool = False):
        self._release_branch_copy_blocks(
            self.pending_branch_copies.pop(request.request_id, [])
        )
        return orig_free_request(self, request, delay_free_blocks=delay_free_blocks)

    def _release_branch_copy_blocks(
        self, copies: list[tuple[int, int, int]]
    ) -> None:
        if not copies:
            return
        pool = self.kv_cache_manager.block_pool
        pool.free_blocks(
            pool.blocks[block_id]
            for _, source, dest in copies
            for block_id in (source, dest)
        )

    Scheduler.__init__ = __init__
    Scheduler.schedule = schedule
    Scheduler._update_request_as_session = _update_request_as_session
    Scheduler.update_from_output = update_from_output
    Scheduler.add_request = add_request
    Scheduler._prepare_branch_request = _prepare_branch_request
    Scheduler._free_request = _free_request
    Scheduler._release_branch_copy_blocks = _release_branch_copy_blocks
    Scheduler._ov_streaming_branch_patched = True


def _patch_worker() -> None:
    from vllm.model_executor.models.interfaces import SupportsMRoPE
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if not getattr(InputBatch, "_ov_streaming_branch_patched", False):
        orig_add = InputBatch.add_request
        orig_swap = InputBatch.swap_states

        def add_request(self, request):
            result = orig_add(self, request)
            if (
                self.allowed_token_ids_mask_cpu_tensor is not None
                and request.sampling_params is not None
                and not request.sampling_params.allowed_token_ids
            ):
                req_index = self.req_id_to_index.get(request.req_id)
                if req_index is not None:
                    self.allowed_token_ids_mask_cpu_tensor[req_index].fill_(False)
            return result

        def swap_states(self, i1: int, i2: int) -> None:
            saved = None
            if self.allowed_token_ids_mask_cpu_tensor is not None:
                # Avoid row-view aliasing bug in upstream tuple swap.
                saved = self.allowed_token_ids_mask_cpu_tensor[[i2, i1]].clone()
            orig_swap(self, i1, i2)
            if saved is not None:
                self.allowed_token_ids_mask_cpu_tensor[[i1, i2]] = saved

        InputBatch.add_request = add_request
        InputBatch.swap_states = swap_states
        InputBatch._ov_streaming_branch_patched = True

    if not getattr(GPUModelRunner, "_ov_streaming_branch_patched", False):
        orig_update_states = GPUModelRunner._update_states
        orig_update_streaming = GPUModelRunner._update_streaming_request

        def _copy_branch_blocks(
            self, copies: list[tuple[int, int, int]]
        ) -> None:
            """Clone writable stream state on the same stream as the next forward."""
            context = self.compilation_config.static_forward_context
            for group in self._kv_cache_spec_attn_group_iterator():
                pairs = [
                    (s, d) for g, s, d in copies if g == group.kv_cache_group_id
                ]
                if not pairs:
                    continue
                spec = group.kv_cache_spec
                if isinstance(spec, MambaSpec):
                    for layer_name in group.layer_names:
                        for state in context[layer_name].kv_cache:
                            for source, dest in pairs:
                                state[dest].copy_(state[source])
                else:
                    assert isinstance(spec, FullAttentionSpec)
                    kernel_bs = self._kernel_block_sizes[group.kv_cache_group_id]
                    ratio = spec.block_size // kernel_bs
                    block_dim = group.backend.get_kv_cache_block_dim(
                        kernel_bs,
                        spec.num_kv_heads,
                        spec.head_size,
                        cache_dtype_str=self.cache_config.cache_dtype,
                    )
                    for layer_name in group.layer_names:
                        kv = context[layer_name].kv_cache
                        for source, dest in pairs:
                            kv.narrow(block_dim, dest * ratio, ratio).copy_(
                                kv.narrow(block_dim, source * ratio, ratio)
                            )

        def _update_states(self, scheduler_output):
            result = orig_update_states(self, scheduler_output)
            branch_copies = getattr(scheduler_output, "branch_block_copies", None)
            if branch_copies:
                self._copy_branch_blocks(branch_copies)
            return result

        def _extend_mrope_positions(self, req_state):
            mrope_model = cast(SupportsMRoPE, self.get_model())
            positions = req_state.mrope_positions
            delta = req_state.mrope_position_delta
            assert positions is not None and delta is not None
            assert req_state.prompt_token_ids is not None

            start = positions.shape[1]
            mm_features = []
            for f in req_state.mm_features:
                pos = f.mm_position
                if pos.offset >= start and f.modality != "prompt_embeds":
                    offset = pos.offset - start
                    mm_features.append(
                        replace(f, mm_position=replace(pos, offset=offset))
                    )

            new_positions, new_delta = mrope_model.get_mrope_input_positions(
                req_state.prompt_token_ids[start:], mm_features
            )
            req_state.mrope_positions = torch.cat(
                [positions, new_positions + start + delta], dim=1
            )
            req_state.mrope_position_delta = delta + new_delta

        def _update_streaming_request(self, req_id, new_req_data):
            if self.uses_mrope and getattr(
                self.get_model(), "supports_incremental_mrope", False
            ):
                orig_init_mrope = self._init_mrope_positions
                self._init_mrope_positions = self._extend_mrope_positions
                try:
                    return orig_update_streaming(self, req_id, new_req_data)
                finally:
                    self._init_mrope_positions = orig_init_mrope
            return orig_update_streaming(self, req_id, new_req_data)

        GPUModelRunner._copy_branch_blocks = _copy_branch_blocks
        GPUModelRunner._update_states = _update_states
        GPUModelRunner._extend_mrope_positions = _extend_mrope_positions
        GPUModelRunner._update_streaming_request = _update_streaming_request
        GPUModelRunner._ov_streaming_branch_patched = True


def apply_branch_patches() -> None:
    """Apply all vLLM 0.25.1 streaming-branch patches idempotently."""
    global _PATCHED
    if _PATCHED:
        return
    _patch_request()
    _patch_output_processor()
    _patch_core_client()
    _patch_async_llm()
    _patch_engine_core()
    _patch_kv_cache()
    _patch_scheduler()
    _patch_worker()
    _PATCHED = True
