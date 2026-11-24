from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from ov_streaming.types import BackendOutput, VisualFrame


@dataclass
class BranchPlan:
    payload: str
    finish_reason: str = "stop"
    release: asyncio.Event = field(default_factory=asyncio.Event)
    submit: asyncio.Event | None = None
    add_started: asyncio.Event = field(default_factory=asyncio.Event)
    completed: asyncio.Event = field(default_factory=asyncio.Event)


class FakeBranch:
    def __init__(self, plan: BranchPlan) -> None:
        self.plan = plan

    async def outputs(self) -> AsyncIterator[BackendOutput]:
        await self.plan.release.wait()
        yield BackendOutput(self.plan.payload, self.plan.finish_reason)
        self.plan.completed.set()


class FakeSource:
    def __init__(self, backend: FakeBackend, request_id: str, cache_salt: str) -> None:
        self.backend = backend
        self.request_id = request_id
        self.cache_salt = cache_salt
        self.text = ""
        self.closed = False
        # One token per character, and image_tokens more per frame.
        self.num_tokens = 0

    async def step(
        self,
        prompt: str,
        frames: Sequence[VisualFrame],
        *,
        allowed_token_ids: Sequence[int],
    ) -> int:
        self._check_running()
        self.text += prompt
        self.num_tokens += len(prompt) + len(frames) * self.backend.image_tokens
        self.backend.steps.append((self.request_id, self.text, tuple(frames)))
        token_id = self.backend.controls.popleft()
        assert token_id in allowed_token_ids
        return token_id

    async def close(self) -> None:
        self._check_running()
        self.closed = True

    async def abort(self) -> None:
        self.backend.aborted_sources.append(self.request_id)

    def _check_running(self) -> None:
        if len(self.backend.steps) == self.backend.stop_source_after:
            raise RuntimeError("source engine stopped")


class FakeBackend:
    def __init__(
        self,
        controls: Sequence[int],
        branches: Sequence[BranchPlan] = (),
        *,
        stop_source_after: int | None = None,
        max_model_len: int | None = None,
        image_tokens: int = 0,
    ) -> None:
        self.max_model_len = max_model_len
        self.image_tokens = image_tokens
        self.controls = deque(controls)
        self.branch_plans = deque(branches)
        self.stop_source_after = stop_source_after
        self.sources: list[FakeSource] = []
        # (source request id, source text after the step, frames of the step)
        self.steps: list[tuple[str, str, tuple[VisualFrame, ...]]] = []
        self.branch_ids: list[str] = []
        self.branch_source_ids: list[str] = []
        self.branch_suffix_token_ids: list[Sequence[int]] = []
        self.branch_max_tokens: list[int] = []
        self.live_branches: dict[str, FakeBranch] = {}
        self.aborted: list[str] = []
        self.aborted_sources: list[str] = []

    def count_tokens(self, text: str) -> int:
        return len(text)

    def open_source(self, request_id: str, *, cache_salt: str) -> FakeSource:
        source = FakeSource(self, request_id, cache_salt)
        self.sources.append(source)
        return source

    async def add_branch(
        self,
        source_request_id: str,
        request_id: str,
        *,
        suffix_token_ids: Sequence[int],
        max_tokens: int,
    ) -> FakeBranch:
        plan = self.branch_plans.popleft()
        self.branch_ids.append(request_id)
        self.branch_source_ids.append(source_request_id)
        self.branch_suffix_token_ids.append(suffix_token_ids)
        self.branch_max_tokens.append(max_tokens)
        plan.add_started.set()
        if plan.submit is not None:
            await plan.submit.wait()
        branch = FakeBranch(plan)
        self.live_branches[request_id] = branch
        return branch

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)
        branch = self.live_branches.get(request_id)
        if branch is not None:
            branch.plan.release.set()
