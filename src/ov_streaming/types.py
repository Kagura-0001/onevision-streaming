from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class VisualFrame:
    text: str
    data: Any = field(repr=False)


@dataclass(frozen=True)
class Memory:
    kind: str  # "question" or the name of the control that produced it
    timestamp_sec: float
    text: str


@dataclass
class Turn:
    timestamp_sec: float
    frames: tuple[VisualFrame, ...] = ()
    questions: tuple[Memory, ...] = ()
    # Written back in this turn, in completion order.
    memories: tuple[Memory, ...] = ()
    control: str | None = None
    # Tokens the turn adds to a source, counted once compaction may replay it.
    tokens: int = 0


@dataclass(frozen=True)
class BackendOutput:
    text: str
    finish_reason: str | None


@dataclass(frozen=True)
class Event:
    type: str
    session_id: str
    input_id: str | None = None
    response_id: str | None = None
    timestamp_sec: float | None = None
    text: str | None = None
    status: str | None = None
    reason: str | None = None
    error: str | None = None
    control: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = {
            f.name: getattr(self, f.name) for f in fields(self) if f.name != "metadata"
        }
        value.update(self.metadata)
        return {key: item for key, item in value.items() if item is not None}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Event:
        names = {f.name for f in fields(cls)} - {"metadata"}
        return cls(
            **{name: value.get(name) for name in names},
            metadata={key: item for key, item in value.items() if key not in names},
        )
