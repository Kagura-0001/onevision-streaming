"""Opt-in verification policies; inference and cache ownership stay in the session."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .renderers import Renderer


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    reason: str
    text: str = ""


class RoadInspectionVerification:
    name = "road_inspection"
    max_tokens = 128

    def instruction(self, renderer: Renderer, timestamp_sec: float) -> str:
        return (
            f"复核视频 {timestamp_sec:.3f} 秒的最新画面，判断现在是否应提醒道路损坏。"
            "提醒中的判断和描述必须符合当前画面可见的证据。"
            "历史提醒只能提供背景，不能代替最新画面的证据。"
            "先确认道路看得清；明确看到铺装或路缘缺失、路面破损、"
            "土石覆盖通行面时才提醒。裸露山坡、河床堆积、转弯、遮挡或模糊"
            "本身不能证明道路损坏。看不清或证据不足时保持静默。"
            "同一损坏连续可见且已提醒时保持静默；离开后再次拍到可以提醒。"
            f"无需提醒时只输出{renderer.controls['silence']}。"
            f"需要提醒时输出{renderer.controls['response']}，"
            "随后用一至两句先描述可见的具体证据，再给必要的核查建议。"
            "不写固定开场白，不推断灾害成因或无法确认的损坏程度、通行能力。"
            "只输出上述标记和必要的提醒正文。"
        )

    def parse(
        self, raw: str, renderer: Renderer, *, truncated: bool
    ) -> VerificationResult:
        if truncated:
            return VerificationResult(False, "truncated")
        raw = raw.strip()
        if raw == renderer.controls["silence"]:
            return VerificationResult(False, "no_alert")
        marker = renderer.controls["response"]
        if not raw.startswith(marker):
            return VerificationResult(False, "invalid_output")
        text = raw[len(marker) :].strip()
        if not text:
            return VerificationResult(False, "empty_answer")
        try:
            renderer.check_text(text)
        except ValueError:
            return VerificationResult(False, "invalid_output")
        return VerificationResult(True, "accepted", text)


def load_verification(
    name: str | None, renderer: Renderer
) -> RoadInspectionVerification | None:
    if name is None:
        return None
    if name != RoadInspectionVerification.name:
        raise ValueError(f"unknown verification policy: {name!r}")
    if not {"silence", "response"} <= renderer.controls.keys():
        raise ValueError("verification requires silence and response controls")
    return RoadInspectionVerification()
