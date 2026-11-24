from __future__ import annotations

import asyncio
import base64
import io
import json
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

from .types import Event


class OVSessionError(RuntimeError):
    """The event stream cannot complete the session's remaining work."""


class OVResponseError(RuntimeError):
    """A response ended without a completed answer."""

    def __init__(self, event: Event) -> None:
        self.event = event
        super().__init__(
            event.error or event.reason or event.status or "response failed"
        )


class OVClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
        limits: httpx.Limits | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.limits = limits

    def session(
        self,
        *,
        enable: Sequence[str] | None = None,
        max_responses: int = 3,
        max_response_tokens: int | None = None,
        max_caption_tokens: int | None = None,
        caption_prompt: str | None = None,
        max_visual_frames: int | None = None,
        compact_to_visual_frames: int | None = None,
        pin_first_question: bool = False,
        answer_commit: str = "on_done",
        verification: str | None = None,
        verification_timeout: float = 10.0,
    ) -> ClientSession:
        return ClientSession(
            self.base_url,
            enable=enable,
            limits=self.limits,
            max_responses=max_responses,
            max_response_tokens=max_response_tokens,
            max_caption_tokens=max_caption_tokens,
            caption_prompt=caption_prompt,
            timeout=self.timeout,
            max_visual_frames=max_visual_frames,
            compact_to_visual_frames=compact_to_visual_frames,
            pin_first_question=pin_first_question,
            answer_commit=answer_commit,
            verification=verification,
            verification_timeout=verification_timeout,
        )


class ClientSession:
    def __init__(
        self,
        base_url: str,
        *,
        enable: Sequence[str] | None = None,
        limits: httpx.Limits | None = None,
        max_responses: int,
        timeout: float,
        max_response_tokens: int | None = None,
        max_caption_tokens: int | None = None,
        caption_prompt: str | None = None,
        max_visual_frames: int | None = None,
        compact_to_visual_frames: int | None = None,
        pin_first_question: bool = False,
        answer_commit: str = "on_done",
        verification: str | None = None,
        verification_timeout: float = 10.0,
    ) -> None:
        if max_caption_tokens is not None and (
            type(max_caption_tokens) is not int or max_caption_tokens < 1
        ):
            raise ValueError("max_caption_tokens must be a positive integer or None")
        if caption_prompt is not None and (not isinstance(caption_prompt, str) or not caption_prompt.strip()):
            raise ValueError("caption_prompt must be non-empty text or None")
        self._base_url = base_url
        self._enable = None if enable is None else tuple(enable)
        self._limits = limits
        self._max_responses = max_responses
        self._max_response_tokens = max_response_tokens
        self._max_caption_tokens = max_caption_tokens
        self._caption_prompt = caption_prompt
        self._max_visual_frames = max_visual_frames
        self._compact_to_visual_frames = compact_to_visual_frames
        self._pin_first_question = pin_first_question
        self._answer_commit = answer_commit
        self._verification = verification
        self._verification_timeout = verification_timeout
        self._timeout = timeout
        self._http: httpx.AsyncClient | None = None
        self.session_id: str | None = None
        self._closed = False
        self._event_consumer_task: asyncio.Task[Any] | None = None
        self._event_stream_done = asyncio.Event()
        self._event_stream_done.set()
        self._response_results: dict[str, Event] = {}
        self._response_changed = asyncio.Event()
        self._stream_failure: OVSessionError | None = None

    async def __aenter__(self) -> ClientSession:
        if self._http is not None:
            raise RuntimeError("client session is already open")
        options = {} if self._limits is None else {"limits": self._limits}
        self._http = httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout, **options
        )
        try:
            body = {
                "max_responses": self._max_responses,
                "answer_commit": self._answer_commit,
            }
            if self._enable is not None:
                body["enable"] = list(self._enable)
            if self._max_response_tokens is not None:
                body["max_response_tokens"] = self._max_response_tokens
            if self._max_caption_tokens is not None:
                body["max_caption_tokens"] = self._max_caption_tokens
            if self._caption_prompt is not None:
                body["caption_prompt"] = self._caption_prompt
            if self._max_visual_frames is not None:
                body["max_visual_frames"] = self._max_visual_frames
            if self._compact_to_visual_frames is not None:
                body["compact_to_visual_frames"] = self._compact_to_visual_frames
            if self._pin_first_question:
                body["pin_first_question"] = True
            if self._verification is not None:
                body["verification"] = self._verification
                body["verification_timeout"] = self._verification_timeout
            response = await self._http.post("/v1/ov/sessions", json=body)
            _raise_for_status(response)
            self.session_id = str(response.json()["session_id"])
            if (
                self._max_caption_tokens is not None
                and response.json().get("max_caption_tokens") != self._max_caption_tokens
            ):
                # Older servers ignore unknown fields; do not silently run a
                # different budget or leave their newly created session open.
                await self._http.delete(f"/v1/ov/sessions/{self.session_id}")
                self.session_id = None
                raise OVSessionError("server did not confirm max_caption_tokens")
            if self._caption_prompt is not None and response.json().get("caption_prompt") != self._caption_prompt:
                await self._http.delete(f"/v1/ov/sessions/{self.session_id}")
                self.session_id = None
                raise OVSessionError("server did not confirm caption_prompt")
            return self
        except BaseException:
            await self._http.aclose()
            self._http = None
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        if exc_type is None:
            await self.close()
        else:
            await self.abort()

    async def send_messages(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        input_id: str | None = None,
    ) -> str:
        """Append an ordered OpenAI-style message suffix to this session."""

        http, session_id = self._require_open()
        resolved_input_id = input_id or uuid.uuid4().hex
        response = await http.post(
            f"/v1/ov/sessions/{session_id}/messages",
            json={
                "input_id": resolved_input_id,
                "messages": _encode_messages(messages),
            },
        )
        _raise_for_status(response)
        return str(response.json()["input_id"])

    def __aiter__(self) -> AsyncIterator[Event]:
        return self.events()

    async def events(self) -> AsyncIterator[Event]:
        if self._event_consumer_task is not None:
            raise RuntimeError("only one event consumer is allowed per session")
        http, session_id = self._require_open()
        self._event_consumer_task = asyncio.current_task()
        self._event_stream_done.clear()
        try:
            async with http.stream(
                "GET",
                f"/v1/ov/sessions/{session_id}/events",
                timeout=None,
            ) as response:
                _raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    value = json.loads(line[5:].strip())
                    if not isinstance(value, dict):
                        raise RuntimeError("server emitted a non-object event")
                    event = Event.from_dict(value)
                    if event.type == "error":
                        raise OVSessionError(
                            event.error or event.reason or "session failed"
                        )
                    if event.type in {"response.created", "response.done"}:
                        if event.response_id is None:
                            raise OVSessionError("response event lacks response_id")
                        if event.type == "response.done":
                            self._response_results.setdefault(event.response_id, event)
                            self._response_changed.set()
                    yield event
                    if event.type == "session.closed":
                        return
                if not self._closed:
                    raise OVSessionError("OV event stream ended before session close")
        except OVSessionError as error:
            self._stream_failure = error
            raise
        except Exception as error:
            self._stream_failure = OVSessionError(str(error))
            raise self._stream_failure from error
        finally:
            self._end_stream()
            self._event_consumer_task = None
            self._event_stream_done.set()

    def wait_response(self, response_id: str) -> asyncio.Future[Event]:
        """Wait for response.done while another task consumes events()."""

        async def wait() -> Event:
            while response_id not in self._response_results:
                if self._stream_failure is not None:
                    raise self._stream_failure
                self._response_changed.clear()
                await self._response_changed.wait()
            event = self._response_results[response_id]
            if event.status != "completed":
                raise OVResponseError(event)
            return event

        return asyncio.create_task(wait())

    def _end_stream(self) -> None:
        self._stream_failure = self._stream_failure or OVSessionError(
            "OV event stream is closed"
        )
        self._response_changed.set()

    async def responses(self) -> AsyncIterator[Event]:
        async for event in self.events():
            if event.type == "response.output_text.done":
                yield event

    async def cancel_response(self, response_id: str) -> bool:
        http, session_id = self._require_open()
        response = await http.delete(
            f"/v1/ov/sessions/{session_id}/responses/{response_id}"
        )
        if response.status_code == 404:
            return False
        _raise_for_status(response)
        return True

    async def commit_answer(self, response_id: str, text: str) -> bool:
        """Write back what the user received of an answer (answer_commit="client")."""
        http, session_id = self._require_open()
        response = await http.post(
            f"/v1/ov/sessions/{session_id}/responses/{response_id}/commit",
            json={"text": text},
        )
        if response.status_code == 404:
            return False
        _raise_for_status(response)
        return True

    async def close(self) -> None:
        await self._shutdown(abort=False)

    async def abort(self) -> None:
        await self._shutdown(abort=True)

    async def _shutdown(self, *, abort: bool) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._http is None or self.session_id is None:
                return
            response = await self._http.request(
                "DELETE" if abort else "POST",
                f"/v1/ov/sessions/{self.session_id}" + ("" if abort else "/close"),
                timeout=self._timeout if abort else None,
            )
            if response.status_code != 404:
                _raise_for_status(response)
            if self._event_consumer_task is not asyncio.current_task():
                await self._event_stream_done.wait()
        finally:
            self._end_stream()
            if self._http is not None:
                await self._http.aclose()

    def _require_open(self) -> tuple[httpx.AsyncClient, str]:
        if self._closed or self._http is None or self.session_id is None:
            raise RuntimeError("client session is not open")
        return self._http, self.session_id


def _image_data_url(
    image: bytes | bytearray | memoryview | str | Path | PILImage,
) -> str:
    if isinstance(image, str) and image.startswith("data:"):
        return image
    if isinstance(image, (str, Path)):
        data = Path(image).read_bytes()
    elif isinstance(image, (bytes, bytearray, memoryview)):
        data = bytes(image)
    else:
        from PIL import Image

        if not isinstance(image, Image.Image):
            raise TypeError("image must be bytes, a path, a PIL image, or a data URL")
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=95)
        data = buffer.getvalue()
    media_type = _image_media_type(data)
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _encode_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not messages:
        raise ValueError("messages must not be empty")
    encoded: list[dict[str, Any]] = []
    for raw_message in messages:
        if not isinstance(raw_message, Mapping):
            raise TypeError("messages must contain mappings")
        message = dict(raw_message)
        content = message.get("content")
        if isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray, memoryview)
        ):
            message["content"] = [_encode_content_part(part) for part in content]
        encoded.append(message)
    return encoded


def _encode_content_part(raw_part: Any) -> dict[str, Any]:
    if not isinstance(raw_part, Mapping):
        raise TypeError("message content parts must be mappings")
    part = dict(raw_part)
    if part.get("type") != "image_url":
        return part
    image_url = part.get("image_url")
    if not isinstance(image_url, Mapping) or "url" not in image_url:
        raise ValueError("image_url content requires image_url.url")
    encoded_image_url = dict(image_url)
    encoded_image_url["url"] = _image_data_url(encoded_image_url["url"])
    part["image_url"] = encoded_image_url
    return part


def _image_media_type(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("image must be JPEG, PNG, or WebP")


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if detail:
        raise RuntimeError(f"OV server returned {response.status_code}: {detail}")
    response.raise_for_status()
