"""Bound the request body before any parser sees it.

`UploadFile.read(max + 1)` is not a defence: by the time a route can call it, Starlette
has already streamed the whole multipart body into a spooled temp file. A multi-gigabyte
body is therefore written to disk before the first size check runs.

This ASGI middleware sits in front of routing and counts raw `http.request` bytes, so the
limit applies to the body itself rather than to a parsed part:

* a declared ``Content-Length`` over the cap is rejected without reading the body at all;
* a missing, chunked or understated ``Content-Length`` is caught by counting as it streams;
* once the cap is crossed, no further chunks are delivered downstream, so no parser, no
  evidence write and no inference can follow.
"""

from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: Hard ceiling for any request body. Evidence is capped at 256 KiB, so the remainder is
#: multipart envelope overhead with generous headroom. Exceeding it is a refusal, not a
#: truncation.
MAX_REQUEST_BYTES = 320 * 1024

_REJECT_BODY = b'{"error":"REQUEST_TOO_LARGE","detail":"request body exceeded the configured limit"}'


class RequestTooLarge(Exception):
    """Raised inside the receive wrapper when the body crosses the cap."""


class RequestSizeLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_bytes: int = MAX_REQUEST_BYTES) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = self._declared_length(scope)
        if declared is not None and declared > self.max_bytes:
            # Reject on the header alone: nothing is read, so nothing is spooled.
            await self._reject(send)
            return

        received = 0
        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        async def receive_wrapper() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestTooLarge()
            return message

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except RequestTooLarge:
            if not response_started:
                await self._reject(send)
            # If a response already began there is no safe way to change it; the body
            # was still cut off, which is the property that matters.

    @staticmethod
    def _declared_length(scope: Scope) -> int | None:
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
        return None

    async def _reject(self, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(_REJECT_BODY)).encode()),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": _REJECT_BODY})
