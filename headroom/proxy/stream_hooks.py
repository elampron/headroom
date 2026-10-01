"""Read-only, bounded SSE observation at the client delivery boundary.

The synchronous completion callback must enqueue local work only. It cannot
replace output or re-drive a model; cancellation still triggers it. Context is
request-local, including across Starlette task boundaries (a mutable carrier).
"""

import copy
import json
import logging
from contextvars import ContextVar

log = logging.getLogger(__name__)
_carrier = ContextVar("headroom_stream_observer", default=None)
MAX_BYTES = 4 * 1024 * 1024


def remember_stream_context(ctx):
    carrier = _carrier.get()
    if carrier is not None:
        try:
            snapshot = copy.copy(ctx)
            snapshot.messages = copy.deepcopy(ctx.messages)
            carrier["ctx"] = snapshot
        except Exception:
            carrier.pop("ctx", None)
            log.warning("Stream context unavailable; delivery unaffected")


class VisibleSSE:
    def __init__(self):
        self.buffer = b""
        self.parts = {}
        self.data_lines = []
        self.size = 0
        self.complete = False
        self.truncated = False
        self.failed = False

    def feed(self, chunk):
        if self.truncated:
            return
        self.size += len(chunk)
        if self.size > MAX_BYTES:
            self.truncated = True
            self.buffer = b""
            return
        self.buffer += chunk
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if line.startswith(b"data:"):
                self.data_lines.append(line[5:].lstrip(b" "))
            elif not line and self.data_lines:
                data = b"\n".join(self.data_lines).strip()
                self.data_lines.clear()
                if data == b"[DONE]":
                    self.complete = True
                    continue
                try:
                    self.event(json.loads(data))
                except (ValueError, TypeError, AttributeError, KeyError):
                    self.failed = True

    def event(self, event):
        kind = event.get("type")
        if kind in {"error", "response.failed", "response.incomplete"} or event.get("error"):
            self.failed = True
        if kind in {"message_stop", "response.completed"}:
            self.complete = True
        # Whitelist visible text. Never collect reasoning/thinking/signatures.
        if kind == "response.output_text.delta":
            key = (event.get("output_index", 0), event.get("content_index", 0))
            self.append(key, event.get("delta"))
        elif kind == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "text":
                self.append((0, event.get("index", 0)), block.get("text"))
        elif kind == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                self.append((0, event.get("index", 0)), delta.get("text"))
        for choice in event.get("choices", []):
            if choice.get("index", 0) == 0:
                self.append((0, 0), choice.get("delta", {}).get("content"))

    def append(self, key, text):
        if isinstance(text, str):
            self.parts.setdefault(key, []).append(text)

    def result(self):
        text = "".join("".join(self.parts[key]) for key in sorted(self.parts))
        return {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "stream_status": "complete"
            if self.complete and not self.failed and not self.truncated
            else "incomplete",
            "stream_truncated": self.truncated,
        }


class StreamHookMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        from headroom.proxy.turn_hooks import registered_turn_hooks

        hooks = [
            h
            for h in registered_turn_hooks()
            if getattr(h, "stream_safe", False) and callable(getattr(h, "on_stream_end", None))
        ]
        if scope["type"] != "http" or not hooks:
            return await self.app(scope, receive, send)
        carrier = {}
        token = _carrier.set(carrier)
        observer = VisibleSSE()
        enabled = False

        async def observe_send(message):
            nonlocal enabled
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                enabled = (
                    message["status"] == 200
                    and b"text/event-stream" in headers.get(b"content-type", b"")
                    and headers.get(b"content-encoding", b"identity") == b"identity"
                )
            # Preserve original messages/bytes and backpressure unchanged.
            await send(message)
            if enabled and message["type"] == "http.response.body":
                try:
                    observer.feed(message.get("body", b""))
                except Exception:
                    observer.failed = True
                    log.warning("Stream observation failed; delivery unaffected")

        try:
            await self.app(scope, receive, observe_send)
        finally:
            _carrier.reset(token)
            ctx = carrier.get("ctx")
            if enabled and ctx is not None:
                try:
                    response = observer.result()
                except Exception:
                    log.warning("Stream result unavailable; delivery unaffected")
                    response = None
                for hook in hooks if response is not None else ():
                    try:
                        hook.on_stream_end(ctx, copy.deepcopy(response))
                    except Exception:
                        log.warning("Stream completion hook failed; delivery unaffected")
