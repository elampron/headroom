import asyncio
import json
from types import SimpleNamespace

import pytest

from headroom.proxy.stream_hooks import StreamHookMiddleware, VisibleSSE
from headroom.proxy.turn_hooks import (
    TurnContext,
    clear_turn_hooks,
    register_turn_hook,
    run_request_hooks,
)


def sse(event):
    return b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\r\n\r\n"


@pytest.mark.parametrize(
    "events",
    [
        [
            {
                "choices": [
                    {"index": 0, "delta": {"content": "Héllo", "reasoning_content": "secret"}}
                ]
            },
            "done",
        ],
        [
            {"type": "response.reasoning_text.delta", "delta": "secret"},
            {"type": "response.output_text.delta", "delta": "Héllo"},
            {"type": "response.completed"},
        ],
        [
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "secret"},
            },
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Héllo"}},
            {"type": "message_stop"},
        ],
    ],
)
def test_provider_streams_fragmented_utf8_reasoning_excluded(events):
    observer = VisibleSSE()
    raw = b"".join(b"data: [DONE]\r\n\r\n" if e == "done" else sse(e) for e in events)
    for byte in raw:
        observer.feed(bytes([byte]))
    result = observer.result()
    assert result["choices"][0]["message"]["content"] == "Héllo"
    assert result["stream_status"] == "complete"
    assert "secret" not in json.dumps(result)


def test_bound_and_incomplete_without_terminal_marker(monkeypatch):
    import headroom.proxy.stream_hooks as module

    observer = VisibleSSE()
    observer.feed(sse({"choices": [{"delta": {"content": "partial"}}]}))
    assert observer.result()["stream_status"] == "incomplete"
    monkeypatch.setattr(module, "MAX_BYTES", observer.size + 2)
    observer.feed(b"too large")
    assert observer.result()["stream_truncated"] is True
    assert observer.result()["choices"][0]["message"]["content"] == "partial"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_real_asgi_stream_bytes_unchanged_final_hook_on_cancel(cancel):
    from fastapi import FastAPI
    from starlette.responses import StreamingResponse

    clear_turn_hooks()
    seen = []
    register_turn_hook(
        SimpleNamespace(stream_safe=True, on_stream_end=lambda c, r: seen.append((c, r)))
    )
    app = FastAPI()
    app.add_middleware(StreamHookMiddleware)
    raw = sse({"choices": [{"delta": {"content": "final answer"}}]})

    @app.get("/stream")
    async def stream():
        run_request_hooks(
            TurnContext(
                "openai",
                "test",
                [{"role": "user", "content": "Question"}],
                session_id="chat",
                request_id="req",
            ),
            stream_safe_only=True,
        )

        async def chunks():
            yield raw
            if cancel:
                raise asyncio.CancelledError()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(chunks(), media_type="text/event-stream")

    delivered = []

    async def send(message):
        delivered.append(message)

    async def receive():
        await asyncio.sleep(30)
        return {"type": "http.disconnect"}

    try:
        await app(
            {
                "type": "http",
                "asgi": {"spec_version": "2.4"},
                "method": "GET",
                "path": "/stream",
                "query_string": b"",
                "headers": [],
                "scheme": "http",
                "server": ("test", 80),
            },
            receive,
            send,
        )
    except asyncio.CancelledError:
        assert cancel
    finally:
        clear_turn_hooks()
    assert b"".join(m.get("body", b"") for m in delivered) == raw + (
        b"" if cancel else b"data: [DONE]\n\n"
    )
    assert len(seen) == 1
    assert seen[0][0].session_id == "chat"
    assert seen[0][1]["stream_status"] == ("incomplete" if cancel else "complete")


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_mix_and_hook_failure_isolated():
    clear_turn_hooks()
    seen = []

    def broken(ctx, result):
        raise ValueError("test")

    register_turn_hook(SimpleNamespace(stream_safe=True, on_stream_end=broken))
    register_turn_hook(
        SimpleNamespace(stream_safe=True, on_stream_end=lambda c, r: seen.append((c.session_id, r)))
    )

    async def app(scope, receive, send):
        run_request_hooks(
            TurnContext("openai", "test", [], session_id=scope["path"]), stream_safe_only=True
        )
        await asyncio.sleep(0)
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": sse({"choices": [{"delta": {"content": scope["path"]}}]})
                + b"data: [DONE]\n\n",
            }
        )

    async def noop(*args):
        pass

    try:
        middleware = StreamHookMiddleware(app)
        await asyncio.gather(
            *(middleware({"type": "http", "path": str(i)}, noop, noop) for i in range(3))
        )
    finally:
        clear_turn_hooks()
    assert len(seen) == 3
    for session, result in seen:
        assert result["choices"][0]["message"]["content"] == session
