"""RovoDev Anthropic Messages API Proxy.

A lightweight Flask server that speaks the Anthropic Messages API protocol
(what Claude Code CLI expects at ``ANTHROPIC_BASE_URL``) and backs it with
``RovoDevCliInferencer`` — giving you unlimited LLM inference through your
Atlassian Rovo Dev quota.

Architecture::

    Claude Code CLI
        │  POST {ANTHROPIC_BASE_URL}/v1/messages
        ▼
    RovoDevAnthropicProxy  (this file, Flask server)
        │  acli rovodev legacy "<prompt>"
        ▼
    RovoDevCliInferencer
        │  (unlimited inference via Rovo Dev quota)
        ▼
    Atlassian AI / Claude

Usage::

    # Start proxy on port 9800 mounted at /vertex/claude
    python -m agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_anthropic_proxy \\
        --port 9800 \\
        --base-path /vertex/claude \\
        --cwd ~/MyProjects

    # In ~/.zshrc — point Claude Code at the proxy:
    export ANTHROPIC_BASE_URL=http://localhost:9800/vertex/claude

    # Then just run Claude Code as normal:
    claude "What does this repo do?"

Notes:
    - The server strips the ``--base-path`` prefix and routes
      ``/v1/messages``, ``/v1/models``, and ``/healthcheck``.
    - Multi-turn conversations are handled by keeping a single
      ``RovoDevCliInferencer`` instance per server lifetime. The full
      ``messages`` history is concatenated into a single prompt on each
      call so context is preserved even across Claude Code's multi-turn
      turns (Claude Code re-sends the whole conversation each time).
    - Streaming responses are emulated: the inferencer runs synchronously
      and the full response is chunked into SSE ``text_delta`` events so
      Claude Code's streaming parser is satisfied.
    - The ``model`` field in the request is logged but ignored — RovoDevCli
      doesn't expose a ``--model`` flag; model selection happens via Rovo Dev
      config / Atlassian AI Gateway.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anthropic SSE helpers
# ---------------------------------------------------------------------------

def _sse_event(event: str, data: dict) -> str:
    """Format a single Server-Sent Event line pair."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _build_streaming_response(text: str, model: str, input_tokens: int = 0) -> Iterator[str]:
    """Yield SSE events that match Anthropic's streaming message format.

    Claude Code's streaming parser expects this sequence:
      1. message_start
      2. content_block_start  (index 0, type text)
      3. ping
      4. content_block_delta* (text_delta chunks)
      5. content_block_stop
      6. message_delta        (stop_reason)
      7. message_stop
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    output_tokens = max(1, len(text.split()))

    # 1. message_start
    yield _sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    })

    # 2. content_block_start
    yield _sse_event("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })

    # 3. ping
    yield _sse_event("ping", {"type": "ping"})

    # 4. content_block_delta — emit in chunks of ~200 chars for realism
    chunk_size = 200
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        yield _sse_event("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": chunk},
        })

    # 5. content_block_stop
    yield _sse_event("content_block_stop", {
        "type": "content_block_stop",
        "index": 0,
    })

    # 6. message_delta (stop reason)
    yield _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })

    # 7. message_stop
    yield _sse_event("message_stop", {"type": "message_stop"})


def _build_sync_response(text: str, model: str, input_tokens: int = 0) -> dict:
    """Build a non-streaming Anthropic Messages API response dict."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    output_tokens = max(1, len(text.split()))
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Prompt extraction
# ---------------------------------------------------------------------------

def _extract_prompt(body: dict) -> str:
    """Flatten an Anthropic Messages API request body into a single prompt string.

    Handles:
    - Multi-turn ``messages`` arrays (concatenated with role labels)
    - ``system`` field prepended as a system block
    - Content blocks that are strings or lists of ``{"type": "text", "text": "..."}``

    Claude Code re-sends the *entire* conversation on each turn, so we pass
    the full history to RovoDevCli. This gives the model full context even
    though RovoDevCli is effectively stateless per call.
    """
    parts = []

    system = body.get("system", "")
    if system:
        if isinstance(system, list):
            # system can be a list of content blocks
            system_text = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in system
            )
        else:
            system_text = str(system)
        if system_text.strip():
            parts.append(f"[System]\n{system_text.strip()}")

    messages = body.get("messages", [])
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
                if not isinstance(block, dict) or block.get("type") == "text"
            )
        else:
            text = str(content)
        if text.strip():
            parts.append(f"[{role.capitalize()}]\n{text.strip()}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Fake model list (satisfies /v1/models discovery)
# ---------------------------------------------------------------------------

_FAKE_MODELS = [
    {
        "id": "claude-sonnet-4-5",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "rovodev-proxy",
        "display_name": "Claude via RovoDev Proxy",
    },
    {
        "id": "claude-opus-4-5",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "rovodev-proxy",
        "display_name": "Claude Opus via RovoDev Proxy",
    },
    {
        "id": "claude-haiku-4-5",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "rovodev-proxy",
        "display_name": "Claude Haiku via RovoDev Proxy",
    },
]


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------

def create_app(
    cwd: Optional[str] = None,
    base_path: str = "/vertex/claude",
    acli_path: Optional[str] = None,
    site_url: Optional[str] = None,
    yolo: bool = True,
    agent_mode: Optional[str] = None,
) -> "Flask":  # noqa: F821
    """Create and configure the Flask proxy app.

    Args:
        cwd: Working directory passed to RovoDevCliInferencer. Defaults to cwd
            at import time.
        base_path: URL prefix to strip before routing (e.g. ``"/vertex/claude"``).
            Must match the path component of ``ANTHROPIC_BASE_URL``.
        acli_path: Explicit path to the ``acli`` binary. Auto-detected if None.
        site_url: Atlassian site URL passed to ``acli rovodev legacy``.
        yolo: Skip tool confirmation prompts (default True for proxy use).
        agent_mode: Optional agent mode (``"ask"``, ``"plan"``, etc.).

    Returns:
        Configured Flask application.
    """
    try:
        from flask import Flask, Response, jsonify, request, stream_with_context
    except ImportError as e:
        raise ImportError(
            "Flask is required for the RovoDev Anthropic proxy. "
            "Install it with: pip install flask"
        ) from e

    from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer import (
        RovoDevCliInferencer,
    )

    working_dir = cwd or os.getcwd()
    base_path = base_path.rstrip("/")

    # Single inferencer instance — reused across requests for session continuity
    _inferencer = RovoDevCliInferencer(
        working_dir=working_dir,
        acli_path=acli_path,
        site_url=site_url,
        yolo=yolo,
        agent_mode=agent_mode,
    )

    app = Flask(__name__)

    # ------------------------------------------------------------------
    # WSGI middleware: strip base_path prefix *before* Flask routing
    # ------------------------------------------------------------------
    # Flask's before_request fires AFTER URL routing, so rewriting
    # PATH_INFO there is too late (the 404 already happened). Instead we
    # wrap the WSGI app so the prefix is stripped before Flask ever sees
    # the request.

    if base_path:
        _inner_wsgi = app.wsgi_app

        def _prefix_strip_middleware(environ, start_response):
            path = environ.get("PATH_INFO", "")
            if path.startswith(base_path):
                environ["PATH_INFO"] = path[len(base_path):] or "/"
                environ["SCRIPT_NAME"] = environ.get("SCRIPT_NAME", "") + base_path
            return _inner_wsgi(environ, start_response)

        app.wsgi_app = _prefix_strip_middleware

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    @app.route("/healthcheck", methods=["GET", "HEAD"])
    def healthcheck():
        return jsonify({"status": "ok", "backend": "rovodev-proxy"})

    @app.route("/", methods=["GET", "HEAD"])
    def root():
        return jsonify({"status": "ok", "backend": "rovodev-proxy"})

    # ------------------------------------------------------------------
    # Models list  (GET /v1/models)
    # ------------------------------------------------------------------

    @app.route("/v1/models", methods=["GET"])
    def list_models():
        return jsonify({"object": "list", "data": _FAKE_MODELS})

    # ------------------------------------------------------------------
    # Messages endpoint  (POST /v1/messages)
    # ------------------------------------------------------------------

    @app.route("/v1/messages", methods=["POST"])
    def messages():
        body = request.get_json(force=True, silent=True) or {}
        stream = body.get("stream", False)
        model = body.get("model", "claude-via-rovodev")

        # Build prompt from the full Anthropic messages conversation
        prompt = _extract_prompt(body)
        input_tokens = max(1, len(prompt.split()))

        logger.info(
            "POST /v1/messages | model=%s stream=%s tokens~=%d cwd=%s",
            model, stream, input_tokens, working_dir,
        )

        # ------------------------------------------------------------------
        # Run inference via RovoDevCliInferencer
        # ------------------------------------------------------------------
        try:
            result = _inferencer.infer(prompt)
            # TerminalInferencerResponse: access .output or str()
            if hasattr(result, "output"):
                text = result.output or ""
            else:
                text = str(result) if result else ""

            if not text:
                text = "(No response from Rovo Dev)"
                logger.warning("Empty response from RovoDevCliInferencer")

        except Exception as exc:
            logger.exception("RovoDevCliInferencer error: %s", exc)
            error_body = {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"RovoDev proxy error: {exc}",
                },
            }
            status = 500
            if stream:
                def _error_stream():
                    yield _sse_event("error", error_body)
                return Response(
                    stream_with_context(_error_stream()),
                    status=status,
                    mimetype="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )
            return jsonify(error_body), status

        # ------------------------------------------------------------------
        # Format response
        # ------------------------------------------------------------------
        if stream:
            def _generate():
                yield from _build_streaming_response(text, model, input_tokens)

            return Response(
                stream_with_context(_generate()),
                status=200,
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )
        else:
            return jsonify(_build_sync_response(text, model, input_tokens))

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RovoDev Anthropic Proxy — serve the Anthropic Messages API "
            "backed by acli rovodev legacy (unlimited inference via Rovo Dev quota)."
        )
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=9800,
        help="Port to listen on (default: 9800). "
             "Use 29576 to drop-in replace proximity on its default port.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1). Use 0.0.0.0 for all interfaces.",
    )
    parser.add_argument(
        "--base-path",
        default="/vertex/claude",
        help=(
            "URL path prefix to strip before routing (default: /vertex/claude). "
            "Must match the path component of ANTHROPIC_BASE_URL. "
            "E.g. if ANTHROPIC_BASE_URL=http://localhost:9800/vertex/claude, "
            "set --base-path /vertex/claude."
        ),
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Working directory for RovoDevCliInferencer (default: current directory).",
    )
    parser.add_argument(
        "--acli-path",
        default=None,
        help="Explicit path to the acli binary (auto-detected if not set).",
    )
    parser.add_argument(
        "--site-url",
        default=os.environ.get("JIRA_URL"),
        help="Atlassian site URL passed to acli rovodev legacy (default: $JIRA_URL).",
    )
    parser.add_argument(
        "--agent-mode",
        default=None,
        choices=["ask", "plan", "default"],
        help="RovoDev agent mode override.",
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Require tool confirmation prompts (default: yolo=True for proxy use).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode (verbose logging, auto-reload).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cwd = os.path.expanduser(args.cwd) if args.cwd else os.getcwd()

    print(f"🤖 RovoDev Anthropic Proxy")
    print(f"   Backend  : acli rovodev legacy (unlimited inference)")
    print(f"   CWD      : {cwd}")
    print(f"   Base path: {args.base_path}")
    print(f"   Listening: http://{args.host}:{args.port}")
    print(f"")
    print(f"   Set in ~/.zshrc:")
    print(f"   export ANTHROPIC_BASE_URL=http://localhost:{args.port}{args.base_path}")
    print(f"   export CLAUDE_CODE_USE_BEDROCK=0")
    print()

    app = create_app(
        cwd=cwd,
        base_path=args.base_path,
        acli_path=args.acli_path,
        site_url=args.site_url,
        yolo=not args.no_yolo,
        agent_mode=args.agent_mode,
    )

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
        use_reloader=False,  # Disable reloader to avoid double-start
    )


if __name__ == "__main__":
    main()
