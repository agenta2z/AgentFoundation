#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Entry point for running the FastAPI WebUI host via Buck2.

Supports two modes:
- demo (default): Plays back pre-authored JSON flow scripts
- real: Connects to a running RankEvolve server via file-based queues

Usage:
    # Demo mode (default)
    buck2 run //rankevolve/src/webui:run_webui -- --port 8087

    # Real mode (requires run_webui_real binary and a running server)
    buck2 run //rankevolve/src/webui:run_webui_real -- --mode real --queue-root <path>

    # Real mode with options
    buck2 run //rankevolve/src/webui:run_webui_real -- --mode real \\
        --queue-root <path> --debug
"""

import argparse
import logging
import sys


def main() -> None:
    """Main entry point for the backend server."""
    parser = argparse.ArgumentParser(description="CoScience Chatbot WebUI")

    # Server options
    parser.add_argument("--port", type=int, default=8087, help="Port to listen on")
    parser.add_argument("--host", default="::", help="Host to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--ssl-keyfile", help="Path to SSL key file")
    parser.add_argument("--ssl-certfile", help="Path to SSL certificate file")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (shows timing details)",
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["demo", "real"],
        default="demo",
        help="Server mode: demo (playback) or real (agent)",
    )

    # Demo-specific options
    parser.add_argument(
        "--iteration",
        type=str,
        default=None,
        help="Start from specific iteration (demo mode, e.g., '1' or 'iteration_1')",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="Start from specific step within iteration (demo mode, 0-based offset)",
    )
    parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="Speed up all delays by this factor (demo mode, e.g., 2 = 2x faster)",
    )

    # Agent-specific options (real mode)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model override (real mode, e.g., 'claude-sonnet-4.5')",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider override (real mode, e.g., 'plugboard', 'anthropic')",
    )
    parser.add_argument(
        "--target-path",
        type=str,
        default=None,
        help="Codebase root path for /task commands (real mode)",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default=None,
        help="Plugboard pipeline override (real mode)",
    )
    parser.add_argument(
        "--enable-knowledge",
        action="store_true",
        help="Enable knowledge bridge (real mode, experimental)",
    )
    parser.add_argument(
        "--queue-root",
        type=str,
        default=None,
        help="Queue root path for connecting to a running server (real mode)",
    )

    args = parser.parse_args()

    # Runtime guard: real mode requires --queue-root
    if args.mode == "real" and not args.queue_root:
        print(
            "ERROR: Real mode requires --queue-root pointing to the server's queue directory."
        )
        print(
            "Start the server first, then pass its queue root path:\n"
            "  buck2 run //rankevolve/src/webui:run_webui_real -- --mode real --queue-root <path>"
        )
        sys.exit(1)

    # Configure logging based on --debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Import uvicorn and app here to avoid import errors during arg parsing
    import uvicorn
    from chatbot_demo_react.backend.main import app

    # Pass startup arguments to app state for use in lifespan
    app.state.mode = args.mode

    if args.mode == "demo":
        app.state.startup_iteration = args.iteration
        app.state.startup_step = args.step
        app.state.startup_speedup = args.speedup
    else:
        app.state.queue_root_path = args.queue_root
        app.state.agent_model = args.model
        app.state.agent_provider = args.provider
        app.state.agent_target_path = args.target_path
        app.state.agent_pipeline = args.pipeline
        app.state.agent_enable_knowledge = args.enable_knowledge

    print(f"Starting CoScience Chatbot WebUI ({args.mode} mode)...")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Mode: {args.mode}")
    if args.debug:
        print("  Debug: Enabled (verbose logging)")

    if args.mode == "demo":
        if args.iteration:
            print(f"  Start iteration: {args.iteration}")
        if args.step:
            print(f"  Start step offset: {args.step}")
        if args.speedup > 1.0:
            print(f"  Speedup: {args.speedup}x faster animations")
    else:
        if args.queue_root:
            print(f"  Queue root: {args.queue_root}")
        if args.model:
            print(f"  Model: {args.model}")
        if args.provider:
            print(f"  Provider: {args.provider}")
        if args.target_path:
            print(f"  Target path: {args.target_path}")
        if args.pipeline:
            print(f"  Pipeline: {args.pipeline}")
        if args.enable_knowledge:
            print("  Knowledge: Enabled")

    if args.ssl_keyfile:
        print("  SSL: Enabled")
    print()

    # Build uvicorn config
    uvicorn_log_level = "debug" if args.debug else "info"
    uvicorn_config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        log_level=uvicorn_log_level,
    )

    if args.ssl_keyfile and args.ssl_certfile:
        uvicorn_config.ssl_keyfile = args.ssl_keyfile
        uvicorn_config.ssl_certfile = args.ssl_certfile

    server = uvicorn.Server(uvicorn_config)
    server.run()


if __name__ == "__main__":
    main()
