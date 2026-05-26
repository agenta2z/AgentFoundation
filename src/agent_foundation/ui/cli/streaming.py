"""Token-streaming display for CLI assistants.

``StreamingPanel`` is a context manager that wraps Rich's ``Live``
display to render Markdown progressively as tokens arrive from an LLM.
All Rich imports are deferred to method bodies so this module can be
imported without pulling Rich into the process.
"""
from __future__ import annotations

from typing import Iterator, Optional

from agent_foundation.ui.cli.theme import ThemeManager, COLORS


class StreamingPanel:
    """Context manager for streaming token-by-token output with Rich.Live.

    Usage::

        with StreamingPanel(title="Assistant") as panel:
            for token in llm_stream:
                panel.append(token)

    While no tokens have arrived the panel shows a spinner.  Once the
    first token lands, it switches to incremental Markdown rendering.
    """

    def __init__(self, title: str = "Assistant", border_color: Optional[str] = None):
        self._title = title
        self._border_color = border_color or COLORS["assistant_border"]
        self._buffer = ""
        self._live = None

    # -- context manager protocol -----------------------------------------------

    def __enter__(self):
        from rich.live import Live
        from rich.console import Console
        from rich.spinner import Spinner

        console = Console(theme=ThemeManager.get_theme())
        self._live = Live(
            Spinner("dots", text="Thinking..."),
            console=console,
            refresh_per_second=12,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._live is not None:
            # Render final state as a proper panel before closing
            if self._buffer:
                self._live.update(self._render())
            self._live.__exit__(exc_type, exc_val, exc_tb)
            self._live = None
        return False

    # -- public API -------------------------------------------------------------

    def append(self, token: str) -> None:
        """Add *token* to the accumulated buffer and refresh the display."""
        self._buffer += token
        if self._live is not None:
            self._live.update(self._render())

    # -- internal ---------------------------------------------------------------

    def _render(self):
        from rich.markdown import Markdown
        from rich.panel import Panel

        md = Markdown(self._buffer)
        return Panel(md, title=self._title, border_style=self._border_color)
