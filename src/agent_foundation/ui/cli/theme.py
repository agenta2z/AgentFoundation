"""Rich theme and color tokens for CLI SOP UI.

Provides a centralized ThemeManager so every CLI component renders
with consistent colors.  All Rich imports are local to avoid loading
Rich at module level.
"""
from __future__ import annotations

# Color tokens -- plain strings so the module is importable without Rich.
COLORS = {
    "user_border": "cyan",
    "assistant_border": "green",
    "phase_completed": "bold green",
    "phase_running": "bold yellow",
    "phase_pending": "dim white",
    "phase_failed": "bold red",
    "info": "blue",
    "muted": "dim",
}


class ThemeManager:
    """Build and cache a Rich ``Theme`` from the color tokens above."""

    _theme = None

    @classmethod
    def get_theme(cls):
        """Return a ``rich.theme.Theme`` (created lazily on first call)."""
        if cls._theme is None:
            from rich.theme import Theme
            from rich.style import Style

            cls._theme = Theme({
                name: Style.parse(spec) for name, spec in COLORS.items()
            })
        return cls._theme
