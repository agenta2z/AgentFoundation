"""Rovo Dev inferencer shared constants and helpers."""

import os
import re
import shutil
import socket
from pathlib import Path
from typing import Optional


# CLI binary
ACLI_BINARY = "acli"
ACLI_SUBCOMMAND = "rovodev"

# Timeouts
DEFAULT_IDLE_TIMEOUT = 1800  # 30 minutes
DEFAULT_TOOL_USE_IDLE_TIMEOUT = 7200  # 2 hours for tool use

# Serve-mode
DEFAULT_PORT_RANGE_START = 19100
DEFAULT_PORT_RANGE_END = 19200
HEALTHCHECK_POLL_INTERVAL = 0.5  # seconds between health check polls
DEFAULT_STARTUP_TIMEOUT = 60  # max seconds to wait for server startup

# ANSI escape code pattern
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class RovoDevNotFoundError(RuntimeError):
    """Raised when the acli binary is not found."""

    def __init__(self, msg: Optional[str] = None) -> None:
        super().__init__(
            msg
            or (
                f"'{ACLI_BINARY}' not found in PATH. "
                "Install the Atlassian CLI: https://developer.atlassian.com/cli"
            )
        )


class RovoDevAuthError(RuntimeError):
    """Raised when authentication with Rovo Dev fails."""

    def __init__(self, msg: Optional[str] = None) -> None:
        super().__init__(
            msg
            or (
                "Rovo Dev authentication failed. "
                "Run 'acli auth login' to authenticate."
            )
        )


class RovoDevServerStartError(RuntimeError):
    """Raised when the Rovo Dev serve process fails to start."""


def find_acli_binary(explicit_path: Optional[str] = None) -> str:
    """Find the acli binary.

    Args:
        explicit_path: Explicit path to the acli binary. If provided, returned as-is.

    Returns:
        Path to the acli binary.

    Raises:
        RovoDevNotFoundError: If acli is not found.
    """
    if explicit_path:
        return explicit_path
    path = shutil.which(ACLI_BINARY)
    if not path:
        raise RovoDevNotFoundError()
    return path


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape sequences from terminal output.

    Strips:
    - ANSI escape codes (colors, cursor movement, etc.)
    - Carriage returns with overwritten content (spinner updates)

    Args:
        text: Text potentially containing ANSI escape codes.

    Returns:
        Clean text with ANSI codes removed.
    """
    # Strip ANSI escape codes
    text = _ANSI_ESCAPE_RE.sub("", text)
    # Strip carriage return overwrites (spinner lines)
    text = re.sub(r"\r[^\n]*", "", text)
    return text


def find_available_port(
    start: int = DEFAULT_PORT_RANGE_START,
    end: int = DEFAULT_PORT_RANGE_END,
) -> int:
    """Find an available TCP port in the given range.

    Args:
        start: Start of port range (inclusive).
        end: End of port range (exclusive).

    Returns:
        An available port number.

    Raises:
        RuntimeError: If no available port is found in the range.
    """
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No available port in range {start}-{end}")


# Environment variables that must be removed to prevent nested-session detection
# when spawning child acli rovodev processes from within a Rovo Dev session.
CONFLICTING_ENV_VARS = [
    "ROVODEV_CLI",       # Prevents nested sessions
    "_PYI_ARCHIVE_FILE", # PyInstaller archive (from parent binary)
]


def clean_env_for_subprocess() -> dict:
    """Return a copy of os.environ with conflicting vars removed.

    When a Rovo Dev inferencer runs inside an existing Rovo Dev CLI session,
    the child ``acli rovodev legacy`` process detects ``ROVODEV_CLI=1`` and
    exits early to prevent nesting. This function strips those vars.
    """
    import os

    env = os.environ.copy()
    for var in CONFLICTING_ENV_VARS:
        env.pop(var, None)
    return env


# Default session persistence directory
DEFAULT_SESSIONS_DIR = os.path.expanduser("~/.rovodev/sessions")


def find_latest_session_id(
    sessions_dir: str = DEFAULT_SESSIONS_DIR,
    workspace_path: str | None = None,
) -> str | None:
    """Find the most recently modified session ID in the sessions directory.

    Checks ``session_context.json`` modification time to find the latest session.
    If ``workspace_path`` is provided, tries to filter by matching workspace first.
    Falls back to the most recent session overall if no workspace match is found
    (sessions created programmatically may not have ``metadata.json``).

    Args:
        sessions_dir: Path to the sessions directory.
        workspace_path: If provided, prefer sessions matching this workspace.

    Returns:
        The session ID (UUID folder name) or None if no sessions found.
    """
    import json

    sessions_path = Path(sessions_dir)
    if not sessions_path.exists():
        return None

    all_sessions: list[tuple[str, float]] = []
    workspace_sessions: list[tuple[str, float]] = []
    resolved_ws = Path(workspace_path).resolve() if workspace_path else None

    for session_dir in sessions_path.iterdir():
        if not session_dir.is_dir():
            continue

        ctx_file = session_dir / "session_context.json"
        if not ctx_file.exists():
            continue

        mtime = ctx_file.stat().st_mtime
        all_sessions.append((session_dir.name, mtime))

        if resolved_ws:
            # Try metadata.json first, then fall back to session_context.json
            session_ws = ""
            meta_file = session_dir / "metadata.json"
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                    session_ws = meta.get("workspace_path", "")
                except (json.JSONDecodeError, OSError):
                    pass
            if not session_ws:
                try:
                    ctx_data = json.loads(ctx_file.read_text())
                    session_ws = ctx_data.get("workspace_path", "")
                except (json.JSONDecodeError, OSError):
                    pass
            if session_ws:
                try:
                    if Path(session_ws).resolve() == resolved_ws:
                        workspace_sessions.append((session_dir.name, mtime))
                except (ValueError, OSError):
                    pass

    candidates = workspace_sessions if workspace_sessions else all_sessions
    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def ensure_session_metadata(
    session_id: str,
    workspace_path: str | None = None,
    sessions_dir: str = DEFAULT_SESSIONS_DIR,
) -> None:
    """Write a ``metadata.json`` for a session if one does not already exist.

    The Rovo Dev CLI's ``--restore`` flag uses ``metadata.json`` to filter
    sessions by workspace.  Sessions created by the CLI in non-interactive
    (legacy) mode do not write this file, so ``--restore`` silently skips
    them.  This helper fills the gap so that programmatically created
    sessions can be restored.

    Args:
        session_id: The session UUID (directory name under *sessions_dir*).
        workspace_path: Absolute path to the workspace for this session.
        sessions_dir: Root sessions directory.
    """
    import json

    session_dir = Path(sessions_dir) / session_id
    meta_file = session_dir / "metadata.json"
    if meta_file.exists():
        return  # Already has metadata

    if not session_dir.exists():
        return  # Session directory doesn't exist

    resolved_ws = str(Path(workspace_path).resolve()) if workspace_path else None
    metadata = {
        "title": None,
        "is_manual_title": False,
        "fork_data": None,
        "workspace_path": resolved_ws,
        "bookmarks": [],
    }
    try:
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except OSError:
        pass  # Best-effort; don't crash if we can't write
