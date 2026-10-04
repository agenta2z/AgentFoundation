"""Private (0600) files the native orchestrator writes: session instructions,
the Claude CLI's settings and MCP config, spilled turn context and tool
results."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Union


def write_private_file(path: Union[str, Path], text: str) -> Path:
    """Write ``text`` to ``path`` as a new 0600 file that atomically replaces
    any file there. Truncating in place would keep a pre-existing file's wider
    permissions (``os.open``'s mode applies only to a file it creates); a
    symlink at ``path`` is replaced, not followed."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return path
