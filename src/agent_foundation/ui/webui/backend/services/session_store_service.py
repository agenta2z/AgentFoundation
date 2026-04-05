# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""SessionStoreService — read-only access to the server's session file store.

The RankEvolve server persists session state to disk as session_state.json
files. This service reads those files so the WebUI backend can serve session
data via HTTP endpoints without going through the queue.

The server is the sole writer; this service is read-only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SessionStoreService:
    """Read-only service for accessing session data from the server's file store.

    The server writes session_state.json files atomically (tmp + os.replace)
    under <server_dir>/sessions/<session_id>_<timestamp>/.

    This service scans those directories and reads the JSON files. It is safe
    for concurrent reads because the server's atomic writes guarantee readers
    always see a complete file.
    """

    def __init__(self, server_dir: str) -> None:
        self._server_dir = Path(server_dir)
        self._sessions_dir = self._server_dir / "sessions"

    @property
    def sessions_dir(self) -> Path:
        return self._sessions_dir

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions with metadata.

        Prefers sessions_index.json (written by server) for fast O(1) reads.
        Falls back to scanning session directories.
        """
        # Fast path: read index file
        index_path = self._sessions_dir / "sessions_index.json"
        if index_path.is_file():
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
                return data.get("sessions", [])
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read sessions_index.json: %s", e)

        # Fallback: scan directories
        return self._scan_sessions()

    def get_session_state(self, session_id: str) -> dict[str, Any] | None:
        """Read full session_state.json for a given session."""
        session_dir = self._find_session_dir(session_id)
        if session_dir is None:
            return None

        state_file = session_dir / "session_state.json"
        if not state_file.is_file():
            return None

        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Failed to read session_state.json for %s: %s", session_id, e
            )
            return None

    def get_session_messages(self, session_id: str) -> list[dict[str, Any]] | None:
        """Extract conversation messages from session_state.json."""
        state = self.get_session_state(session_id)
        if state is None:
            return None

        conversation = state.get("conversation")
        if conversation is None:
            return []

        return conversation.get("messages", [])

    def get_session_config(self, session_id: str) -> dict[str, Any] | None:
        """Extract config (model, target_path, provider) from session_state.json."""
        state = self.get_session_state(session_id)
        if state is None:
            return None

        info = state.get("info", {})
        return {
            "model": info.get("model", ""),
            "target_path": info.get("target_path", ""),
            "provider": info.get("provider", ""),
        }

    def get_turn_data(self, session_id: str, turn_number: int) -> dict | None:
        """Read all turn data (prompt, response, metadata) from parts files."""
        session_dir = self._find_session_dir(session_id)
        if not session_dir:
            return None
        turn_dir = session_dir / f"turn_{turn_number:03d}"
        if not turn_dir.is_dir():
            return None

        result: dict[str, Any] = {"turn_number": turn_number}
        parts_base = turn_dir / "session.jsonl.parts"
        for subdir_name, key in [
            ("UserInput", "user_input"),
            ("PromptTemplate", "prompt_template"),
            ("TemplateFeed", "template_feed"),
            ("RenderedPrompt", "rendered_prompt"),
            ("TemplateConfig", "template_config"),
            ("ApiPayload", "api_payload"),
            ("InferenceResponse", "response"),
            ("TurnMetadata", "metadata"),
        ]:
            subdir = parts_base / subdir_name
            if subdir.is_dir():
                files = sorted(subdir.iterdir(), reverse=True)
                if files:
                    try:
                        content = files[0].read_text(encoding="utf-8")
                        if key in ("metadata", "template_feed", "api_payload", "template_config"):
                            try:
                                content = json.loads(content)
                            except (json.JSONDecodeError, ValueError):
                                pass
                        result[key] = content
                    except OSError:
                        pass
        return result

    def _find_session_dir(self, session_id: str) -> Path | None:
        """Find the session directory for a given session_id.

        Session directories are named <session_id>_<YYYYMMDD_HHMMSS>.
        If multiple match (e.g. from restarts), return the most recent.
        """
        if not self._sessions_dir.is_dir():
            return None

        prefix = f"{session_id}_"
        candidates = [
            d
            for d in self._sessions_dir.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        ]

        if not candidates:
            return None

        # Most recent by name (timestamp suffix) or mtime
        candidates.sort(key=lambda d: d.name, reverse=True)
        return candidates[0]

    def _scan_sessions(self) -> list[dict[str, Any]]:
        """Scan session directories and build a session list from state files."""
        if not self._sessions_dir.is_dir():
            return []

        sessions = []
        for subdir in sorted(self._sessions_dir.iterdir()):
            if not subdir.is_dir():
                continue
            state_file = subdir / "session_state.json"
            if not state_file.is_file():
                continue
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                if data.get("status") == "closed":
                    continue
                info = data.get("info", {})
                sessions.append(
                    {
                        "session_id": info.get("session_id", ""),
                        "session_type": info.get("session_type", ""),
                        "model": info.get("model", ""),
                        "target_path": info.get("target_path", ""),
                        "provider": info.get("provider", ""),
                        "status": data.get("status", "active"),
                        "created_at": info.get("created_at", 0),
                        "last_active": info.get("last_active", 0),
                    }
                )
            except (json.JSONDecodeError, OSError):
                continue

        return sessions
