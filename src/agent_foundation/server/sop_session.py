"""SOPSession — a conversational session scoped to a single SOP run.

Lives at <parent_session>/sops/<sop_run_folder>/. Each SOP run gets its own
session.jsonl, turn folders, and nested tasks/ directory.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Optional

from agent_foundation.server.turn_artifacts import save_turn_artifacts

logger = logging.getLogger(__name__)


class SOPSession:
    """A conversational session scoped to a single SOP run."""

    def __init__(
        self,
        parent_session_dir: Path,
        sop_name: str,
        instance_id: str,
        sop_config: dict[str, Any] | None = None,
        yolo_mode: bool = False,
    ):
        self.parent_session_dir = parent_session_dir
        self.sop_name = sop_name
        self.instance_id = instance_id
        self.sop_config = sop_config or {}
        self.yolo_mode = yolo_mode
        self._turn_count = 0

        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        uid = instance_id or uuid.uuid4().hex[:8]
        self.folder_name = f"{sop_name}__{ts}__{uid}"
        self.session_dir: Optional[Path] = None

    def allocate(self) -> Path:
        """Create the SOP session directory structure."""
        sops_dir = self.parent_session_dir / "sops"
        sops_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir = sops_dir / self.folder_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        tasks_dir = self.session_dir / "tasks"
        tasks_dir.mkdir(exist_ok=True)

        # Write initial state
        self._write_sop_state()
        self._write_definition_snapshot()
        self._init_session_jsonl()

        logger.info("SOP session allocated: %s", self.session_dir)
        return self.session_dir

    def get_tasks_dir(self) -> Path:
        """Return the tasks/ directory for nested task allocation."""
        if self.session_dir is None:
            raise RuntimeError("SOPSession not allocated; call allocate() first")
        return self.session_dir / "tasks"

    def save_turn_data(
        self,
        turn_idx: int,
        *,
        source: Literal["human", "synthetic", "system"] = "human",
        prompt_data: dict[str, Any] | None = None,
        user_input: str = "",
        response: str = "",
        phase_id: str = "",
        **kwargs: Any,
    ) -> Path:
        """Write turn artifacts to turn_NNN/ folder. Returns the turn directory."""
        if self.session_dir is None:
            raise RuntimeError("SOPSession not allocated")

        turn_dir = self.session_dir / f"turn_{turn_idx:03d}"
        save_turn_artifacts(
            turn_dir,
            prompt_data=prompt_data,
            user_input=user_input,
            response=response,
            source=source,
            phase_id=phase_id,
            instance_id=self.instance_id,
            **kwargs,
        )

        self._append_jsonl({
            "type": "TurnComplete",
            "turn": turn_idx,
            "phase_id": phase_id,
            "source": source,
            "timestamp": datetime.now(UTC).isoformat(),
        })

        self._turn_count = max(self._turn_count, turn_idx)
        return turn_dir

    def get_jsonl_path(self) -> Path:
        """Path to the SOP-run session.jsonl."""
        if self.session_dir is None:
            raise RuntimeError("SOPSession not allocated")
        return self.session_dir / "session.jsonl"

    def to_persistent_dict(self) -> dict[str, Any]:
        """Serializable state for sop_state.json."""
        return {
            "sop_name": self.sop_name,
            "instance_id": self.instance_id,
            "folder_name": self.folder_name,
            "yolo_mode": self.yolo_mode,
            "turn_count": self._turn_count,
            "session_dir": str(self.session_dir) if self.session_dir else None,
        }

    def _write_sop_state(self) -> None:
        if self.session_dir:
            (self.session_dir / "sop_state.json").write_text(
                json.dumps(self.to_persistent_dict(), indent=2), encoding="utf-8"
            )

    def _write_definition_snapshot(self) -> None:
        if self.session_dir and self.sop_config:
            (self.session_dir / "sop_definition_snapshot.json").write_text(
                json.dumps(self.sop_config, indent=2), encoding="utf-8"
            )

    def _init_session_jsonl(self) -> None:
        if self.session_dir:
            jsonl_path = self.session_dir / "session.jsonl"
            if not jsonl_path.exists():
                self._append_jsonl({
                    "type": "SOPSessionCreated",
                    "sop_name": self.sop_name,
                    "instance_id": self.instance_id,
                    "yolo_mode": self.yolo_mode,
                    "timestamp": datetime.now(UTC).isoformat(),
                })

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        if self.session_dir:
            jsonl_path = self.session_dir / "session.jsonl"
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
