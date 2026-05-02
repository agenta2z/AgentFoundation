"""Chat message and metadata models for conversational server sessions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MessageMetadata:
    model: str | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None
    duration_ms: int | None = None
    is_tool_call: bool = False
    tool_name: str | None = None
    is_auto_advance: bool = False
    is_task_ref: bool = False
    widget_type: str | None = None
    widget_data: dict | None = None
    task_status: str | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "model": self.model,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "duration_ms": self.duration_ms,
        }
        if self.is_tool_call:
            d["is_tool_call"] = True
        if self.tool_name is not None:
            d["tool_name"] = self.tool_name
        if self.is_auto_advance:
            d["is_auto_advance"] = True
        if self.is_task_ref:
            d["is_task_ref"] = True
        if self.widget_type is not None:
            d["widget_type"] = self.widget_type
        if self.widget_data is not None:
            d["widget_data"] = self.widget_data
        if self.task_status is not None:
            d["task_status"] = self.task_status
        return d

    @classmethod
    def from_dict(cls, data: dict) -> MessageMetadata:
        return cls(
            model=data.get("model"),
            tokens_input=data.get("tokens_input"),
            tokens_output=data.get("tokens_output"),
            duration_ms=data.get("duration_ms"),
            is_tool_call=data.get("is_tool_call", False),
            tool_name=data.get("tool_name"),
            is_auto_advance=data.get("is_auto_advance", False),
            is_task_ref=data.get("is_task_ref", False),
            widget_type=data.get("widget_type"),
            widget_data=data.get("widget_data"),
            task_status=data.get("task_status"),
        )


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "task_ref" | "widget_response"
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: MessageMetadata | None = None
    task_id: str | None = None
    multi_task_id: str | None = None

    def to_api_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    def to_dict(self) -> dict:
        d: dict = {
            "role": self.role,
            "content": self.content,
            "id": self.id,
            "timestamp": self.timestamp,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }
        if self.task_id is not None:
            d["task_id"] = self.task_id
        if self.multi_task_id is not None:
            d["multi_task_id"] = self.multi_task_id
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ChatMessage:
        metadata = None
        if data.get("metadata"):
            metadata = MessageMetadata.from_dict(data["metadata"])
        return cls(
            role=data["role"],
            content=data["content"],
            id=data.get("id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metadata=metadata,
            task_id=data.get("task_id"),
            multi_task_id=data.get("multi_task_id"),
        )
