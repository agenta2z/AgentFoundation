"""Conversation manager for multi-turn chat sessions."""

from __future__ import annotations

from agent_foundation.server.schema import ChatMessage, MessageMetadata


class Conversation:
    """Manages conversation history for multi-turn chat."""

    def __init__(self, system_prompt: str) -> None:
        self.system_prompt: str = system_prompt
        self.messages: list[ChatMessage] = []

    def add_user_message(self, content: str) -> ChatMessage:
        msg = ChatMessage(role="user", content=content)
        self.messages.append(msg)
        return msg

    def add_assistant_message(
        self,
        content: str,
        metadata: MessageMetadata | None = None,
    ) -> ChatMessage:
        msg = ChatMessage(role="assistant", content=content, metadata=metadata)
        self.messages.append(msg)
        return msg

    def add_tool_call_message(
        self,
        tool_name: str,
        arguments: dict,
    ) -> ChatMessage:
        content = f"[Tool Call: {tool_name}] {arguments}"
        metadata = MessageMetadata(is_tool_call=True, tool_name=tool_name)
        msg = ChatMessage(role="assistant", content=content, metadata=metadata)
        self.messages.append(msg)
        return msg

    def add_tool_result_message(
        self,
        tool_name: str,
        result: str,
    ) -> ChatMessage:
        content = f"[Tool Result: {tool_name}] {result}"
        metadata = MessageMetadata(tool_name=tool_name, is_auto_advance=True)
        msg = ChatMessage(role="user", content=content, metadata=metadata)
        self.messages.append(msg)
        return msg

    def add_auto_advance_message(self, content: str) -> ChatMessage:
        metadata = MessageMetadata(is_auto_advance=True)
        msg = ChatMessage(role="user", content=content, metadata=metadata)
        self.messages.append(msg)
        return msg

    def add_widget_response(self, content: str) -> ChatMessage:
        metadata = MessageMetadata(is_auto_advance=True)
        msg = ChatMessage(role="user", content=content, metadata=metadata)
        self.messages.append(msg)
        return msg

    def add_widget_response_card(
        self,
        widget_type: str,
        widget_data: dict,
        content: str = "",
    ) -> ChatMessage:
        metadata = MessageMetadata(
            widget_type=widget_type,
            widget_data=widget_data,
        )
        msg = ChatMessage(role="widget_response", content=content, metadata=metadata)
        self.messages.append(msg)
        return msg

    def add_task_ref(
        self,
        task_id: str,
        label: str,
        tool_name: str,
        multi_task_id: str | None = None,
        status: str = "queued",
    ) -> ChatMessage:
        metadata = MessageMetadata(
            is_task_ref=True, tool_name=tool_name, task_status=status
        )
        msg = ChatMessage(
            role="task_ref",
            content=label,
            metadata=metadata,
            task_id=task_id,
            multi_task_id=multi_task_id,
        )
        self.messages.append(msg)
        return msg

    def update_task_ref_status(self, task_id: str, status: str) -> bool:
        for msg in reversed(self.messages):
            if msg.role == "task_ref" and msg.task_id == task_id:
                if msg.metadata is None:
                    msg.metadata = MessageMetadata(is_task_ref=True)
                msg.metadata.task_status = status
                return True
        return False

    def get_api_messages(self) -> list[dict[str, str]]:
        """Convert history to API format, excluding UI-only roles."""
        ui_only_roles = {"task_ref", "widget_response"}
        return [m.to_api_dict() for m in self.messages if m.role not in ui_only_roles]

    def clear(self) -> None:
        self.messages.clear()

    def to_dict(self) -> dict:
        return {
            "system_prompt": self.system_prompt,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Conversation:
        conv = cls(system_prompt=data["system_prompt"])
        conv.messages = [ChatMessage.from_dict(m) for m in data.get("messages", [])]
        return conv
