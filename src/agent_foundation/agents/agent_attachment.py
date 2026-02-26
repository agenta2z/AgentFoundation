from typing import Any, Callable

from attr import attrs, attrib


def default_agent_attachment_formatter(id: str, description: str, content: Any, show_content: bool) -> str:
    """
    Default formatter for agent attachments.

    Args:
        id: Unique identifier for the attachment
        description: Human-readable description of the attachment
        content: The actual content of the attachment
        show_content: Whether to include the content in the output

    Returns:
        XML-formatted string representation of the attachment
    """
    if show_content:
        return f""" <Attachment>
  <ID>{id}</ID>
  <Description>{description}</Description>
  <Content>{content}</Content>
 </Attachment>"""
    else:
        return f""" <Attachment>
  <ID>{id}</ID>
  <Description>{description}</Description>
 </Attachment>"""


@attrs(slots=True)
class AgentAttachment:
    """
    Represents an attachment that can be included with agent messages.

    Attachments provide a structured way to include additional context or data
    with agent interactions, with customizable formatting.

    Attributes:
        id: Unique identifier for the attachment
        description: Human-readable description of what the attachment contains
        content: The actual content (can be any type)
        formatter: Function to format the attachment for display
    """
    id: str = attrib()
    description: str = attrib()
    content: Any = attrib()
    formatter: Callable[[str, str, Any, bool], str] = attrib(default=default_agent_attachment_formatter)

    def __str__(self) -> str:
        """Returns a string representation without the full content."""
        return self.formatter(self.id, self.description, self.content, False)

    @property
    def text(self) -> str:
        """Alias for __str__. Returns string representation without full content."""
        return str(self)

    @property
    def full_text(self) -> str:
        """Returns a string representation including the full content."""
        return self.formatter(self.id, self.description, self.content, True)