"""
UI elements for rendering chat messages.

This module provides reusable message bubble components for chat interfaces,
supporting different roles (user, assistant) and styling.
"""

from typing import Optional, Literal
from dash import html


def create_message_bubble(
    content: str,
    role: Literal["user", "assistant"] = "user",
    timestamp: Optional[str] = None,
    # User message styling
    user_bg_color: str = "#40414F",
    user_text_color: str = "#ECECF1",
    user_align: str = "flex-end",
    user_max_width: str = "80%",
    user_avatar: str = "👤",
    # Assistant message styling
    assistant_bg_color: str = "#444654",
    assistant_text_color: str = "#ECECF1",
    assistant_align: str = "flex-start",
    assistant_max_width: str = "100%",
    assistant_avatar: str = "🤖",
    # Common styling
    padding: str = "16px",
    border_radius: str = "8px",
    font_size: str = "15px",
    line_height: str = "1.6",
    avatar_size: str = "20px",
    avatar_margin: str = "12px",
    timestamp_font_size: str = "11px",
    timestamp_color: str = "#8E8EA0",
    timestamp_margin_top: str = "8px"
) -> html.Div:
    """
    Create a styled message bubble for chat interfaces.

    This creates ChatGPT/Claude-style message bubbles with different styling
    for user and assistant messages.

    Args:
        content: The message content (can be str or html element)
        role: Message role - "user" or "assistant"
        timestamp: Optional timestamp string to display
        user_bg_color: Background color for user messages
        user_text_color: Text color for user messages
        user_align: Alignment for user messages (flex-end = right)
        user_max_width: Maximum width for user messages
        user_avatar: Avatar emoji/character for user
        assistant_bg_color: Background color for assistant messages
        assistant_text_color: Text color for assistant messages
        assistant_align: Alignment for assistant messages (flex-start = left)
        assistant_max_width: Maximum width for assistant messages
        assistant_avatar: Avatar emoji/character for assistant
        padding: Internal padding of the message bubble
        border_radius: Border radius for rounded corners
        font_size: Font size for message text
        line_height: Line height for message text
        avatar_size: Font size for avatar emoji
        avatar_margin: Margin between avatar and message
        timestamp_font_size: Font size for timestamp
        timestamp_color: Color for timestamp
        timestamp_margin_top: Top margin for timestamp

    Returns:
        html.Div containing the styled message bubble

    Example:
        >>> # User message
        >>> msg = create_message_bubble(
        ...     content="Hello!",
        ...     role="user",
        ...     timestamp="10:30 AM"
        ... )

        >>> # Assistant message
        >>> msg = create_message_bubble(
        ...     content="Hi! How can I help?",
        ...     role="assistant",
        ...     timestamp="10:30 AM"
        ... )

        >>> # Custom styling
        >>> msg = create_message_bubble(
        ...     content="Custom message",
        ...     role="user",
        ...     user_bg_color="#0066FF",
        ...     user_avatar="😊"
        ... )
    """
    # Select styling based on role
    if role == "user":
        bg_color = user_bg_color
        text_color = user_text_color
        align = user_align
        max_width = user_max_width
        avatar = user_avatar
    else:  # assistant
        bg_color = assistant_bg_color
        text_color = assistant_text_color
        align = assistant_align
        max_width = assistant_max_width
        avatar = assistant_avatar

    # Create timestamp element if provided
    timestamp_element = None
    if timestamp:
        timestamp_element = html.Div(
            timestamp,
            style={
                'fontSize': timestamp_font_size,
                'color': timestamp_color,
                'marginTop': timestamp_margin_top
            }
        )

    # Create the message bubble
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Div(
                        avatar,
                        style={
                            'fontSize': avatar_size,
                            'marginRight': avatar_margin,
                            'flexShrink': '0'
                        }
                    ),
                    html.Div(
                        children=[
                            html.Div(
                                content,
                                style={
                                    'whiteSpace': 'pre-wrap',
                                    'wordBreak': 'break-word',
                                    'lineHeight': line_height,
                                    'fontSize': font_size
                                }
                            ),
                            timestamp_element
                        ],
                        style={'flex': '1'}
                    )
                ],
                style={
                    'display': 'flex',
                    'padding': padding,
                    'backgroundColor': bg_color,
                    'borderRadius': border_radius,
                    'maxWidth': max_width,
                    'color': text_color
                }
            )
        ],
        style={
            'display': 'flex',
            'justifyContent': align,
            'width': '100%'
        }
    )


def create_welcome_message(
    title: str = "Welcome!",
    subtitle: Optional[str] = None,
    icon: str = "👋",
    title_color: str = "#ECECF1",
    subtitle_color: str = "#8E8EA0",
    title_font_size: str = "24px",
    subtitle_font_size: str = "15px",
    icon_size: str = "48px",
    spacing: str = "16px"
) -> html.Div:
    """
    Create a welcome message for empty chat state.

    Args:
        title: Main welcome text
        subtitle: Optional subtitle text
        icon: Icon/emoji to display
        title_color: Color of the title
        subtitle_color: Color of the subtitle
        title_font_size: Font size for title
        subtitle_font_size: Font size for subtitle
        icon_size: Font size for icon
        spacing: Spacing between elements

    Returns:
        html.Div containing the welcome message

    Example:
        >>> welcome = create_welcome_message(
        ...     title="Welcome to ChatBot",
        ...     subtitle="How can I help you today?"
        ... )
    """
    children = [
        html.Div(
            icon,
            style={
                'fontSize': icon_size,
                'marginBottom': spacing
            }
        ),
        html.Div(
            title,
            style={
                'fontSize': title_font_size,
                'fontWeight': 'bold',
                'color': title_color,
                'marginBottom': spacing if subtitle else '0'
            }
        )
    ]

    if subtitle:
        children.append(
            html.Div(
                subtitle,
                style={
                    'fontSize': subtitle_font_size,
                    'color': subtitle_color
                }
            )
        )

    return html.Div(
        children=children,
        style={
            'display': 'flex',
            'flexDirection': 'column',
            'justifyContent': 'center',
            'alignItems': 'center',
            'height': '100%',
            'textAlign': 'center',
            'padding': '40px'
        }
    )
