"""
UI elements for chat input areas.

This module provides reusable input components for chat interfaces,
including text areas and send buttons.
"""

from typing import Optional
from dash import html, dcc


def create_chat_input_area(
    input_id: str,
    send_button_id: str,
    placeholder: str = "Type a message...",
    # Textarea styling
    input_width: str = "calc(100% - 50px)",
    input_height: str = "50px",
    input_min_height: str = "50px",
    input_max_height: str = "150px",
    input_bg_color: str = "#40414F",
    input_text_color: str = "#ECECF1",
    input_border_color: str = "#565869",
    input_border_radius: str = "8px",
    input_font_size: str = "15px",
    input_padding: str = "12px 16px",
    # Send button styling
    send_button_icon: str = "↑",
    send_button_size: str = "36px",
    send_button_bg_color: str = "#19C37D",
    send_button_text_color: str = "white",
    send_button_border_radius: str = "6px",
    send_button_font_size: str = "20px",
    send_button_position_right: str = "20px",
    send_button_position_bottom: str = "20px",
    # Container styling
    container_max_width: str = "800px",
    container_padding: str = "15px 20px",
    container_bg_color: str = "#343541",
    border_top_color: str = "#565869",
    container_gap: str = "10px"
) -> html.Div:
    """
    Create a chat input area with text box and send button.

    This creates a ChatGPT/Claude-style input area with a text area that
    auto-expands and a circular send button positioned in the corner.

    Args:
        input_id: HTML id for the textarea element
        send_button_id: HTML id for the send button element
        placeholder: Placeholder text for the textarea
        input_width: Width of the textarea
        input_height: Initial height of the textarea
        input_min_height: Minimum height of the textarea
        input_max_height: Maximum height (with auto-resize)
        input_bg_color: Background color of the textarea
        input_text_color: Text color in the textarea
        input_border_color: Border color of the textarea
        input_border_radius: Border radius for rounded corners
        input_font_size: Font size for text
        input_padding: Internal padding of the textarea
        send_button_icon: Icon/character for the send button (default: ↑)
        send_button_size: Size of the circular send button
        send_button_bg_color: Background color of send button
        send_button_text_color: Color of the icon
        send_button_border_radius: Border radius of send button
        send_button_font_size: Font size of the icon
        send_button_position_right: Right position offset
        send_button_position_bottom: Bottom position offset
        container_max_width: Maximum width of the input container
        container_padding: Padding around the input area
        container_bg_color: Background color of the container
        border_top_color: Color of the top border
        container_gap: Gap between elements

    Returns:
        html.Div containing the input area with textarea and send button

    Example:
        >>> # Default ChatGPT-style input
        >>> input_area = create_chat_input_area(
        ...     input_id='chat-input',
        ...     send_button_id='send-btn'
        ... )

        >>> # Custom styled input
        >>> input_area = create_chat_input_area(
        ...     input_id='chat-input',
        ...     send_button_id='send-btn',
        ...     placeholder="Ask me anything...",
        ...     send_button_bg_color="#0066FF",
        ...     container_bg_color="#FFFFFF"
        ... )
    """
    return html.Div(
        children=[
            html.Div(
                children=[
                    dcc.Textarea(
                        id=input_id,
                        placeholder=placeholder,
                        value='',
                        style={
                            'width': input_width,
                            'height': input_height,
                            'minHeight': input_min_height,
                            'maxHeight': input_max_height,
                            'padding': input_padding,
                            'backgroundColor': input_bg_color,
                            'color': input_text_color,
                            'border': f'1px solid {input_border_color}',
                            'borderRadius': input_border_radius,
                            'fontSize': input_font_size,
                            'resize': 'none',
                            'fontFamily': 'inherit',
                            'outline': 'none',
                            'boxSizing': 'border-box'
                        }
                    ),
                    html.Button(
                        send_button_icon,
                        id=send_button_id,
                        n_clicks=0,
                        style={
                            'position': 'absolute',
                            'right': send_button_position_right,
                            'bottom': send_button_position_bottom,
                            'width': send_button_size,
                            'height': send_button_size,
                            'backgroundColor': send_button_bg_color,
                            'color': send_button_text_color,
                            'border': 'none',
                            'borderRadius': send_button_border_radius,
                            'fontSize': send_button_font_size,
                            'cursor': 'pointer',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'transition': 'background-color 0.2s',
                            'flexShrink': '0'
                        }
                    )
                ],
                style={
                    'position': 'relative',
                    'padding': container_padding,
                    'maxWidth': container_max_width,
                    'margin': '0 auto',
                    'width': '100%',
                    'display': 'flex',
                    'alignItems': 'flex-end',
                    'gap': container_gap,
                    'boxSizing': 'border-box'
                }
            )
        ],
        style={
            'borderTop': f'1px solid {border_top_color}',
            'backgroundColor': container_bg_color,
            'flexShrink': '0'
        }
    )
