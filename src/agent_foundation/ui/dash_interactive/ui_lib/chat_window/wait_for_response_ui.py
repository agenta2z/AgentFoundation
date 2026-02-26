"""
UI elements for displaying waiting/loading states in chat interfaces.

This module provides typing indicators and loading animations similar to
popular AI chat platforms like ChatGPT and Claude.
"""

from typing import Optional
from dash import html


def create_typing_indicator(
    label: str = "Thinking",
    label_color: str = "#8E8EA0",
    dot_char: str = "•",
    dot_count: int = 3,
    dot_size: str = "20px",
    dot_spacing: str = "2px",
    animation_duration: str = "1.3s",
    animation_delay_increment: float = 0.15,
    animation_name: str = "wave",
    container_style: Optional[dict] = None
) -> html.Div:
    """
    Create an animated typing indicator with bouncing dots.

    This creates a ChatGPT/Claude-style loading animation with a label and
    sequentially animated dots that create a wave effect.

    Args:
        label: Text to display before the dots (e.g., "Thinking", "Typing", "Loading")
        label_color: CSS color for the label text
        dot_char: Character to use for dots (default: bullet •)
        dot_count: Number of dots to display (default: 3)
        dot_size: Font size for the dots
        dot_spacing: Margin between dots
        animation_duration: CSS animation duration (e.g., "1.3s")
        animation_delay_increment: Delay between each dot's animation start (in seconds)
        animation_name: CSS animation keyframe name (must match CSS definition)
        container_style: Optional additional styles for the container div

    Returns:
        html.Div containing the animated typing indicator

    Example:
        >>> # Default ChatGPT-style indicator
        >>> indicator = create_typing_indicator()

        >>> # Custom "Loading" indicator
        >>> indicator = create_typing_indicator(
        ...     label="Loading",
        ...     label_color="#19C37D",
        ...     dot_count=4,
        ...     animation_duration="1s"
        ... )

    Note:
        Requires CSS keyframe animation named by `animation_name` parameter.
        Default "wave" animation should be defined in the app's CSS:

        @keyframes wave {
            0%, 60%, 100% {
                transform: translateY(0);
                opacity: 0.7;
            }
            30% {
                transform: translateY(-10px);
                opacity: 1;
            }
        }
    """
    # Create the animated dots
    dots = []
    for i in range(dot_count):
        delay = f"{i * animation_delay_increment}s"
        dot = html.Span(
            dot_char,
            className='dot',
            style={
                'animation': f'{animation_name} {animation_duration} ease-in-out infinite',
                'animationDelay': delay,
                'fontSize': dot_size,
                'margin': f'0 {dot_spacing}',
                'display': 'inline-block'
            }
        )
        dots.append(dot)

    # Default container style
    default_style = {
        'display': 'flex',
        'alignItems': 'center'
    }

    # Merge with custom style if provided
    if container_style:
        default_style.update(container_style)

    # Create the complete indicator
    return html.Div([
        html.Span(
            label,
            style={
                'marginRight': '8px',
                'color': label_color
            }
        ),
        html.Span(
            className='typing-indicator',
            children=dots,
            style={'display': 'inline-block'}
        )
    ], style=default_style)


def create_pulsing_indicator(
    text: str = "Loading...",
    text_color: str = "#8E8EA0",
    animation_name: str = "pulse",
    animation_duration: str = "1.5s"
) -> html.Div:
    """
    Create a simple pulsing text indicator.

    Args:
        text: Text to display
        text_color: CSS color for the text
        animation_name: CSS animation keyframe name
        animation_duration: CSS animation duration

    Returns:
        html.Div containing the pulsing text

    Example:
        >>> indicator = create_pulsing_indicator("Processing...")

    Note:
        Requires CSS keyframe animation for pulsing effect.
    """
    return html.Div(
        text,
        style={
            'color': text_color,
            'animation': f'{animation_name} {animation_duration} ease-in-out infinite'
        }
    )


def create_spinner_with_text(
    text: str = "Loading",
    text_color: str = "#8E8EA0",
    spinner_char: str = "◐",
    spinner_size: str = "20px",
    animation_name: str = "spin",
    animation_duration: str = "1s"
) -> html.Div:
    """
    Create a spinning character with text label.

    Args:
        text: Text to display next to spinner
        text_color: CSS color for the text
        spinner_char: Character to use for spinner (e.g., ◐, ◑, ◒, ◓, ⟳, ↻)
        spinner_size: Font size for the spinner
        animation_name: CSS animation keyframe name
        animation_duration: CSS animation duration

    Returns:
        html.Div containing the spinner and text

    Example:
        >>> indicator = create_spinner_with_text("Processing", spinner_char="⟳")

    Note:
        Requires CSS keyframe animation for rotation:

        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
    """
    return html.Div([
        html.Span(
            spinner_char,
            style={
                'fontSize': spinner_size,
                'marginRight': '8px',
                'display': 'inline-block',
                'animation': f'{animation_name} {animation_duration} linear infinite'
            }
        ),
        html.Span(
            text,
            style={'color': text_color}
        )
    ], style={'display': 'flex', 'alignItems': 'center'})
