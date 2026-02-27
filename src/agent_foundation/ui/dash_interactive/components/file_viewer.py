"""
File viewer panel component for displaying file contents in a slide-out panel.
"""

from typing import Any, Dict, List, Optional

from dash import dcc, html
from dash.dependencies import Input, Output, State

from agent_foundation.ui.dash_interactive.components.base import BaseComponent


class FileViewerPanel(BaseComponent):
    """
    A slide-out panel component for viewing file contents.

    This component provides a right-side panel that can display markdown
    or text file contents with syntax highlighting and proper formatting.

    Attributes:
        component_id (str): Unique identifier for this component
        panel_width (str): Width of the panel (default "50%")
    """

    def __init__(
        self,
        component_id: str = "file-viewer",
        panel_width: str = "50%",
        style: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the file viewer panel component.

        Args:
            component_id: Unique identifier for this component
            panel_width: Width of the panel when open (CSS value)
            style: Optional CSS style overrides
        """
        # Set panel_width before calling super().__init__ since _get_default_style uses it
        self.panel_width = panel_width
        super().__init__(component_id, style)

    def _get_default_style(self) -> Dict[str, Any]:
        """Get default styling for the file viewer panel."""
        return {
            "position": "fixed",
            "top": "0",
            "right": f"-{self.panel_width}",
            "width": self.panel_width,
            "height": "100vh",
            "backgroundColor": "#1e1e1e",
            "borderLeft": "1px solid #333",
            "zIndex": "1000",
            "transition": "right 0.3s ease-in-out",
            "display": "flex",
            "flexDirection": "column",
            "overflow": "hidden",
        }

    def layout(self) -> html.Div:
        """
        Generate the file viewer panel layout.

        Returns:
            Dash Div containing the panel with header and content area
        """
        return html.Div(
            id=self.get_id("panel"),
            className="file-viewer-panel",
            style=self.style,
            children=[
                # Header
                self._create_header(),
                # Content area
                self._create_content_area(),
                # Store for panel state
                dcc.Store(
                    id=self.get_id("store"),
                    data={"open": False, "path": None, "content": "", "title": ""},
                ),
            ],
        )

    def _create_header(self) -> html.Div:
        """Create the panel header with title and close button."""
        return html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "padding": "15px 20px",
                "borderBottom": "1px solid #333",
                "backgroundColor": "#252526",
                "flexShrink": "0",
            },
            children=[
                html.H3(
                    id=self.get_id("title"),
                    children="File Viewer",
                    style={
                        "margin": "0",
                        "color": "#fff",
                        "fontSize": "16px",
                        "fontWeight": "500",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                        "whiteSpace": "nowrap",
                        "flex": "1",
                    },
                ),
                html.Button(
                    "✕",
                    id=self.get_id("close-btn"),
                    n_clicks=0,
                    style={
                        "background": "none",
                        "border": "none",
                        "color": "#fff",
                        "fontSize": "20px",
                        "cursor": "pointer",
                        "padding": "5px 10px",
                        "marginLeft": "10px",
                        "borderRadius": "4px",
                        "transition": "background-color 0.2s",
                    },
                ),
            ],
        )

    def _create_content_area(self) -> html.Div:
        """Create the scrollable content area."""
        return html.Div(
            id=self.get_id("content"),
            style={
                "flex": "1",
                "overflow": "auto",
                "padding": "20px",
                "color": "#d4d4d4",
                "fontFamily": "'SF Mono', Monaco, 'Cascadia Code', monospace",
                "fontSize": "13px",
                "lineHeight": "1.6",
            },
            children=[
                dcc.Markdown(
                    id=self.get_id("markdown"),
                    children="",
                    style={
                        "color": "#d4d4d4",
                        "backgroundColor": "transparent",
                    },
                    className="file-viewer-markdown",
                )
            ],
        )

    def get_callback_inputs(self) -> List[Input]:
        """Get list of callback inputs."""
        return [Input(self.get_id("close-btn"), "n_clicks")]

    def get_callback_outputs(self) -> List[Output]:
        """Get list of callback outputs."""
        return [
            Output(self.get_id("panel"), "style"),
            Output(self.get_id("title"), "children"),
            Output(self.get_id("markdown"), "children"),
            Output(self.get_id("store"), "data"),
        ]

    def get_callback_states(self) -> List[State]:
        """Get list of callback states."""
        return [State(self.get_id("store"), "data")]

    def get_open_style(self) -> Dict[str, Any]:
        """Get the style dict for open panel state."""
        return {**self.style, "right": "0"}

    def get_closed_style(self) -> Dict[str, Any]:
        """Get the style dict for closed panel state."""
        return {**self.style, "right": f"-{self.panel_width}"}

    @staticmethod
    def get_css_styles() -> str:
        """
        Get CSS styles for the file viewer panel.

        Returns:
            CSS string to be injected into the page
        """
        return """
        .file-viewer-panel::-webkit-scrollbar {
            width: 8px;
        }
        .file-viewer-panel::-webkit-scrollbar-track {
            background: #1e1e1e;
        }
        .file-viewer-panel::-webkit-scrollbar-thumb {
            background: #424242;
            border-radius: 4px;
        }
        .file-viewer-panel::-webkit-scrollbar-thumb:hover {
            background: #555;
        }

        .file-viewer-markdown h1,
        .file-viewer-markdown h2,
        .file-viewer-markdown h3,
        .file-viewer-markdown h4 {
            color: #e0e0e0;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }

        .file-viewer-markdown h1 { font-size: 1.8em; border-bottom: 1px solid #333; padding-bottom: 0.3em; }
        .file-viewer-markdown h2 { font-size: 1.5em; border-bottom: 1px solid #333; padding-bottom: 0.3em; }
        .file-viewer-markdown h3 { font-size: 1.25em; }

        .file-viewer-markdown code {
            background-color: #2d2d2d;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-size: 0.9em;
        }

        .file-viewer-markdown pre {
            background-color: #2d2d2d;
            padding: 1em;
            border-radius: 6px;
            overflow-x: auto;
        }

        .file-viewer-markdown pre code {
            background-color: transparent;
            padding: 0;
        }

        .file-viewer-markdown ul, .file-viewer-markdown ol {
            padding-left: 1.5em;
            margin: 0.5em 0;
        }

        .file-viewer-markdown li {
            margin: 0.3em 0;
        }

        .file-viewer-markdown table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }

        .file-viewer-markdown th, .file-viewer-markdown td {
            border: 1px solid #444;
            padding: 8px 12px;
            text-align: left;
        }

        .file-viewer-markdown th {
            background-color: #2d2d2d;
        }

        .file-viewer-markdown blockquote {
            border-left: 3px solid #444;
            margin: 1em 0;
            padding-left: 1em;
            color: #aaa;
        }

        .file-viewer-markdown a {
            color: #58a6ff;
        }

        .file-viewer-markdown hr {
            border: none;
            border-top: 1px solid #333;
            margin: 1.5em 0;
        }
        """


def create_view_file_button(
    file_path: str,
    button_id: str,
    button_text: str = "📄 View",
) -> html.Button:
    """
    Create a button that triggers file viewing.

    Args:
        file_path: Path to the file to view
        button_id: Unique ID for the button
        button_text: Text to display on the button

    Returns:
        Dash Button element
    """
    return html.Button(
        button_text,
        id={"type": "view-file-btn", "path": file_path, "index": button_id},
        n_clicks=0,
        style={
            "backgroundColor": "#2d5a88",
            "color": "#fff",
            "border": "none",
            "borderRadius": "4px",
            "padding": "4px 10px",
            "fontSize": "12px",
            "cursor": "pointer",
            "marginLeft": "10px",
            "transition": "background-color 0.2s",
        },
    )
