"""
Log graph visualization component for displaying hierarchical log structure.
"""
from typing import Any, Dict, List, Optional
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import dash_cytoscape as cyto

from agent_foundation.ui.dash_interactive.components.base import BaseComponent


class LogGraphVisualization(BaseComponent):
    """
    Component for visualizing hierarchical log groups as a graph.

    This component displays the hierarchical structure of log groups
    using an interactive tree/graph visualization.

    Attributes:
        component_id (str): Unique identifier for this component
        graph_data (Dict): Hierarchical graph structure
    """

    def __init__(
        self,
        component_id: str = "log-graph",
        graph_data: Optional[Dict[str, Any]] = None,
        style: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the log graph visualization component.

        Args:
            component_id: Unique identifier for this component
            graph_data: Hierarchical graph structure data
            style: Optional CSS style overrides
        """
        super().__init__(component_id, style)
        self.graph_data = graph_data or {}

    def _get_default_style(self) -> Dict[str, Any]:
        """Get default styling for the log graph."""
        return {
            'height': '100%',
            'backgroundColor': '#2C2C2C',
            'padding': '10px'
        }

    def layout(self) -> html.Div:
        """
        Generate the log graph layout with dual rendering options.

        Returns:
            Dash Div containing the graph visualization (Plotly or Cytoscape)
        """
        return html.Div(
            id=self.get_id(),
            children=[
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.H4(
                                    "Log Group Graph",
                                    style={
                                        'color': '#ECECF1',
                                        'margin': '0 0 10px 0',
                                        'fontSize': '16px',
                                        'fontWeight': '600',
                                        'display': 'inline-block',
                                        'marginRight': '20px'
                                    }
                                ),
                                # Removed old refresh button from here - now in floating panel
                                html.Div(
                                    children=[
                                        html.Label(
                                            'Rendering:',
                                            style={
                                                'color': '#ECECF1',
                                                'fontSize': '12px',
                                                'marginRight': '8px'
                                            }
                                        ),
                                        dcc.Dropdown(
                                            id=self.get_id('rendering-mode'),
                                            options=[
                                                {'label': 'Plotly (Static Pan/Zoom)', 'value': 'plotly'},
                                                {'label': 'Cytoscape (Draggable)', 'value': 'cytoscape'}
                                            ],
                                            value='plotly',  # Default to stable existing version
                                            clearable=False,
                                            style={
                                                'width': '220px',
                                                'fontSize': '12px'
                                            }
                                        )
                                    ],
                                    style={
                                        'display': 'inline-block',
                                        'verticalAlign': 'middle'
                                    }
                                ),
                                html.Div(
                                    children=[
                                        html.Label(
                                            'Label:',
                                            style={
                                                'color': '#ECECF1',
                                                'fontSize': '12px',
                                                'marginRight': '8px',
                                                'marginLeft': '20px'
                                            }
                                        ),
                                        dcc.Dropdown(
                                            id=self.get_id('label-mode'),
                                            options=[
                                                {'label': 'Log Name', 'value': 'name'},
                                                {'label': 'Node ID', 'value': 'id'}
                                            ],
                                            value='name',  # Default to log name
                                            clearable=False,
                                            style={
                                                'width': '150px',
                                                'fontSize': '12px'
                                            }
                                        )
                                    ],
                                    style={
                                        'display': 'inline-block',
                                        'verticalAlign': 'middle'
                                    }
                                )
                            ],
                            style={'marginBottom': '10px'}
                        ),
                        html.Div(
                            id=self.get_id('selected-info'),
                            style={
                                'color': '#8E8EA0',
                                'fontSize': '12px',
                                'marginBottom': '10px'
                            }
                        )
                    ],
                    style={'padding': '10px'}
                ),
                # Plotly Graph (existing, default visible)
                html.Div(
                    id=self.get_id('plotly-container'),
                    children=[
                        dcc.Graph(
                            id=self.get_id('graph'),
                            figure=self._create_figure(),
                            style={'height': '100%'},
                            config={
                                'displayModeBar': True,
                                'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'autoScale2d'],
                                'modeBarButtonsToAdd': ['pan2d', 'zoom2d'],
                                'scrollZoom': True
                            }
                        ),
                        # Loading overlay
                        html.Div(
                            id=self.get_id('plotly-loading-overlay'),
                            children=[
                                html.Div(
                                    "Loading log graph...",
                                    style={
                                        'color': '#ECECF1',
                                        'fontSize': '18px',
                                        'fontWeight': '500'
                                    }
                                )
                            ],
                            style={
                                'position': 'absolute',
                                'top': '0',
                                'left': '0',
                                'right': '0',
                                'bottom': '0',
                                'backgroundColor': 'rgba(44, 44, 44, 0.95)',
                                'display': 'none',  # Hidden by default
                                'alignItems': 'center',
                                'justifyContent': 'center',
                                'zIndex': '2000'  # Above graph
                            }
                        )
                    ],
                    style={'display': 'block', 'height': 'calc(100% - 120px)', 'position': 'relative'}
                ),
                # Cytoscape Graph (new, initially hidden)
                html.Div(
                    id=self.get_id('cytoscape-container'),
                    children=[
                        # Cytoscape graph component
                        cyto.Cytoscape(
                            id=self.get_id('cytoscape'),
                            elements=[],  # Will be populated by callback
                            layout={'name': 'preset'},  # Use our calculated positions
                            style={'width': '100%', 'height': '100%'},
                            stylesheet=self._get_cytoscape_stylesheet(),
                            zoom=1,
                            pan={'x': 0, 'y': 0},
                            autoungrabify=False,
                            userZoomingEnabled=True,
                            userPanningEnabled=True,
                            boxSelectionEnabled=False
                        ),
                        # Loading overlay
                        html.Div(
                            id=self.get_id('cytoscape-loading-overlay'),
                            children=[
                                html.Div(
                                    "Loading log graph...",
                                    style={
                                        'color': '#ECECF1',
                                        'fontSize': '18px',
                                        'fontWeight': '500'
                                    }
                                )
                            ],
                            style={
                                'position': 'absolute',
                                'top': '0',
                                'left': '0',
                                'right': '0',
                                'bottom': '0',
                                'backgroundColor': 'rgba(44, 44, 44, 0.95)',
                                'display': 'none',  # Hidden by default
                                'alignItems': 'center',
                                'justifyContent': 'center',
                                'zIndex': '2000'  # Above controls
                            }
                        ),
                        # Floating control buttons (Plotly-style modebar in top-right)
                        html.Div(
                            id=self.get_id('cytoscape-controls'),
                            children=[
                                html.Button(
                                    '⟲',
                                    id=self.get_id('cytoscape-reset-btn'),
                                    n_clicks=0,
                                    title='Reset view',
                                    style={
                                        'padding': '4px 6px',
                                        'backgroundColor': 'rgba(255, 255, 255, 0.05)',
                                        'color': '#ECECF1',
                                        'border': '1px solid rgba(255, 255, 255, 0.1)',
                                        'borderRadius': '2px',
                                        'cursor': 'pointer',
                                        'fontSize': '14px',
                                        'marginLeft': '2px',
                                        'transition': 'background-color 0.2s',
                                        'minWidth': '24px',
                                        'height': '24px',
                                        'display': 'inline-flex',
                                        'alignItems': 'center',
                                        'justifyContent': 'center'
                                    }
                                ),
                                html.Button(
                                    '⊡',
                                    id=self.get_id('cytoscape-fit-btn'),
                                    n_clicks=0,
                                    title='Fit to screen',
                                    style={
                                        'padding': '4px 6px',
                                        'backgroundColor': 'rgba(255, 255, 255, 0.05)',
                                        'color': '#ECECF1',
                                        'border': '1px solid rgba(255, 255, 255, 0.1)',
                                        'borderRadius': '2px',
                                        'cursor': 'pointer',
                                        'fontSize': '14px',
                                        'marginLeft': '2px',
                                        'transition': 'background-color 0.2s',
                                        'minWidth': '24px',
                                        'height': '24px',
                                        'display': 'inline-flex',
                                        'alignItems': 'center',
                                        'justifyContent': 'center'
                                    }
                                ),
                                html.Button(
                                    '📷',
                                    id=self.get_id('cytoscape-download-png-btn'),
                                    n_clicks=0,
                                    title='Download PNG',
                                    style={
                                        'padding': '4px 6px',
                                        'backgroundColor': 'rgba(255, 255, 255, 0.05)',
                                        'color': '#ECECF1',
                                        'border': '1px solid rgba(255, 255, 255, 0.1)',
                                        'borderRadius': '2px',
                                        'cursor': 'pointer',
                                        'fontSize': '12px',
                                        'marginLeft': '2px',
                                        'transition': 'background-color 0.2s',
                                        'minWidth': '24px',
                                        'height': '24px',
                                        'display': 'inline-flex',
                                        'alignItems': 'center',
                                        'justifyContent': 'center'
                                    }
                                ),
                                html.Button(
                                    '{ }',
                                    id=self.get_id('cytoscape-download-json-btn'),
                                    n_clicks=0,
                                    title='Download JSON',
                                    style={
                                        'padding': '4px 6px',
                                        'backgroundColor': 'rgba(255, 255, 255, 0.05)',
                                        'color': '#ECECF1',
                                        'border': '1px solid rgba(255, 255, 255, 0.1)',
                                        'borderRadius': '2px',
                                        'cursor': 'pointer',
                                        'fontSize': '10px',
                                        'marginLeft': '2px',
                                        'transition': 'background-color 0.2s',
                                        'minWidth': '24px',
                                        'height': '24px',
                                        'display': 'inline-flex',
                                        'alignItems': 'center',
                                        'justifyContent': 'center',
                                        'fontFamily': 'monospace'
                                    }
                                )
                            ],
                            style={
                                'position': 'absolute',
                                'top': '10px',
                                'right': '10px',
                                'display': 'flex',
                                'alignItems': 'center',
                                'backgroundColor': 'rgba(44, 44, 44, 0.8)',
                                'backdropFilter': 'blur(4px)',
                                'borderRadius': '3px',
                                'padding': '3px',
                                'boxShadow': '0 2px 4px rgba(0,0,0,0.3)',
                                'zIndex': '1000'
                            }
                        )
                    ],
                    style={'display': 'none', 'height': 'calc(100% - 120px)', 'position': 'relative'}
                )
            ],
            style=self.style
        )

    def _create_figure(self, label_mode: str = 'name') -> go.Figure:
        """
        Create the Plotly figure for the graph visualization.

        Args:
            label_mode: Either 'name' for log name or 'id' for node ID

        Returns:
            Plotly Figure object
        """
        if not self.graph_data or 'nodes' not in self.graph_data:
            return self._create_empty_figure()

        nodes = self.graph_data['nodes']
        edges = self.graph_data.get('edges', [])

        # Separate nodes by type
        normal_nodes = [node for node in nodes if node.get('node_type', 'normal') == 'normal']
        exit_nodes = [node for node in nodes if node.get('node_type', 'normal') == 'exit']

        node_traces = []

        # Create trace for normal nodes
        if normal_nodes:
            normal_x = [node['x'] for node in normal_nodes]
            normal_y = [node['y'] for node in normal_nodes]
            normal_labels = [node['id'] if label_mode == 'id' else node['name'] for node in normal_nodes]
            normal_text = [
                f"{label}<br>Logs: {node['log_count']}"
                for label, node in zip(normal_labels, normal_nodes)
            ]
            normal_customdata = [[node['id']] for node in normal_nodes]

            normal_trace = go.Scatter(
                x=normal_x,
                y=normal_y,
                mode='markers+text',
                marker=dict(
                    size=30,
                    color='#19C37D',  # Green for normal nodes
                    line=dict(color='#ECECF1', width=2)
                ),
                text=normal_labels,
                textposition='top center',
                textfont=dict(color='#ECECF1', size=12),
                hovertext=normal_text,
                hoverinfo='text',
                customdata=normal_customdata,
                name='Nodes'
            )
            node_traces.append(normal_trace)

        # Create trace for exit nodes
        if exit_nodes:
            exit_x = [node['x'] for node in exit_nodes]
            exit_y = [node['y'] for node in exit_nodes]
            exit_labels = [node['id'] if label_mode == 'id' else node['name'] for node in exit_nodes]
            exit_text = [
                f"{label}<br>Exit Point"
                for label in exit_labels
            ]
            exit_customdata = [[node['id']] for node in exit_nodes]

            exit_trace = go.Scatter(
                x=exit_x,
                y=exit_y,
                mode='markers+text',
                marker=dict(
                    size=35,
                    color='#FF6B6B',  # Red/coral for exit nodes
                    symbol='diamond',  # Different symbol for exit nodes
                    line=dict(color='#FFD93D', width=2)  # Yellow border
                ),
                text=exit_labels,
                textposition='top center',
                textfont=dict(color='#ECECF1', size=12),
                hovertext=exit_text,
                hoverinfo='text',
                customdata=exit_customdata,
                name='Exit Points'
            )
            node_traces.append(exit_trace)

        # Create edge traces with arrows
        edge_traces = []
        edge_annotations = []

        for edge in edges:
            x0, y0 = edge['source_pos']
            x1, y1 = edge['target_pos']

            # Calculate the distance and direction
            dx = x1 - x0
            dy = y1 - y0
            distance = (dx**2 + dy**2)**0.5

            if distance > 0:
                # Node radius in data units (approximation: marker size 30px ~ 18 data units)
                # Shorten from both source and target
                node_radius = 18

                # Shorten from source node
                source_ratio = node_radius / distance
                edge_start_x = x0 + dx * source_ratio
                edge_start_y = y0 + dy * source_ratio

                # Shorten from target node
                target_ratio = (distance - node_radius) / distance
                edge_end_x = x0 + dx * target_ratio
                edge_end_y = y0 + dy * target_ratio

                # Position arrow slightly before edge end
                arrow_gap = 8  # Small gap before the node
                arrow_ratio = (distance - node_radius - arrow_gap) / distance
                arrow_x = x0 + dx * arrow_ratio
                arrow_y = y0 + dy * arrow_ratio
            else:
                edge_start_x, edge_start_y = x0, y0
                edge_end_x, edge_end_y = x1, y1
                arrow_x, arrow_y = x1, y1

            # Create line trace (shortened from both ends to touch nodes)
            edge_trace = go.Scatter(
                x=[edge_start_x, edge_end_x, None],
                y=[edge_start_y, edge_end_y, None],
                mode='lines',
                line=dict(color='#565869', width=2),
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)

            # Add arrow annotation
            edge_annotations.append(
                dict(
                    x=edge_end_x,
                    y=edge_end_y,
                    ax=arrow_x,
                    ay=arrow_y,
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor='#565869'
                )
            )

        # Combine traces
        data = edge_traces + node_traces

        # Create layout with arrow annotations
        layout = go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=20),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                fixedrange=False  # Allow panning/zooming
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                fixedrange=False  # Allow panning/zooming
            ),
            plot_bgcolor='#2C2C2C',
            paper_bgcolor='#2C2C2C',
            height=400,
            annotations=edge_annotations,
            dragmode='pan'  # Default to pan mode
        )

        return go.Figure(data=data, layout=layout)

    def _create_empty_figure(self) -> go.Figure:
        """Create an empty placeholder figure."""
        layout = go.Layout(
            showlegend=False,
            margin=dict(b=20, l=20, r=20, t=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#2C2C2C',
            paper_bgcolor='#2C2C2C',
            height=400,
            annotations=[
                dict(
                    text="No log data available<br>Run a graph to see logs",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color='#8E8EA0', size=14)
                )
            ]
        )
        return go.Figure(layout=layout)

    def get_callback_inputs(self) -> List[Input]:
        """Get list of callback inputs."""
        return [
            Input(self.get_id('graph'), 'clickData')
        ]

    def get_callback_outputs(self) -> List[Output]:
        """Get list of callback outputs."""
        return [
            Output(self.get_id('graph'), 'figure'),
            Output(self.get_id('selected-info'), 'children')
        ]

    def update_graph(self, graph_data: Dict[str, Any], label_mode: str = 'name') -> go.Figure:
        """
        Update the graph with new data.

        Args:
            graph_data: New graph structure data
            label_mode: Either 'name' for log name or 'id' for node ID

        Returns:
            Updated Plotly Figure
        """
        self.graph_data = graph_data
        return self._create_figure(label_mode)

    def create_figure(self, hierarchy: List[Dict[str, Any]], label_mode: str = 'name') -> go.Figure:
        """
        Create a figure from hierarchical log structure.

        Args:
            hierarchy: Hierarchical structure from LogCollector
            label_mode: Either 'name' for log name or 'id' for node ID

        Returns:
            Plotly Figure object
        """
        if not hierarchy:
            return self._create_empty_figure()

        # Convert hierarchy to graph data
        graph_data = self.process_hierarchy_to_graph(hierarchy)
        self.graph_data = graph_data
        return self._create_figure(label_mode)

    def create_figure_from_graph(self, graph_data: Dict[str, Any], label_mode: str = 'name') -> go.Figure:
        """
        Create a figure directly from graph data (nodes and edges).

        This method is used for WorkGraph structures where we have a DAG
        with convergence nodes, rather than a pure tree hierarchy.

        Args:
            graph_data: Dictionary with 'nodes', 'edges', and 'agent' information
            label_mode: Either 'name' for log name or 'id' for node ID

        Returns:
            Plotly Figure object
        """
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        agent = graph_data.get('agent', {})

        if not nodes:
            return self._create_empty_figure()

        # Process nodes and edges for visualization using DAG layout
        processed_graph_data = self._process_dag_to_graph(nodes, edges, agent)
        self.graph_data = processed_graph_data
        return self._create_figure(label_mode)

    @staticmethod
    def _process_dag_to_graph(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], agent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process WorkGraph DAG structure into visualization data with proper layout.

        Treats all nodes equally, including the agent node. No fake edges are created.
        The graph structure reflects the actual parent-child relationships from the logs.

        Args:
            nodes: List of WorkGraph nodes with id, label, log_count
            edges: List of edges with source and target
            agent: Agent information (kept for API compatibility but not used for special treatment)

        Returns:
            Dictionary with nodes and edges for visualization
        """
        # Build node map and adjacency lists
        node_map = {node['id']: node for node in nodes}
        children_map = {}
        parents_map = {}

        for edge in edges:
            parent_id = edge['source']
            child_id = edge['target']

            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(child_id)

            if child_id not in parents_map:
                parents_map[child_id] = []
            parents_map[child_id].append(parent_id)

        # Find root nodes (no parents) and assign levels using BFS
        root_nodes = [node_id for node_id in node_map.keys() if node_id not in parents_map]
        node_levels = {}
        queue = [(node_id, 0) for node_id in root_nodes]  # Start at level 0

        while queue:
            node_id, level = queue.pop(0)
            if node_id in node_levels:
                node_levels[node_id] = max(node_levels[node_id], level)  # Take max level for convergence
            else:
                node_levels[node_id] = level
                for child_id in children_map.get(node_id, []):
                    queue.append((child_id, level + 1))

        # Group nodes by level
        level_groups = {}
        for node_id, level in node_levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node_id)

        # Calculate positions for all nodes
        vis_nodes = []
        node_positions = {}

        for level in sorted(level_groups.keys()):
            level_nodes = level_groups[level]
            node_spacing = 300
            start_x = -(len(level_nodes) - 1) * node_spacing / 2

            for i, node_id in enumerate(level_nodes):
                x = start_x + i * node_spacing
                y = -level * 120  # Vertical spacing

                node = node_map[node_id]
                vis_nodes.append({
                    'id': node['id'],
                    'name': node.get('label', node.get('name', node_id)),
                    'log_count': node['log_count'],
                    'x': x,
                    'y': y,
                    'level': level,
                    'node_type': node.get('node_type', 'normal')  # Preserve node type
                })
                node_positions[node_id] = (x, y)

        # Create edges - ONLY the real edges from the input
        vis_edges = []
        for edge in edges:
            source_id = edge['source']
            target_id = edge['target']
            vis_edges.append({
                'source': source_id,
                'target': target_id,
                'source_pos': node_positions[source_id],
                'target_pos': node_positions[target_id]
            })

        return {
            'nodes': vis_nodes,
            'edges': vis_edges
        }

    @staticmethod
    def process_hierarchy_to_graph(hierarchy: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process hierarchical log structure into graph visualization data.

        Args:
            hierarchy: Hierarchical structure from LogCollector

        Returns:
            Dictionary with nodes and edges for visualization
        """
        nodes = []
        edges = []
        node_positions = {}

        def _process_node(node, level=0, position=0, parent_pos=None):
            """Recursively process nodes and assign positions."""
            node_id = node['id']
            node_name = node['name'] if node.get('name') else node['log_group_id']

            # Calculate position
            x = position
            y = -level * 100  # Vertical spacing

            node_positions[node_id] = (x, y)

            nodes.append({
                'id': node_id,
                'name': node_name,
                'log_count': node.get('log_count', 0),
                'x': x,
                'y': y,
                'level': level
            })

            # Create edge from parent
            if parent_pos:
                edges.append({
                    'source': node.get('parent_id'),
                    'target': node_id,
                    'source_pos': parent_pos,
                    'target_pos': (x, y)
                })

            # Process children
            children = node.get('children', [])
            child_spacing = 200
            start_x = x - (len(children) - 1) * child_spacing / 2

            for i, child in enumerate(children):
                child_x = start_x + i * child_spacing
                _process_node(child, level + 1, child_x, (x, y))

        # Process root nodes
        root_spacing = 300
        for i, root in enumerate(hierarchy):
            _process_node(root, 0, i * root_spacing)

        return {
            'nodes': nodes,
            'edges': edges
        }

    def _get_cytoscape_stylesheet(self) -> List[Dict[str, Any]]:
        """
        Get Cytoscape stylesheet to match current Plotly visual appearance.

        Returns:
            List of stylesheet dictionaries for Cytoscape
        """
        return [
            # Node styling (default/normal nodes)
            {
                'selector': 'node',
                'style': {
                    'content': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'background-color': '#19C37D',
                    'color': '#ECECF1',
                    'font-size': '12px',
                    'width': '30px',
                    'height': '30px',
                    'border-width': '2px',
                    'border-color': '#ECECF1',
                    'text-wrap': 'wrap',
                    'text-max-width': '100px'
                }
            },
            # Exit node styling (red/orange color)
            {
                'selector': 'node.exit',
                'style': {
                    'background-color': '#FF6B6B',  # Red/coral color for exit nodes
                    'border-color': '#FFD93D',       # Yellow border for visibility
                    'shape': 'octagon',              # Different shape to distinguish
                    'width': '35px',
                    'height': '35px'
                }
            },
            # Node hover effect
            {
                'selector': 'node:selected',
                'style': {
                    'background-color': '#1DAC71',
                    'border-color': '#FFFFFF',
                    'border-width': '3px'
                }
            },
            # Exit node selected effect
            {
                'selector': 'node.exit:selected',
                'style': {
                    'background-color': '#FF5252',
                    'border-color': '#FFFFFF',
                    'border-width': '3px'
                }
            },
            # Edge styling with arrows
            {
                'selector': 'edge',
                'style': {
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    'target-arrow-color': '#565869',
                    'line-color': '#565869',
                    'width': 2,
                    'arrow-scale': 1.5
                }
            },
            # Edge hover effect
            {
                'selector': 'edge:selected',
                'style': {
                    'line-color': '#7A7A8A',
                    'target-arrow-color': '#7A7A8A'
                }
            }
        ]

    def convert_to_cytoscape_elements(self, graph_data: Dict[str, Any], label_mode: str = 'name') -> List[Dict[str, Any]]:
        """
        Convert graph data to Cytoscape elements format.

        Args:
            graph_data: Dictionary with 'nodes' and 'edges' from visualization
            label_mode: Either 'name' for log name or 'id' for node ID

        Returns:
            List of Cytoscape element dictionaries
        """
        if not graph_data or 'nodes' not in graph_data:
            return []

        elements = []
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])

        # Convert nodes
        for node in nodes:
            # Create label based on label_mode
            label_text = node['id'] if label_mode == 'id' else node.get('name', '')
            label = f"{label_text}"
            if node.get('log_count', 0) > 0:
                label += f"\n({node['log_count']} logs)"

            elements.append({
                'data': {
                    'id': node['id'],
                    'label': label,
                    'log_count': node.get('log_count', 0),
                    'name': node.get('name', ''),
                    'node_type': node.get('node_type', 'normal')  # Include node type for styling
                },
                'position': {
                    'x': node.get('x', 0),
                    'y': -node.get('y', 0)  # Invert Y to match Plotly orientation
                },
                'classes': node.get('node_type', 'normal')  # Add class for CSS-like styling
            })

        # Convert edges
        for edge in edges:
            elements.append({
                'data': {
                    'source': edge['source'],
                    'target': edge['target']
                }
            })

        return elements
