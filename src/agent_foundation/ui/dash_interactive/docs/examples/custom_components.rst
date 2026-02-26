Custom Components Example
==========================

This example demonstrates how to extend the framework by creating custom
components and integrating them with the application.

.. contents:: On This Page
   :local:
   :depth: 2

Creating a Custom Component
---------------------------

Extend ``BaseComponent`` to create custom UI elements:

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive.components.base import BaseComponent
   from dash import html, dcc
   from dash.dependencies import Input, Output
   from typing import Dict, Any, List

   class TokenCounter(BaseComponent):
       """Custom component that displays token usage."""

       def __init__(self, component_id: str = "token-counter", style: Dict = None):
           super().__init__(component_id, style)
           self.total_tokens = 0

       def _get_default_style(self) -> Dict[str, Any]:
           return {
               'padding': '10px',
               'backgroundColor': '#2C2C2C',
               'borderRadius': '8px',
               'color': '#ECECF1'
           }

       def layout(self) -> html.Div:
           return html.Div(
               id=self.get_id(),
               children=[
                   html.H4("Token Usage", style={'margin': '0 0 10px 0'}),
                   html.Div(
                       id=self.get_id('display'),
                       children=f"Total: {self.total_tokens}",
                       style={'fontSize': '24px', 'fontWeight': 'bold'}
                   ),
                   html.Button(
                       "Reset",
                       id=self.get_id('reset-btn'),
                       n_clicks=0,
                       style={
                           'marginTop': '10px',
                           'padding': '5px 15px',
                           'backgroundColor': '#19C37D',
                           'color': 'white',
                           'border': 'none',
                           'borderRadius': '4px',
                           'cursor': 'pointer'
                       }
                   )
               ],
               style=self.style
           )

       def get_callback_inputs(self) -> List[Input]:
           return [Input(self.get_id('reset-btn'), 'n_clicks')]

       def get_callback_outputs(self) -> List[Output]:
           return [Output(self.get_id('display'), 'children')]

Custom Monitor Tab
------------------

Add a custom tab to the monitor panel:

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive import DashInteractiveAppWithLogs
   from dash import html, dcc
   from dash.dependencies import Input, Output, State

   class MyAppWithSettings(DashInteractiveAppWithLogs):
       """App with custom settings in monitor panel."""

       def __init__(self, **kwargs):
           # Create custom tab content BEFORE parent init
           custom_tabs = [
               {
                   'id': 'settings',
                   'label': '⚙️ Settings',
                   'content': self._create_settings_content()
               }
           ]

           # Store reference for callbacks
           self._settings_content = custom_tabs[0]['content']

           super().__init__(custom_monitor_tabs=custom_tabs, **kwargs)

           # Register custom callbacks
           self._register_settings_callbacks()

       def _create_settings_content(self):
           """Create settings panel content."""
           return html.Div([
               html.Div([
                   html.Label("Temperature", style={'fontSize': '10px', 'color': '#8E8EA0'}),
                   dcc.Slider(
                       id='settings-temperature',
                       min=0,
                       max=1,
                       step=0.1,
                       value=0.7,
                       marks={0: '0', 0.5: '0.5', 1: '1'}
                   )
               ], style={'marginBottom': '15px'}),

               html.Div([
                   html.Label("Model", style={'fontSize': '10px', 'color': '#8E8EA0'}),
                   dcc.Dropdown(
                       id='settings-model',
                       options=[
                           {'label': 'GPT-4', 'value': 'gpt-4'},
                           {'label': 'GPT-3.5', 'value': 'gpt-3.5-turbo'},
                           {'label': 'Claude', 'value': 'claude-2'}
                       ],
                       value='gpt-4',
                       style={'backgroundColor': '#40414F', 'color': '#ECECF1'}
                   )
               ], style={'marginBottom': '15px'}),

               html.Div([
                   html.Label("Max Tokens", style={'fontSize': '10px', 'color': '#8E8EA0'}),
                   dcc.Input(
                       id='settings-max-tokens',
                       type='number',
                       value=500,
                       style={
                           'width': '100%',
                           'backgroundColor': '#40414F',
                           'color': '#ECECF1',
                           'border': '1px solid #565869',
                           'borderRadius': '4px',
                           'padding': '5px'
                       }
                   )
               ], style={'marginBottom': '15px'}),

               html.Div(
                   id='settings-status',
                   children='Settings ready',
                   style={'fontSize': '9px', 'color': '#19C37D'}
               )
           ], style={'padding': '10px'})

       def _register_settings_callbacks(self):
           """Register callbacks for settings."""
           @self.app.callback(
               Output('settings-status', 'children'),
               [Input('settings-temperature', 'value'),
                Input('settings-model', 'value'),
                Input('settings-max-tokens', 'value')]
           )
           def update_settings(temp, model, tokens):
               return f"✓ {model} | temp={temp} | max={tokens}"

   if __name__ == '__main__':
       app = MyAppWithSettings(title="Custom Settings Demo")
       app.run()

Custom Main Tab
---------------

Add a custom tab to the main panel:

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive import DashInteractiveAppWithLogs
   from dash import html, dcc
   import plotly.graph_objs as go

   def create_metrics_tab():
       """Create a metrics dashboard tab."""
       return html.Div([
           html.H3("Metrics Dashboard", style={'color': '#ECECF1', 'marginBottom': '20px'}),

           html.Div([
               html.Div([
                   html.H4("Messages", style={'color': '#8E8EA0', 'fontSize': '12px'}),
                   html.Div("0", id='metrics-messages',
                           style={'fontSize': '36px', 'color': '#19C37D', 'fontWeight': 'bold'})
               ], style={'flex': '1', 'textAlign': 'center'}),

               html.Div([
                   html.H4("Tokens", style={'color': '#8E8EA0', 'fontSize': '12px'}),
                   html.Div("0", id='metrics-tokens',
                           style={'fontSize': '36px', 'color': '#19C37D', 'fontWeight': 'bold'})
               ], style={'flex': '1', 'textAlign': 'center'}),

               html.Div([
                   html.H4("Latency (avg)", style={'color': '#8E8EA0', 'fontSize': '12px'}),
                   html.Div("0ms", id='metrics-latency',
                           style={'fontSize': '36px', 'color': '#19C37D', 'fontWeight': 'bold'})
               ], style={'flex': '1', 'textAlign': 'center'})
           ], style={'display': 'flex', 'marginBottom': '30px'}),

           dcc.Graph(
               id='metrics-chart',
               figure=go.Figure(
                   data=[go.Bar(x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], y=[10, 20, 15, 30, 25])],
                   layout=go.Layout(
                       title='Daily Usage',
                       paper_bgcolor='#1E1E1E',
                       plot_bgcolor='#2C2C2C',
                       font={'color': '#ECECF1'}
                   )
               ),
               style={'height': '300px'}
           )
       ], style={'padding': '20px', 'backgroundColor': '#1E1E1E', 'height': '100%'})

   custom_tabs = [
       {
           'id': 'metrics',
           'label': 'Metrics',
           'content': create_metrics_tab()
       }
   ]

   if __name__ == '__main__':
       app = DashInteractiveAppWithLogs(
           title="Metrics Tab Demo",
           custom_main_tabs=custom_tabs
       )
       app.run()

Custom Chat Message Styling
---------------------------

Modify the ChatWindow component styling:

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive import DashInteractiveAppWithLogs
   from dash import html

   def custom_message_renderer(message: dict) -> html.Div:
       """Custom message renderer with different styling."""
       role = message.get('role', 'user')
       content = message.get('content', '')
       timestamp = message.get('timestamp', '')

       if role == 'user':
           return html.Div([
               html.Div(content, style={
                   'backgroundColor': '#4A90D9',  # Blue for user
                   'color': 'white',
                   'padding': '12px 16px',
                   'borderRadius': '18px 18px 4px 18px',
                   'maxWidth': '70%',
                   'marginLeft': 'auto'
               }),
               html.Div(timestamp, style={
                   'fontSize': '10px',
                   'color': '#8E8EA0',
                   'textAlign': 'right',
                   'marginTop': '4px'
               })
           ], style={'marginBottom': '15px'})
       else:
           return html.Div([
               html.Div("🤖 Assistant", style={
                   'fontSize': '11px',
                   'color': '#19C37D',
                   'marginBottom': '4px'
               }),
               html.Div(content, style={
                   'backgroundColor': '#2D2D2D',
                   'color': '#ECECF1',
                   'padding': '12px 16px',
                   'borderRadius': '18px 18px 18px 4px',
                   'maxWidth': '70%',
                   'border': '1px solid #19C37D'
               }),
               html.Div(timestamp, style={
                   'fontSize': '10px',
                   'color': '#8E8EA0',
                   'marginTop': '4px'
               })
           ], style={'marginBottom': '15px'})

   # Use by overriding update_messages in ChatWindow

Extending the Application
-------------------------

Create a fully customized application subclass:

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive import DashInteractiveAppWithLogs
   from dash import html, dcc
   from dash.dependencies import Input, Output, State
   import time

   class ProductionChatApp(DashInteractiveAppWithLogs):
       """Production-ready chat application with custom features."""

       def __init__(self, api_key: str = None, **kwargs):
           self.api_key = api_key
           self.request_count = 0
           self.start_time = time.time()

           # Custom tabs
           custom_monitor = [
               {'id': 'stats', 'label': '📊 Stats', 'content': self._create_stats_content()}
           ]

           super().__init__(custom_monitor_tabs=custom_monitor, **kwargs)
           self._register_custom_callbacks()

       def _create_stats_content(self):
           return html.Div([
               html.Div([
                   html.Span("Requests: ", style={'color': '#8E8EA0'}),
                   html.Span("0", id='stats-requests', style={'color': '#19C37D'})
               ]),
               html.Div([
                   html.Span("Uptime: ", style={'color': '#8E8EA0'}),
                   html.Span("0s", id='stats-uptime', style={'color': '#19C37D'})
               ]),
               html.Div([
                   html.Span("API Status: ", style={'color': '#8E8EA0'}),
                   html.Span("✓ Connected" if self.api_key else "✗ No API Key",
                            id='stats-api',
                            style={'color': '#19C37D' if self.api_key else '#FF4500'})
               ])
           ], style={'padding': '10px', 'fontSize': '10px'})

       def _register_custom_callbacks(self):
           @self.app.callback(
               [Output('stats-requests', 'children'),
                Output('stats-uptime', 'children')],
               Input('response-poll-interval', 'n_intervals')
           )
           def update_stats(n):
               uptime = int(time.time() - self.start_time)
               return str(self.request_count), f"{uptime}s"

       def handle_message(self, message: str, session_id: str) -> str:
           """Custom message handler with tracking."""
           self.request_count += 1
           # Your processing logic here
           return f"Processed request #{self.request_count}: {message}"

   if __name__ == '__main__':
       app = ProductionChatApp(
           title="Production Chat",
           api_key="your-api-key",
           port=8050,
           debug=True
       )
       app.set_message_handler(app.handle_message)
       app.run()

Custom CSS Injection
--------------------

Inject custom CSS for advanced styling:

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive import DashInteractiveAppWithLogs

   app = DashInteractiveAppWithLogs(title="Styled App")

   # Inject custom CSS
   app.app.index_string = app.app.index_string.replace(
       '</head>',
       '''
       <style>
           /* Custom scrollbar */
           ::-webkit-scrollbar {
               width: 8px;
           }
           ::-webkit-scrollbar-track {
               background: #1E1E1E;
           }
           ::-webkit-scrollbar-thumb {
               background: #19C37D;
               border-radius: 4px;
           }

           /* Custom animations */
           @keyframes fadeIn {
               from { opacity: 0; transform: translateY(10px); }
               to { opacity: 1; transform: translateY(0); }
           }

           .message-appear {
               animation: fadeIn 0.3s ease-out;
           }

           /* Button hover effects */
           button:hover {
               filter: brightness(1.1);
               transition: filter 0.2s;
           }
       </style>
       </head>
       '''
   )

   app.run()

See Also
--------

- :doc:`../components/base` - BaseComponent reference
- :doc:`../components/index` - All component documentation
- :doc:`../architecture` - Architecture overview
- :doc:`../styling` - Styling guide
