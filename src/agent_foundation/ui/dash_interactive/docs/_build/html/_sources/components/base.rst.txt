=============
BaseComponent
=============

.. module:: agent_foundation.ui.dash_interactive.components.base
   :synopsis: Abstract base class for UI components

The ``BaseComponent`` class is the abstract foundation for all UI components in the
Dash Interactive framework.


Overview
========

``BaseComponent`` provides a consistent interface for creating modular, reusable
Dash components. All UI components in the framework inherit from this class.


Class Definition
================

.. code-block:: python

   class BaseComponent(ABC):
       """
       Abstract base class for reusable Dash components.

       This class provides a standard interface for creating modular
       Dash components with consistent styling and behavior.
       """


Constructor
-----------

.. code-block:: python

   def __init__(self, component_id: str, style: Optional[Dict[str, Any]] = None):
       """
       Initialize the base component.

       Args:
           component_id: Unique identifier for this component
           style: Optional CSS style dictionary to override defaults
       """

**Parameters:**

* ``component_id`` (str): A unique string identifier for the component instance.
  This ID is used to generate IDs for all sub-elements.

* ``style`` (Dict, optional): CSS style dictionary that will be merged with
  the component's default styles. Custom styles override defaults.


Attributes
----------

.. list-table:: Instance Attributes
   :widths: 25 75
   :header-rows: 1

   * - Attribute
     - Description
   * - ``component_id``
     - The unique identifier passed to the constructor
   * - ``style``
     - Merged dictionary of default and custom styles
   * - ``_default_style``
     - The component's default style (set by ``_get_default_style()``)


Abstract Methods
================

These methods MUST be implemented by all subclasses.


_get_default_style
------------------

.. code-block:: python

   @abstractmethod
   def _get_default_style(self) -> Dict[str, Any]:
       """
       Get default CSS style for this component.

       Returns:
           Dictionary of CSS properties and values
       """
       pass

**Purpose:** Define the default visual appearance of the component.

**Example Implementation:**

.. code-block:: python

   def _get_default_style(self) -> Dict[str, Any]:
       return {
           'backgroundColor': '#343541',
           'color': '#ECECF1',
           'padding': '20px',
           'borderRadius': '8px'
       }


layout
------

.. code-block:: python

   @abstractmethod
   def layout(self) -> Any:
       """
       Generate the Dash layout for this component.

       Returns:
           Dash component (html.Div, dcc.Graph, etc.) or layout
       """
       pass

**Purpose:** Create and return the Dash layout structure.

**Example Implementation:**

.. code-block:: python

   def layout(self) -> html.Div:
       return html.Div(
           id=self.get_id(),
           children=[
               html.H3("My Component"),
               html.Button("Click Me", id=self.get_id('button'))
           ],
           style=self.style
       )


get_callback_inputs
-------------------

.. code-block:: python

   @abstractmethod
   def get_callback_inputs(self) -> List[Any]:
       """
       Get list of Dash Input objects for callbacks.

       Returns:
           List of dash.dependencies.Input objects
       """
       pass

**Purpose:** Declare which elements trigger callbacks.

**Example Implementation:**

.. code-block:: python

   def get_callback_inputs(self) -> List[Input]:
       return [
           Input(self.get_id('button'), 'n_clicks'),
           Input(self.get_id('input'), 'value')
       ]


get_callback_outputs
--------------------

.. code-block:: python

   @abstractmethod
   def get_callback_outputs(self) -> List[Any]:
       """
       Get list of Dash Output objects for callbacks.

       Returns:
           List of dash.dependencies.Output objects
       """
       pass

**Purpose:** Declare which elements are updated by callbacks.

**Example Implementation:**

.. code-block:: python

   def get_callback_outputs(self) -> List[Output]:
       return [
           Output(self.get_id('display'), 'children'),
           Output(self.get_id('status'), 'className')
       ]


Helper Methods
==============

get_id
------

.. code-block:: python

   def get_id(self, suffix: str = "") -> str:
       """
       Generate a unique ID for sub-components.

       Args:
           suffix: Suffix to append to component_id

       Returns:
           Unique component ID string
       """
       if suffix:
           return f"{self.component_id}-{suffix}"
       return self.component_id

**Purpose:** Generate consistent, unique IDs for all elements within the component.

**Examples:**

.. code-block:: python

   component = MyComponent(component_id="my-comp")

   component.get_id()            # Returns: "my-comp"
   component.get_id('button')    # Returns: "my-comp-button"
   component.get_id('input')     # Returns: "my-comp-input"
   component.get_id('container') # Returns: "my-comp-container"


Creating Custom Components
==========================

To create a custom component, inherit from ``BaseComponent`` and implement all
abstract methods:

.. code-block:: python

   from agent_foundation.ui.dash_interactive.components.base import BaseComponent
   from dash import html, dcc
   from dash.dependencies import Input, Output
   from typing import Any, Dict, List

   class StatusIndicator(BaseComponent):
       """A simple status indicator component."""

       def __init__(
           self,
           component_id: str = "status-indicator",
           initial_status: str = "Ready",
           style: Dict[str, Any] = None
       ):
           super().__init__(component_id, style)
           self.initial_status = initial_status

       def _get_default_style(self) -> Dict[str, Any]:
           return {
               'padding': '10px 20px',
               'backgroundColor': '#2C2C2C',
               'borderRadius': '8px',
               'display': 'flex',
               'alignItems': 'center',
               'gap': '10px'
           }

       def layout(self) -> html.Div:
           return html.Div(
               id=self.get_id(),
               children=[
                   # Status LED
                   html.Div(
                       id=self.get_id('led'),
                       style={
                           'width': '12px',
                           'height': '12px',
                           'borderRadius': '50%',
                           'backgroundColor': '#19C37D'
                       }
                   ),
                   # Status text
                   html.Span(
                       id=self.get_id('text'),
                       children=self.initial_status,
                       style={'color': '#ECECF1'}
                   ),
                   # Refresh button
                   html.Button(
                       "Refresh",
                       id=self.get_id('refresh-btn'),
                       n_clicks=0,
                       style={
                           'marginLeft': 'auto',
                           'backgroundColor': '#40414F',
                           'color': '#ECECF1',
                           'border': 'none',
                           'padding': '5px 10px',
                           'borderRadius': '4px',
                           'cursor': 'pointer'
                       }
                   )
               ],
               style=self.style
           )

       def get_callback_inputs(self) -> List[Input]:
           return [
               Input(self.get_id('refresh-btn'), 'n_clicks')
           ]

       def get_callback_outputs(self) -> List[Output]:
           return [
               Output(self.get_id('text'), 'children'),
               Output(self.get_id('led'), 'style')
           ]

       def update_status(self, status: str, is_ok: bool = True):
           """Helper method to generate status update values."""
           led_style = {
               'width': '12px',
               'height': '12px',
               'borderRadius': '50%',
               'backgroundColor': '#19C37D' if is_ok else '#FF4500'
           }
           return status, led_style


Using Your Custom Component
---------------------------

.. code-block:: python

   from dash import Dash, html
   from dash.dependencies import Input, Output

   app = Dash(__name__)

   status = StatusIndicator(component_id="my-status", initial_status="Starting...")

   app.layout = html.Div([
       status.layout()
   ])

   @app.callback(
       status.get_callback_outputs(),
       status.get_callback_inputs()
   )
   def handle_refresh(n_clicks):
       if n_clicks:
           return status.update_status("Refreshed!", True)
       return status.update_status("Ready", True)

   if __name__ == '__main__':
       app.run(debug=True)


Best Practices
==============

1. **Use meaningful component IDs**

   Choose descriptive IDs that indicate the component's purpose:

   .. code-block:: python

      # Good
      ChatHistoryList(component_id="sidebar-chat-history")

      # Avoid
      ChatHistoryList(component_id="comp1")

2. **Keep styles in _get_default_style**

   Don't hardcode styles in ``layout()``; use the style dictionary:

   .. code-block:: python

      # Good
      def layout(self):
          return html.Div(style=self.style)

      # Avoid
      def layout(self):
          return html.Div(style={'backgroundColor': '#343541'})

3. **Use get_id() for all element IDs**

   This ensures unique IDs when multiple instances exist:

   .. code-block:: python

      # Good
      html.Button(id=self.get_id('submit'))

      # Avoid
      html.Button(id='submit')

4. **Document callback requirements**

   Make it clear what callbacks expect:

   .. code-block:: python

      def get_callback_inputs(self) -> List[Input]:
          """
          Inputs:
          - refresh-btn.n_clicks: Triggers status refresh
          - interval.n_intervals: Auto-refresh trigger
          """
          return [...]


See Also
========

* :doc:`index` - Components overview
* :doc:`../architecture` - System architecture
* :doc:`../examples/custom_components` - More custom component examples
