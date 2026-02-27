=======
Styling
=======

This document describes the visual styling and theming of the Dash Interactive framework.


Color Scheme
============

Dash Interactive uses a dark theme inspired by ChatGPT's interface.


Primary Colors
--------------

.. list-table:: Primary Colors
   :widths: 30 20 50
   :header-rows: 1

   * - Element
     - Color
     - Hex Code
   * - Sidebar Background
     - Dark Gray
     - ``#202123``
   * - Main Background
     - Medium Gray
     - ``#343541``
   * - Message Background
     - Light Gray
     - ``#444654``
   * - Input Background
     - Dark Input
     - ``#40414F``
   * - Border Color
     - Subtle Border
     - ``#4D4D4F``


Text Colors
-----------

.. list-table:: Text Colors
   :widths: 30 20 50
   :header-rows: 1

   * - Element
     - Color
     - Hex Code
   * - Primary Text
     - Light Gray
     - ``#ECECF1``
   * - Secondary Text
     - Muted Gray
     - ``#8E8EA0``
   * - Placeholder Text
     - Dim Gray
     - ``#6E6E80``


Accent Colors
-------------

.. list-table:: Accent Colors
   :widths: 30 20 50
   :header-rows: 1

   * - Element
     - Color
     - Hex Code
   * - Active/Success
     - Green
     - ``#19C37D``
   * - Active Hover
     - Dark Green
     - ``#10A37F``
   * - Exit Nodes
     - Coral Red
     - ``#FF6B6B``
   * - Exit Border
     - Yellow
     - ``#FFD93D``


Log Level Colors
----------------

Log entries are color-coded by severity level:

.. list-table:: Log Level Colors
   :widths: 20 20 20 40
   :header-rows: 1

   * - Level
     - Name
     - Color
     - Hex Code
   * - 10
     - DEBUG
     - Cyan
     - ``#00BFFF``
   * - 20
     - INFO
     - Green
     - ``#19C37D``
   * - 30
     - WARNING
     - Orange
     - ``#FFA500``
   * - 40
     - ERROR
     - Red-Orange
     - ``#FF4500``
   * - 50
     - CRITICAL
     - Crimson
     - ``#DC143C``


Typography
==========

Font Stack
----------

The framework uses a system font stack for optimal rendering:

.. code-block:: css

   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                Roboto, "Helvetica Neue", Arial, sans-serif;

For code and log entries:

.. code-block:: css

   font-family: Consolas, Monaco, "Courier New", monospace;


Font Sizes
----------

.. list-table:: Font Sizes
   :widths: 40 30 30
   :header-rows: 1

   * - Element
     - Size
     - Weight
   * - Welcome Header
     - 32px
     - 600
   * - Section Headers
     - 16px
     - 600
   * - Body Text
     - 14-15px
     - 400
   * - Timestamps
     - 11px
     - 400
   * - Log Entries
     - 12px
     - 400
   * - Metadata
     - 9-10px
     - 400


Component Styling
=================

Sidebar (ChatHistoryList)
-------------------------

.. code-block:: python

   style = {
       'width': '300px',
       'height': '100vh',
       'backgroundColor': '#202123',
       'color': '#ECECF1',
       'overflowY': 'auto',
       'display': 'flex',
       'flexDirection': 'column',
       'borderRight': '1px solid #4D4D4F'
   }

**Session Items:**

.. code-block:: python

   # Normal state
   {'backgroundColor': 'transparent'}

   # Active/Selected state
   {'backgroundColor': '#343541'}


Chat Window (ChatWindow)
------------------------

.. code-block:: python

   style = {
       'flex': '1',
       'height': '100%',
       'backgroundColor': '#343541',
       'display': 'flex',
       'flexDirection': 'column',
       'position': 'relative',
       'overflow': 'hidden'
   }

**Message Styling:**

.. code-block:: python

   # User message
   user_style = {
       'backgroundColor': '#40414F',
       'maxWidth': '80%',
       'alignSelf': 'flex-end'
   }

   # Assistant message
   assistant_style = {
       'backgroundColor': '#444654',
       'maxWidth': '100%',
       'alignSelf': 'flex-start'
   }


Buttons
-------

**Primary Button (Send):**

.. code-block:: python

   style = {
       'backgroundColor': '#19C37D',
       'color': 'white',
       'border': 'none',
       'borderRadius': '6px',
       'cursor': 'pointer',
       'transition': 'background-color 0.2s'
   }

**Secondary Button (New Chat):**

.. code-block:: python

   style = {
       'backgroundColor': 'transparent',
       'color': '#ECECF1',
       'border': '1px solid #4D4D4F',
       'borderRadius': '6px',
       'cursor': 'pointer',
       'transition': 'background-color 0.2s'
   }

**Tab Button Active:**

.. code-block:: python

   style = {
       'backgroundColor': '#19C37D',
       'color': '#ECECF1',
       'borderBottom': '2px solid #19C37D'
   }

**Tab Button Inactive:**

.. code-block:: python

   style = {
       'backgroundColor': '#40414F',
       'color': '#8E8EA0',
       'borderBottom': '2px solid transparent'
   }


Log Graph (LogGraphVisualization)
---------------------------------

**Graph Container:**

.. code-block:: python

   style = {
       'height': '100%',
       'backgroundColor': '#2C2C2C',
       'padding': '10px'
   }

**Node Styling (Plotly):**

.. code-block:: python

   # Normal nodes
   marker = {
       'size': 30,
       'color': '#19C37D',
       'line': {'color': '#ECECF1', 'width': 2}
   }

   # Exit nodes
   marker = {
       'size': 35,
       'color': '#FF6B6B',
       'symbol': 'diamond',
       'line': {'color': '#FFD93D', 'width': 2}
   }

**Cytoscape Stylesheet:**

.. code-block:: python

   stylesheet = [
       {
           'selector': 'node',
           'style': {
               'background-color': '#19C37D',
               'color': '#ECECF1',
               'border-width': '2px',
               'border-color': '#ECECF1'
           }
       },
       {
           'selector': 'node.exit',
           'style': {
               'background-color': '#FF6B6B',
               'border-color': '#FFD93D',
               'shape': 'octagon'
           }
       },
       {
           'selector': 'edge',
           'style': {
               'line-color': '#565869',
               'target-arrow-color': '#565869',
               'curve-style': 'bezier'
           }
       }
   ]


Log Details (LogDetailsPanel)
-----------------------------

**Panel Container:**

.. code-block:: python

   style = {
       'height': '100%',
       'backgroundColor': '#1E1E1E',
       'overflowY': 'auto',
       'padding': '10px'
   }

**Log Entry Card:**

.. code-block:: python

   style = {
       'marginBottom': '12px',
       'borderRadius': '6px',
       'border': '1px solid #3C3C3C',
       'overflow': 'hidden'
   }

**Level Badge:**

.. code-block:: python

   def get_level_badge_style(level_color):
       return {
           'color': level_color,
           'fontWeight': 'bold',
           'fontSize': '11px',
           'padding': '3px 8px',
           'backgroundColor': f"{level_color}22",  # 22 = ~13% opacity
           'borderRadius': '3px',
           'display': 'inline-block'
       }


Monitor Panel
-------------

**Floating Panel:**

.. code-block:: python

   style = {
       'position': 'fixed',
       'bottom': '20px',
       'right': '20px',
       'width': '280px',
       'zIndex': '3000',
       'backgroundColor': 'rgba(44, 44, 44, 0.95)',
       'borderRadius': '6px',
       'boxShadow': '0 4px 12px rgba(0,0,0,0.4)',
       'border': '1px solid rgba(255,255,255,0.1)'
   }


Animations
==========

Typing Indicator
----------------

The typing indicator uses a wave animation:

.. code-block:: css

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


Transitions
-----------

Most interactive elements use smooth transitions:

.. code-block:: css

   transition: background-color 0.2s ease;
   transition: all 0.2s ease;


Customizing Styles
==================

Overriding Default Styles
-------------------------

Pass custom styles when creating components:

.. code-block:: python

   from agent_foundation.ui.dash_interactive.components.chat_window import ChatWindow

   chat_window = ChatWindow(
       component_id="my-chat",
       style={
           'backgroundColor': '#1a1a2e',  # Custom background
           'borderLeft': '3px solid #00ff88'  # Custom border
       }
   )

The custom styles are merged with defaults, with custom taking precedence.


Creating Theme Variants
-----------------------

Create a custom theme by subclassing:

.. code-block:: python

   class LightThemeChatWindow(ChatWindow):
       def _get_default_style(self):
           return {
               'flex': '1',
               'height': '100%',
               'backgroundColor': '#ffffff',  # Light background
               'color': '#333333',             # Dark text
               'display': 'flex',
               'flexDirection': 'column'
           }


CSS Variables (Future)
----------------------

.. note::
   CSS custom properties support is planned for future versions,
   which will make theming even easier.


Responsive Design
=================

The framework uses flexbox for responsive layouts:

.. code-block:: python

   # Main container
   style = {
       'display': 'flex',
       'height': '100vh',
       'width': '100vw',
       'overflow': 'hidden'
   }

**Sidebar:**

* Fixed width of 300px
* Full height with scroll

**Main Panel:**

* Flexible width (``flex: 1``)
* Fills remaining space


Accessibility
=============

Color Contrast
--------------

All text colors maintain WCAG AA compliance:

* Primary text (#ECECF1) on dark backgrounds (#343541) = 10.9:1 ratio
* Secondary text (#8E8EA0) on dark backgrounds = 4.9:1 ratio


Interactive States
------------------

* Hover states provide visual feedback
* Active states are clearly distinguished
* Focus indicators for keyboard navigation


See Also
========

* :doc:`architecture` - System architecture
* :doc:`components/index` - Component documentation
* :doc:`examples/custom_components` - Customization examples
