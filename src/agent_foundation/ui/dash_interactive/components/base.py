"""
Base component class for building modular Dash UI components.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dash import html, dcc


class BaseComponent(ABC):
    """
    Abstract base class for reusable Dash components.

    This class provides a standard interface for creating modular
    Dash components with consistent styling and behavior.

    Attributes:
        component_id (str): Unique identifier for this component instance
        style (Dict): CSS style dictionary for the component
    """

    def __init__(self, component_id: str, style: Optional[Dict[str, Any]] = None):
        """
        Initialize the base component.

        Args:
            component_id: Unique identifier for this component
            style: Optional CSS style dictionary
        """
        self.component_id = component_id
        self.style = style or {}
        self._default_style = self._get_default_style()
        # Merge default style with custom style
        self.style = {**self._default_style, **self.style}

    @abstractmethod
    def _get_default_style(self) -> Dict[str, Any]:
        """
        Get default CSS style for this component.

        Returns:
            Dictionary of CSS properties
        """
        pass

    @abstractmethod
    def layout(self) -> Any:
        """
        Generate the Dash layout for this component.

        Returns:
            Dash component or layout
        """
        pass

    @abstractmethod
    def get_callback_inputs(self) -> List[Any]:
        """
        Get list of Dash Input objects for callbacks.

        Returns:
            List of dash.dependencies.Input objects
        """
        pass

    @abstractmethod
    def get_callback_outputs(self) -> List[Any]:
        """
        Get list of Dash Output objects for callbacks.

        Returns:
            List of dash.dependencies.Output objects
        """
        pass

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
