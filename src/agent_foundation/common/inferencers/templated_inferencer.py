"""
TemplatedInferencer - Inferencer wrapper with built-in template management.

This module provides a wrapper class that adds template management capabilities
to any InferencerBase subclass, enabling template-based prompting without
modifying the existing class hierarchy.

Example:
    >>> from agent_foundation.common.inferencers.templated_inferencer import TemplatedInferencer
    >>> from agent_foundation.common.inferencers.api_inferencers.claude_api_inferencer import ClaudeApiInferencer
    >>> from rich_python_utils.string_utils.formatting.template_manager import TemplateManager
    >>>
    >>> templated = TemplatedInferencer(
    ...     base_inferencer=ClaudeApiInferencer(max_retry=3),
    ...     template_manager=TemplateManager(templates="path/to/templates/")
    ... )
    >>> result = templated("find_element", feed={"html": html, "description": "search box"})
"""

from typing import Any, Dict, Optional

from attr import attrs, attrib

from science_modeling_tools.common.inferencers.inferencer_base import InferencerBase
from rich_python_utils.string_utils.formatting.template_manager import TemplateManager


@attrs
class TemplatedInferencer:
    """
    Inferencer wrapper with built-in template management.

    Wraps any InferencerBase to provide template-based prompting:
    - Pass template key + feed variables instead of raw prompts
    - TemplateManager handles lookup, versioning, and substitution
    - Base inferencer handles LLM call, retry, post-processing

    This uses composition over inheritance, so no changes to the existing
    inferencer class hierarchy are required.

    Attributes:
        base_inferencer: Any InferencerBase subclass (e.g., ClaudeApiInferencer)
        template_manager: TemplateManager for resolving template keys to prompts

    Example:
        >>> templated = TemplatedInferencer(
        ...     base_inferencer=ClaudeApiInferencer(max_retry=3),
        ...     template_manager=TemplateManager(
        ...         templates="path/to/templates/",
        ...         template_formatter=handlebars_format,
        ...     )
        ... )
        >>>
        >>> # Call with template key + variables
        >>> result = templated("find_element", feed={"html": html, "description": "search box"})
        >>>
        >>> # Or use raw prompt directly
        >>> result = templated.infer_raw("What is 2+2?")
    """
    base_inferencer: InferencerBase = attrib()
    template_manager: TemplateManager = attrib()

    def __call__(
        self,
        template_key: str,
        feed: Optional[Dict[str, Any]] = None,
        inference_config: Any = None,
        active_template_type: Optional[str] = None,
        active_template_root_space: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Call with template key instead of raw prompt.

        Args:
            template_key: Template to use (e.g., "find_element", "execute_task").
                          Resolved by template_manager to a prompt string.
            feed: Variables for template substitution. These are passed to the
                  template_manager as the `feed` parameter.
                  Example: {"html": "<html>...</html>", "description": "search box"}
            inference_config: Optional configuration passed to base inferencer.
            active_template_type: Override template type for this call (e.g., "main", "reflection").
                                  If None, uses the template_manager's default.
            active_template_root_space: Override template root space for this call
                                        (e.g., "action_agent"). If None, uses the
                                        template_manager's default.
            **kwargs: Additional inference args passed to base inferencer
                      (e.g., temperature, max_tokens).

        Returns:
            LLM response from base_inferencer. Type depends on the inferencer
            and any post-processors configured.

        Example:
            >>> result = templated("find_element", feed={"html": html, "description": "input box"})
            >>> print(result)  # "42" (the __id__ value)
            >>> # Or with template type override
            >>> result = templated("Search", feed={...}, active_template_type="reflection")
        """
        prompt = self.template_manager(
            template_key,
            feed=feed,
            active_template_type=active_template_type,
            active_template_root_space=active_template_root_space,
        )
        return self.base_inferencer(prompt, inference_config=inference_config, **kwargs)

    def infer_raw(self, prompt: str, inference_config: Any = None, **kwargs) -> Any:
        """
        Passthrough for raw prompts without template resolution.

        Use this when you have a pre-formatted prompt and don't need
        template resolution.

        Args:
            prompt: Raw prompt string to send to the LLM.
            inference_config: Optional configuration passed to base inferencer.
            **kwargs: Additional inference args passed to base inferencer.

        Returns:
            LLM response from base_inferencer.

        Example:
            >>> result = templated.infer_raw("What is the capital of France?")
            >>> print(result)  # "Paris"
        """
        return self.base_inferencer(prompt, inference_config=inference_config, **kwargs)

    def infer(
        self,
        template_key: str,
        feed: Optional[Dict[str, Any]] = None,
        inference_config: Any = None,
        active_template_type: Optional[str] = None,
        active_template_root_space: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Alias for __call__ for explicit method invocation.

        Args:
            template_key: Template to use.
            feed: Variables for template substitution.
            inference_config: Optional configuration passed to base inferencer.
            active_template_type: Override template type for this call.
            active_template_root_space: Override template root space for this call.
            **kwargs: Additional inference args.

        Returns:
            LLM response from base_inferencer.
        """
        return self.__call__(
            template_key,
            feed=feed,
            inference_config=inference_config,
            active_template_type=active_template_type,
            active_template_root_space=active_template_root_space,
            **kwargs
        )
