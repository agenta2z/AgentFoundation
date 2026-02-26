import json
from typing import Any, Dict, Sequence, Union

import requests


def generate_text(
    prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]],
    model: str = "claude-3-5-sonnet-20241022",
    service_url: str = None,
    max_new_tokens: int = None,
    temperature: float = 0.7,
    top_p: float = None,
    api_key: str = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate text using a generic HTTP service.

    This function creates an input dictionary with the provided parameters and sends
    a POST request to the specified service URL.

    Args:
        prompt_or_messages: The input prompt or messages to generate text from
        model: The model to use for generation (defaults to claude-3-5-sonnet-20241022)
        service_url: The URL of the HTTP service to call (required, should include port if needed)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for text generation (0.0 to 1.0)
        top_p: Top-p parameter for nucleus sampling
        api_key: API key for the service (if required)
        **kwargs: Additional parameters to pass to the service

    Returns:
        Dict containing the service response

    Raises:
        ValueError: If service_url is not provided
        requests.RequestException: If the HTTP request fails
    """
    if not service_url:
        raise ValueError("service_url is required for HTTP request generation")

    # Create input dictionary with all parameters
    input_data = {"prompt_or_messages": prompt_or_messages, "model": model}

    # Add optional parameters if provided
    if max_new_tokens is not None:
        input_data["max_new_tokens"] = max_new_tokens
    if temperature is not None:
        input_data["temperature"] = temperature
    if top_p is not None:
        input_data["top_p"] = top_p
    if api_key is not None:
        input_data["api_key"] = api_key

    # Add any additional kwargs
    input_data.update(kwargs)

    try:
        # Make POST request to the service
        response = requests.post(
            service_url,
            json=input_data,
            headers={"Content-Type": "application/json"},
            timeout=300,  # 5 minute timeout
        )

        # Raise an exception for bad status codes
        response.raise_for_status()

        # Return the JSON response
        return response.json()

    except requests.exceptions.RequestException as e:
        # Return error response in consistent format
        return {
            "success": False,
            "error": f"HTTP request failed: {str(e)}",
            "response": None,
        }
    except json.JSONDecodeError as e:
        # Handle JSON parsing errors
        return {
            "success": False,
            "error": f"Failed to parse JSON response: {str(e)}",
            "response": response.text if "response" in locals() else None,
        }
