from abc import abstractmethod
from typing import Any, Dict, Union

from attr import attrib, attrs

from science_modeling_tools.common.inferencers.inferencer_base import InferencerBase
from rich_python_utils.string_utils import add_prefix


@attrs
class RemoteInferencerBase(InferencerBase):
    """
    Base class for implementing remote inference logic.

    This class extends `InferencerBase` to provide a framework for remote inference operations
    by defining a common structure for constructing requests, sending them to a remote service,
    and parsing the responses. Subclasses should implement the abstract methods to specify the
    behavior of the client, request construction, sending the request, and parsing the response.

    Attributes:
        service_url (str): The URL of the remote service where inference requests will be sent.
        service_url_prefix (str): A prefix to be added to the service URL, useful for defining
            common prefixes like protocols (e.g., 'http://'). Defaults to an empty string.
        timeout (int): The timeout in seconds for remote service requests. Defaults to 300 seconds (5 minutes).
    """

    service_url: str = attrib(default="")
    service_url_prefix: str = attrib(default="")
    timeout: int = attrib(default=300)

    def __attrs_post_init__(self):
        """
        Post-initialization method that sets the full service URL by adding the prefix to the base URL.

        This method automatically adjusts the `service_url` by prepending the `service_url_prefix`
        if it is provided, ensuring that URLs are correctly formed and free of redundant prefixes.

        Uses:
            add_prefix: A utility function to add the prefix to the service URL without repeating it.

        Example:
            If `service_url` is 'example.com' and `service_url_prefix` is 'http://',
            the resulting `service_url` will be 'http://example.com'.
        """
        if self.service_url_prefix:
            self.service_url = add_prefix(
                self.service_url,
                prefix=self.service_url_prefix,
                sep="",
                avoid_repeat=True,
            )
        super().__attrs_post_init__()

    @abstractmethod
    def get_client(self):
        """
        Abstract method to retrieve or create a client for sending requests.

        Subclasses must implement this method to provide the logic for initializing or retrieving
        the client that will be used to send requests to the remote service.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.

        Returns:
            Any: The client object to be used for sending requests.
        """
        raise NotImplementedError

    @abstractmethod
    def construct_request(self, inference_input: Any, **_inference_args) -> dict:
        """
        Abstract method to construct the request payload for the remote inference service.

        Subclasses must implement this method to define how the request is structured,
        including the prompt and any additional inference arguments required by the service.

        Args:
            inference_input (str): The input prompt for the inference.
            _inference_args: Additional keyword arguments for constructing the request.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.

        Returns:
            dict: The constructed request payload.
        """
        raise NotImplementedError

    @abstractmethod
    def _send_request(self, client, request):
        """
        Abstract method to send the constructed request to the remote service using the client.

        Subclasses must implement this method to handle the logic of sending requests, including
        any necessary error handling and retries specific to the remote service interaction.

        Args:
            client: The client object used to send the request.
            request: The request payload to be sent to the remote service.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.

        Returns:
            Any: The response from the remote service.
        """
        raise NotImplementedError

    def _parse_response(self, response: Any) -> Union[str, Dict, Any]:
        """
        Parses the response received from the remote service.

        This method extracts the relevant inference results from the response data structure
        returned by the remote service. The output can vary depending on the service response
        format and may include strings, dictionaries, or other data types.

        Args:
            response (Any): The raw response data from the remote service.

        Returns:
            Union[str, Dict, Any]: The parsed result of the inference. Typically a string or
            dictionary, but can also be other types depending on the service implementation.
        """
        return response

    def _infer(
        self, inference_input: str, inference_config: Any = None, **_inference_args
    ) -> str:
        """
        Executes the full inference process by constructing, sending, and parsing a request to a remote service.

        This method integrates the steps of getting the client, constructing the request, sending it,
        and parsing the response. It utilizes the methods defined in the subclass to perform these operations.

        Args:
            inference_input (str): The input prompt for the inference.
            _inference_args: Additional keyword arguments for constructing the request.

        Returns:
            Union[str, Dict, Any]: The parsed inference result from the remote service, which could be
            a string, dictionary, or other types based on the specific service response.
        """
        client = self.get_client()
        request = self.construct_request(inference_input, **_inference_args)
        response = self._send_request(client, request)
        return self._parse_response(response)
