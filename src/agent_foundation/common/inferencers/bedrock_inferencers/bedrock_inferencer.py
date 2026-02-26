from typing import Union, Sequence

from attr import attrib, attrs
import json
from science_modeling_tools.common.infra.bedrock.constants import (
    BEDROCK_SERVICE_NAME_BEDROCK_RUNTIME,
    BEDROCK_SERVICE_REGION_US_WEST2,
    BEDROCK_RUNTIME_SERVICE_URL_PREFIX
)
from science_modeling_tools.common.inferencers.remote_inferencer_base import RemoteInferencerBase
from science_modeling_tools.common.infra.bedrock.client import (
    get_bedrock_session,
    get_bedrock_client,
    get_bedrock_runtime_service_url
)


@attrs
class BedrockInferencer(RemoteInferencerBase):
    """
    An inferencer class for performing inference using the Bedrock service.

    This class extends `RemoteInferencerBase` and is tailored for Bedrock services, handling the setup of
    service clients, request construction, and response parsing specific to Bedrock runtime environments.

    Attributes:
        service_name (str): The name of the Bedrock service to connect to. Defaults to Bedrock runtime service.
        region (str): The AWS region where the Bedrock service is hosted. Defaults to US West 2.
        read_timeout (int): Timeout in milliseconds for reading responses. Defaults to 3000.
        connect_timeout (int): Timeout in milliseconds for establishing connections. Defaults to 3000.
        max_attempts (int): Maximum number of attempts for retrying failed requests. Defaults to 3.
        access_key (str): Access key for authentication. Defaults to None. Use together with the 'secret_key' attribute
            from the `InferencerBase` class.
    """
    service_name: str = attrib(default=BEDROCK_SERVICE_NAME_BEDROCK_RUNTIME)
    region: str = attrib(default=BEDROCK_SERVICE_REGION_US_WEST2)
    read_timeout: int = attrib(default=3000)
    connect_timeout: int = attrib(default=3000)
    max_attempts: int = attrib(default=3)
    access_key: Union[str, Sequence[str]] = attrib(default=None)

    def __attrs_post_init__(self):
        """
        Post-initialization method that sets the service URL and prefix if they are not provided.

        This method configures the `service_url` based on the service name and region. If the service name
        matches the Bedrock runtime service, it will set the `service_url` accordingly. Raises a ValueError
        if `service_url` cannot be determined and is not provided explicitly.

        Raises:
            ValueError: If the `service_url` is not set and cannot be determined from the service name.
        """
        if not self.service_url:
            if self.service_name == BEDROCK_SERVICE_NAME_BEDROCK_RUNTIME:
                self.service_url = get_bedrock_runtime_service_url(self.region)
            if not self.service_url:
                raise ValueError(f"'service_url' must be provided")
        if not self.service_url_prefix:
            self.service_url_prefix = BEDROCK_RUNTIME_SERVICE_URL_PREFIX
        super().__attrs_post_init__()

    def get_client(self):
        """
         Creates and returns the session and client objects for interacting with the Bedrock service.

         This method sets up an AWS session and Bedrock client using the provided credentials and service configuration.
         It supports custom AWS access keys and secret keys if specified.

         Returns:
             tuple: A tuple containing the AWS session and the Bedrock client objects.

         Raises:
             Exception: If there is an error in creating the session or client due to misconfiguration or connectivity issues.
         """
        session = get_bedrock_session(
            access_key=self.access_key,
            secret_key=self.secret_key
        )
        client = get_bedrock_client(
            session,
            service_url=self.service_url,
            service_name=self.service_name,
            region=self.region,
            read_timeout=self.read_timeout,
            connect_timeout=self.connect_timeout,
            max_attempts=self.max_attempts
        )

        return session, client

    def _send_request(self, client, request):
        """
        Sends an inference request to the Bedrock service using the provided client.

        This method sends a JSON-formatted request to the Bedrock model endpoint using the Bedrock client.
        The request can be a string or a dictionary, which will be serialized to JSON if not already a string.
        The response is parsed from JSON format.

        Args:
            client (tuple): A tuple containing the AWS session and Bedrock client.
            request (Union[str, dict]): The request payload, which can be a string or dictionary.

        Returns:
            dict: The JSON-parsed response from the Bedrock service.

        Raises:
            Exception: If the request fails or if there is an error in parsing the response.
        """
        _, client = client
        if not isinstance(request, str):
            request = json.dumps(request)
        response = client.invoke_model(
            body=request,
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json",
        )
        return json.loads(
            response.get("body").read().decode_with_existence_flags("utf-8")
        )