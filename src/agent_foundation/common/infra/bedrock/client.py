import os
from typing import Union, List

from boto3 import Session
from botocore.config import Config

from science_modeling_tools.common.infra.bedrock.constants import (
    BEDROCK_SERVICE_NAME_BEDROCK_RUNTIME,
    BEDROCK_SERVICE_REGION_US_WEST2,
    BEDROCK_RUNTIME_SERVICE_URL_PATTERN
)
from rich_python_utils.common_utils import get_, get_multiple, iter_

ENVVAR_BEDROCK_ACCESS_KEY = 'BEDROCK_ACCESS_KEY'
ENVVAR_BEDROCK_SECRET_ACCESS_KEY = 'BEDROCK_SECRET_ACCESS_KEY'


def resolve_access_keys(
        access_key: str = None,
        secret_key: str = None,
        key_index: Union[int, List[int]] = 0
):
    access_key = os.environ.get(ENVVAR_BEDROCK_ACCESS_KEY, access_key)
    secret_key = os.environ.get(ENVVAR_BEDROCK_SECRET_ACCESS_KEY, secret_key)

    if access_key is not None and secret_key is not None:
        access_key = access_key.strip()
        secret_key = secret_key.strip()
    else:
        return None, None

    if access_key and secret_key:
        if ' ' in access_key:
            access_key = access_key.split(' ')
        else:
            access_key = [_key for _key in access_key.split(' ') if _key]
        if ' ' in secret_key:
            secret_key = [_key for _key in secret_key.split(' ') if _key]
        else:
            secret_key = [secret_key]

        if len(access_key) == 1 != len(secret_key) == 1:
            raise ValueError(f"number of 'access_key' must be the same as the number of 'secret_key', got `{len(access_key)}` and `{len(secret_key)}`")

    if key_index is None:
        return access_key, secret_key
    else:
        return (
            get_multiple(access_key, *iter_(key_index)),
            get_multiple(secret_key, *iter_(key_index))
        )


def get_bedrock_session(
        access_key: str = None,
        secret_key: str = None,
):
    access_key, secret_key = resolve_access_keys(
        access_key=access_key, secret_key=secret_key
    )
    if access_key and secret_key:
        session = Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    else:
        session = Session()
    return session


def get_bedrock_client(
        session: Session,
        service_url: str,
        service_name: str = BEDROCK_SERVICE_NAME_BEDROCK_RUNTIME,
        region: str = BEDROCK_SERVICE_REGION_US_WEST2,
        read_timeout: int = 300,
        connect_timeout: int = 300,
        max_attempts: int = 3
):
    boto_config = Config(
        read_timeout=read_timeout, connect_timeout=connect_timeout, retries={'max_attempts': max_attempts}
    )
    return session.client(
        service_name=service_name,
        region_name=region,
        endpoint_url=service_url,
        config=boto_config,
    )


def get_bedrock_runtime_service_url(region: str) -> str:
    if not region:
        raise ValueError("'region' must be provided")
    return BEDROCK_RUNTIME_SERVICE_URL_PATTERN.format(region=region)
