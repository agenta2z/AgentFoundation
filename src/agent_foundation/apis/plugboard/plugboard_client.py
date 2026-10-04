# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Streaming client for Plugboard, Meta's internal LLM gateway.

Talks to the owned Thrift interface (``//ai_productivity/plugboard/if:*``)
directly and authenticates with CAT (crypto auth tokens) — so nothing here
depends on any product tree. Transport is ServiceRouter on a prod Linux host
and secure-thrift (x2p) everywhere else.

Every Thrift/auth import is deferred into the function that needs it: this
module is reachable from ``agent_foundation`` imports made by consumers that do
not depend on the Plugboard targets, and a module-level import would turn a
missing optional backend into an import-time crash for all of them.
"""

from __future__ import annotations

import contextlib
import platform
import uuid
from collections.abc import AsyncIterator
from typing import Any, AsyncGenerator, Dict, List, Optional

# The Plugboard service tier; also the ACL service identity CATs are minted for.
PLUGBOARD_TIER = "metamate_platform.plugboard"
DEFAULT_PIPELINE = "usecase-dev-ai"

_CAT_TIMEOUT_SECONDS = 3600
_CLIENT_ID = "agent_foundation.plugboard"


def _get_plugboard_cats() -> str:
    """Mint and serialize CAT tokens scoped to the Plugboard service identity."""
    import py3_asyncio.infrasec.authorization.acl.thrift_types as acl_constants
    from corp_crypto_auth_token_util import CryptoAuthTokenUtil
    from py3_asyncio.infrasec.authorization.acl.thrift_types import Identity

    identity = Identity(
        id_type=acl_constants.SERVICE_IDENTITY,
        id_data=PLUGBOARD_TIER,
    )
    return CryptoAuthTokenUtil.serialize_crypto_auth_token_list(
        CryptoAuthTokenUtil.get_all_crypto_auth_tokens(
            identity,
            token_timeout_seconds=_CAT_TIMEOUT_SECONDS,
        )
    )


def _is_prod_network() -> bool:
    """True on a prod host — corp and cloud hosts read as non-prod."""
    import os

    path = "/etc/fbwhoami"
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            for line in f:
                parts = line.strip().split("=", 1)
                if len(parts) == 2:
                    var, value = parts
                    if var == "DEVICE_HOSTNAME_SCHEME" and value.startswith("corp_"):
                        return False
                    if var == "CLOUD_PROVIDER" and len(value) > 0:
                        return False
        return True
    except Exception:
        return False


def _has_service_router() -> bool:
    return platform.system() == "Linux" and _is_prod_network()


@contextlib.asynccontextmanager
async def _plugboard_thrift_client() -> AsyncGenerator[Any, None]:
    """Yield a CAT-authenticated async Thrift client for Plugboard."""
    from corp_crypto_auth_token_util import CryptoAuthTokenUtil
    from facebook.ai_productivity.plugboard.plugboard.thrift_clients import (
        AiProductivity_Plugboard,
    )

    headers = {CryptoAuthTokenUtil.CRYPTO_AUTH_TOKEN_HEADER: _get_plugboard_cats()}

    if _has_service_router():
        from servicerouter.python.async_client import get_sr_client
        from servicerouter.python.client_params import ClientParams

        params = ClientParams()
        params.setClientId(_CLIENT_ID)
        async with get_sr_client(
            AiProductivity_Plugboard, PLUGBOARD_TIER, params=params, headers=headers
        ) as client:
            yield client
    else:
        from x2p.secure_thrift.python.client import get_client

        async with get_client(
            AiProductivity_Plugboard, PLUGBOARD_TIER, headers=headers
        ) as client:
            yield client


class PlugboardClient:
    """Streaming LLM client over Meta's internal Plugboard gateway."""

    def __init__(
        self,
        pipeline: str = DEFAULT_PIPELINE,
        model_pipeline_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        self.pipeline = pipeline
        self.model_pipeline_overrides: Dict[str, str] = model_pipeline_overrides or {}

    def _get_pipeline_for_model(self, model: str) -> str:
        return self.model_pipeline_overrides.get(model, self.pipeline)

    async def stream_response(
        self,
        messages: List[Dict[str, str]],
        system: str,
        model: str,
        max_tokens: int,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Yield assistant text chunks for one completion."""
        from facebook.ai_productivity.plugboard.plugboard.thrift_types import (
            ContentPart,
            Message,
            ModelParams,
            RunPipelineRequest,
        )
        from facebook.ai_productivity.stream_defs.thrift_types import STREAM_ASSISTANT

        pb_messages: List[Any] = [
            Message(role="system", content_parts=[ContentPart(text=system)])
        ]
        for msg in messages:
            pb_messages.append(
                Message(
                    role=msg["role"],
                    content_parts=[ContentPart(text=msg["content"])],
                )
            )

        # ``temperature`` is OMITTED when None rather than sent as a default: the
        # newer Claude deployments reject it outright ("`temperature` is
        # deprecated for this model", InvalidRequestException), so a caller that
        # does not care about sampling must be able to not set the field at all.
        model_param_kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            model_param_kwargs["temperature"] = temperature

        request = RunPipelineRequest(
            history=pb_messages,
            pipeline=self._get_pipeline_for_model(model),
            model_params=ModelParams(**model_param_kwargs),
            request_correlator=f"{_CLIENT_ID}~{uuid.uuid4()}",
        )

        async with _plugboard_thrift_client() as ctx:
            (_init, stream) = await ctx.run_pipeline_streaming(request)
            async for chunk in stream:
                if chunk.stream_id == STREAM_ASSISTANT and chunk.message.content:
                    yield chunk.message.content

    async def close(self) -> None:
        """No persistent resources — the Thrift client is per-call scoped."""
