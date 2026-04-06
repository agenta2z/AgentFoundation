"""Gateway access mode detection and pre-flight checks for AI Gateway.

Supports three modes to reach the AI Gateway:
- direct: Shell out to `atlas slauth token` CLI, send requests via httpx
- proximity: Forward to localhost proximity proxy (handles auth internally)
- slauth_server: Use AI Gateway SDK with SlauthServerAuthFilter (existing approach)
- auto: Detect and cascade through available modes
"""

import logging
import shutil
import socket
import subprocess
from enum import StrEnum
from os import environ
from typing import Dict, Tuple

import httpx

logger = logging.getLogger(__name__)

# Default ports and URLs
DEFAULT_PROXIMITY_PORT = 29576
DEFAULT_SLAUTH_SERVER_URL = "http://localhost:5000"
DEFAULT_AI_GATEWAY_BASE_URL = "https://ai-gateway.us-east-1.staging.atl-paas.net"
DEFAULT_USE_CASE_ID = "ai-lab-agent"
DEFAULT_CLOUD_ID = "local"

# Cascade order for auto mode
_CASCADE_ORDER = ("direct", "proximity", "slauth_server")


class GatewayMode(StrEnum):
    """Enumeration of AI Gateway access modes."""

    DIRECT = "direct"
    PROXIMITY = "proximity"
    SLAUTH_SERVER = "slauth_server"
    AUTO = "auto"


# Mapping from Bedrock model IDs to Anthropic-native model names (for proximity mode)
BEDROCK_TO_ANTHROPIC_MODEL: Dict[str, str] = {
    "anthropic.claude-sonnet-4-5-20250929-v1:0": "claude-sonnet-4-5-20250929",
    "anthropic.claude-sonnet-4-20250514-v1:0": "claude-sonnet-4-20250514",
    "anthropic.claude-opus-4-6-v1": "claude-opus-4-6",
    "anthropic.claude-opus-4-1-20250805-v1:0": "claude-opus-4-1-20250805",
    "anthropic.claude-opus-4-20250514-v1:0": "claude-opus-4-20250514",
    "anthropic.claude-3-7-sonnet-20250219-v1:0": "claude-3-7-sonnet-20250219",
    "anthropic.claude-3-5-sonnet-20241022-v2:0": "claude-3-5-sonnet-v2@20241022",
    "anthropic.claude-3-5-sonnet-20240620-v1:0": "claude-3-5-sonnet-20240620",
    "anthropic.claude-haiku-4-5-20251001-v1:0": "claude-haiku-4-5-20251001",
    "anthropic.claude-3-5-haiku-20241022-v1:0": "claude-3-5-haiku-20241022",
}


def check_direct_available() -> Tuple[bool, str]:
    """Check if direct SLAuth token generation via atlas CLI is available.

    Verifies that the `atlas` CLI exists and can generate a token.

    Returns:
        Tuple of (available, reason). reason is empty string if available.
    """
    atlas_path = shutil.which("atlas")
    if not atlas_path:
        return False, "atlas CLI not found on PATH"

    try:
        result = subprocess.run(
            "atlas slauth token --aud=ai-gateway --env=staging --groups=atlassian-all --ttl 60m",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if not result.stdout.strip():
            return False, "atlas slauth token returned empty output"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "atlas slauth token timed out after 15s"
    except subprocess.CalledProcessError as e:
        return False, f"atlas slauth token failed: {e.stderr.strip() or str(e)}"
    except Exception as e:
        return False, f"atlas slauth token error: {e}"


def check_proximity_available(port: int = DEFAULT_PROXIMITY_PORT) -> Tuple[bool, str]:
    """Check if the proximity AI gateway proxy is running.

    Tries the healthcheck endpoint first, then falls back to a HEAD request
    on the Vertex Claude messages path (some proxy versions don't expose /healthcheck).

    Args:
        port: Port number for the proximity proxy.

    Returns:
        Tuple of (available, reason). reason is empty string if available.
    """
    base = f"http://localhost:{port}"
    try:
        # Try /healthcheck first (some versions support it)
        resp = httpx.get(f"{base}/healthcheck", timeout=2.0)
        if resp.status_code == 200:
            return True, ""
        # Fall back to HEAD on the vertex claude path
        resp = httpx.head(f"{base}/vertex/claude", timeout=2.0)
        # Any response (even 404) means the server is running
        return True, ""
    except httpx.ConnectError:
        return False, f"proximity proxy not reachable on localhost:{port}"
    except Exception as e:
        return False, f"proximity healthcheck error: {e}"


def check_slauth_server_available(url: str = DEFAULT_SLAUTH_SERVER_URL) -> Tuple[bool, str]:
    """Check if the atlas SLAuth server is running.

    Attempts a TCP socket connection to the server's host and port.

    Args:
        url: SLAuth server URL (e.g. "http://localhost:5000").

    Returns:
        Tuple of (available, reason). reason is empty string if available.
    """
    try:
        # Parse host and port from URL
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5000

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            return True, ""
        return False, f"SLAuth server not reachable on {host}:{port}"
    except Exception as e:
        return False, f"SLAuth server check error: {e}"


def detect_available_mode(
    proximity_port: int = DEFAULT_PROXIMITY_PORT,
    slauth_server_url: str = DEFAULT_SLAUTH_SERVER_URL,
) -> GatewayMode:
    """Detect the first available gateway mode by trying each in cascade order.

    Order: direct → proximity → slauth_server

    Args:
        proximity_port: Port for proximity proxy health check.
        slauth_server_url: URL for SLAuth server check.

    Returns:
        The first available GatewayMode.

    Raises:
        RuntimeError: If no gateway mode is available.
    """
    checks = {
        GatewayMode.DIRECT: lambda: check_direct_available(),
        GatewayMode.PROXIMITY: lambda: check_proximity_available(proximity_port),
        GatewayMode.SLAUTH_SERVER: lambda: check_slauth_server_available(slauth_server_url),
    }

    reasons = []
    for mode_str in _CASCADE_ORDER:
        mode = GatewayMode(mode_str)
        available, reason = checks[mode]()
        if available:
            logger.info(f"Auto-detected gateway mode: {mode}")
            return mode
        reasons.append(f"  {mode}: {reason}")
        logger.debug(f"Gateway mode {mode} not available: {reason}")

    raise RuntimeError(
        "No AI Gateway access mode is available. Tried:\n"
        + "\n".join(reasons)
        + "\n\nTo fix, do ONE of:\n"
        "  1. Install atlas CLI: atlas plugin install -n slauth\n"
        "  2. Start proximity proxy: proximity ai-gateway\n"
        "  3. Start SLAuth server: atlas slauth server --port 5000"
    )


def get_direct_slauth_token(env: str = "staging") -> str:
    """Generate a SLAuth token by shelling out to the atlas CLI.

    Args:
        env: Environment ("staging" or "prod").

    Returns:
        The SLAuth token string.

    Raises:
        subprocess.CalledProcessError: If the atlas command fails.
    """
    if env == "prod":
        cmd = "atlas slauth token --aud=ai-gateway --env=prod --ttl 60m"
    else:
        cmd = "atlas slauth token --aud=ai-gateway --env=staging --groups=atlassian-all --ttl 60m"

    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=30)
    return result.stdout.strip()


def build_direct_headers(
    token: str,
    user_id: str,
    cloud_id: str = DEFAULT_CLOUD_ID,
    use_case_id: str = DEFAULT_USE_CASE_ID,
) -> Dict[str, str]:
    """Build HTTP headers for direct AI Gateway access.

    Args:
        token: SLAuth token from atlas CLI.
        user_id: User identifier.
        cloud_id: Atlassian cloud ID.
        use_case_id: AI Gateway use case ID.

    Returns:
        Dict of HTTP headers.
    """
    return {
        "Content-Type": "application/json",
        "Authorization": f"SLAUTH {token}",
        "X-Atlassian-UserId": user_id,
        "X-Atlassian-CloudId": cloud_id,
        "X-Atlassian-UseCaseId": use_case_id,
    }


def bedrock_model_to_anthropic(bedrock_model: str) -> str:
    """Convert a Bedrock model ID to Anthropic-native model name.

    Used by proximity mode which expects Anthropic model names.

    Args:
        bedrock_model: Bedrock model ID (e.g. "anthropic.claude-opus-4-6-v1").

    Returns:
        Anthropic-native model name (e.g. "claude-opus-4-6").
    """
    if bedrock_model in BEDROCK_TO_ANTHROPIC_MODEL:
        return BEDROCK_TO_ANTHROPIC_MODEL[bedrock_model]
    # Fallback: strip "anthropic." prefix and version suffix
    name = bedrock_model
    if name.startswith("anthropic."):
        name = name[len("anthropic."):]
    # Remove trailing version like "-v1:0" or "-v2:0"
    for suffix in ["-v1:0", "-v2:0", "-v1", "-v2"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name
