# pyre-strict

"""OpenClaw inferencer shared constants, helpers, and exceptions.

Provides utilities for both CLI mode (Docker/kubectl subprocess) and
Gateway mode (WebSocket JSON-RPC) of the OpenClawInferencer.
"""

import json
import re
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional


# ─── Transport mode enum ───────────────────────────────────────────────────────

class OpenClawMode(str, Enum):
    """Transport mode for ``OpenClawInferencer``.

    Three distinct deployment configurations are supported:

    ``PodGateway``
        The OpenClaw gateway runs inside a Docker/kubectl pod (the default setup
        with ``./run.sh start``).  The inferencer connects to it via WebSocket on
        ``ws://127.0.0.1:18789`` (port-forwarded from the pod).

        - Auth token: auto-discovered from pod via ``docker exec → kubectl exec``
        - Session transcripts: stored inside pod at
          ``/sandbox/.openclaw/agents/main/sessions/``
        - New-session detection: checks pod filesystem via ``docker exec``
        - Supports true token-by-token streaming and persistent sessions

    ``LocalGateway``
        The OpenClaw gateway runs natively on the local machine
        (``openclaw gateway`` or launched by the OpenClaw Control UI).
        No Docker or kubectl involved.

        - Auth token: read from local ``~/.openclaw/openclaw.json``
        - Session transcripts: stored locally at
          ``~/.openclaw/agents/main/sessions/``
        - New-session detection: checks local filesystem directly
        - Supports true token-by-token streaming and persistent sessions

    ``PodCLI``
        No gateway.  The inferencer runs ``openclaw agent --local --json``
        inside the Docker/kubectl pod as a blocking subprocess.

        - No streaming (full response returned as one chunk)
        - No cross-run session persistence
        - Always works if the Docker container is running
        - Simplest option for one-shot queries

    Example::

        from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.common import OpenClawMode

        # Pod gateway (default):
        inf = OpenClawInferencer(mode=OpenClawMode.PodGateway)

        # Local gateway (Control UI running):
        inf = OpenClawInferencer.from_local_config()
        # equivalent to:
        inf = OpenClawInferencer(mode=OpenClawMode.LocalGateway, auth_token=token)

        # Pod CLI (simplest, no gateway needed):
        inf = OpenClawInferencer(mode=OpenClawMode.PodCLI)
    """

    PodGateway = "pod_gateway"
    """Gateway in Docker/kubectl pod — streaming, persistent sessions, token from pod."""

    LocalGateway = "local_gateway"
    """Gateway running locally — streaming, persistent sessions, token from ~/.openclaw."""

    PodCLI = "pod_cli"
    """CLI inside Docker/kubectl pod — blocking, no streaming, no session persistence."""


# ─── Docker / kubectl defaults ─────────────────────────────────────────────────

DEFAULT_DOCKER_CONTAINER: str = "openshell-cluster-openshell"
DEFAULT_KUBECTL_NAMESPACE: str = "openshell"
DEFAULT_KUBECTL_POD: str = "atlassian-openclaw-gateway"
DEFAULT_KUBECTL_CONTAINER: str = "agent"

# ─── OpenClaw config paths (inside the pod) ────────────────────────────────────

DEFAULT_OPENCLAW_CONFIG_PATH: str = "/sandbox/.openclaw/openclaw.json"
DEFAULT_OPENCLAW_STATE_DIR: str = "/sandbox/.openclaw"

# ─── Gateway connection ────────────────────────────────────────────────────────

DEFAULT_GATEWAY_URL: str = "ws://127.0.0.1:18789"

# Scopes required for the agent method (operator.write needed for agent calls).
# Using "openclaw-control-ui" client ID + Origin header grants these scopes
# with token auth (plain "cli" mode strips write scopes — no device identity).
GATEWAY_SCOPES: list[str] = [
    "operator.admin",
    "operator.read",
    "operator.write",
    "operator.approvals",
    "operator.pairing",
]

PROTOCOL_VERSION_MIN: int = 1
PROTOCOL_VERSION_MAX: int = 10

# ─── Default agent params ─────────────────────────────────────────────────────

DEFAULT_SESSION_ID: str = "main"
DEFAULT_TIMEOUT_SECONDS: int = 600

# ─── Retry / rate-limit detection ─────────────────────────────────────────────

# Substrings (lowercase) that indicate a rate limit / quota error.
# Used by retry logic to decide whether to retry with a continuation prompt.
RATE_LIMIT_SIGNALS: list[str] = [
    "rate_limit",
    "rate limit",
    "429",
    "quota exceeded",
    "resource exhausted",
    "too many requests",
    "overloaded",
    "capacity",
]

# ─── ANSI escape code pattern ──────────────────────────────────────────────────

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# Lines starting with these prefixes are stale plugin config warnings emitted
# by openclaw CLI to stdout — they must be stripped before JSON parsing.
_PLUGIN_WARNING_PREFIXES: tuple[str, ...] = (
    "- plugins.entries.",
    "Config warnings:",
    "  plugins:",
)


# ─── Exceptions ───────────────────────────────────────────────────────────────


class OpenClawError(RuntimeError):
    """Base exception for all OpenClaw inferencer errors."""


class OpenClawNotFoundError(OpenClawError):
    """Raised when Docker, kubectl, or the OpenClaw pod is unreachable."""

    def __init__(self, msg: Optional[str] = None) -> None:
        super().__init__(
            msg
            or (
                "OpenClaw is not reachable. Ensure the Docker container "
                f"'{DEFAULT_DOCKER_CONTAINER}' is running via './run.sh start'."
            )
        )


class OpenClawRateLimitError(OpenClawError):
    """Raised when the LLM backend returns a rate-limit / quota error."""


class OpenClawTimeoutError(OpenClawError):
    """Raised when no streaming event arrives within the configured timeout."""


class OpenClawAuthError(OpenClawError):
    """Raised when the gateway rejects the auth token."""


# ─── Utility: ANSI / noise stripping ─────────────────────────────────────────


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape sequences and carriage-return overwrites from text."""
    text = _ANSI_ESCAPE_RE.sub("", text)
    text = re.sub(r"\r[^\n]*", "", text)
    return text


def strip_plugin_warnings(text: str) -> str:
    """Remove stale plugin-config warning lines emitted by openclaw CLI.

    These appear in stdout before the JSON output when openclaw.json has
    deprecated or unknown plugin entries.  They must be stripped before
    attempting JSON extraction.
    """
    lines = [
        line
        for line in text.splitlines()
        if not any(line.startswith(prefix) for prefix in _PLUGIN_WARNING_PREFIXES)
    ]
    return "\n".join(lines)


# ─── Utility: JSON extraction ─────────────────────────────────────────────────


def extract_json_from_output(text: str) -> Optional[dict]:  # type: ignore[type-arg]
    """Extract the last valid JSON object from CLI stdout text.

    The ``--json`` flag causes openclaw to print a JSON blob at the end of
    stdout, after any TUI formatting noise.  Scans backward from the end of
    the text to find the last ``{...}`` block and validates with
    ``json.loads()``.

    Args:
        text: Raw stdout text (may contain ANSI codes / plugin warnings).

    Returns:
        Parsed dict if a valid JSON object is found, else ``None``.
    """
    text = text.rstrip()
    if not text.endswith("}"):
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[i:])  # type: ignore[return-value]
                except json.JSONDecodeError:
                    return None
    return None


def parse_cli_json_output(
    stdout: str,
    stderr: str,
    return_code: int,
) -> dict:  # type: ignore[type-arg]
    """Parse the JSON output from ``openclaw agent --local --json``.

    Strips ANSI codes and plugin warnings, then attempts JSON extraction.
    Falls back to plain-text extraction if JSON parsing fails.

    Args:
        stdout: Raw stdout from the subprocess.
        stderr: Raw stderr from the subprocess.
        return_code: Process exit code.

    Returns:
        Dict with keys: ``output``, ``raw_output``, ``return_code``,
        ``success``, ``session_id``, ``model``, ``usage``, ``error``.
    """
    # openclaw --local --json writes JSON to stderr (mixed with plugin warnings),
    # not to stdout. Try stdout first, then fall back to stderr.
    clean_stdout = strip_ansi_codes(strip_plugin_warnings(stdout))
    clean_stderr = strip_ansi_codes(strip_plugin_warnings(stderr))

    data = extract_json_from_output(clean_stdout) or extract_json_from_output(clean_stderr)
    raw_output = stdout if stdout.strip() else stderr

    if data:
        # openclaw --local --json format: top-level "payloads" array
        # (not nested under "result" — that's gateway mode format)
        payloads = data.get("payloads") or data.get("result", {}).get("payloads", [])
        agent_meta = data.get("meta", {}).get("agentMeta", {})
        output_text = "\n".join(
            p.get("text", "") for p in payloads if p.get("text")
        ).strip()
        status = data.get("status", "ok")
        return {
            "output": output_text,
            "raw_output": raw_output,
            "return_code": return_code,
            "success": return_code == 0 and not data.get("meta", {}).get("aborted", False),
            "session_id": agent_meta.get("sessionId"),
            "model": agent_meta.get("model"),
            "usage": agent_meta.get("usage", {}),
            "error": None,
        }

    # Fallback: return plain-text cleaned output
    clean = clean_stdout or clean_stderr
    return {
        "output": clean.strip(),
        "raw_output": raw_output,
        "return_code": return_code,
        "success": return_code == 0,
        "session_id": None,
        "model": None,
        "usage": {},
        "error": stderr.strip() if return_code != 0 else None,
    }


def is_rate_limit_error(message: str) -> bool:
    """Return ``True`` if *message* looks like a rate-limit / quota error."""
    lo = (message or "").lower()
    return any(sig in lo for sig in RATE_LIMIT_SIGNALS)


# ─── Subprocess helper ────────────────────────────────────────────────────────


def run_subprocess(
    cmd: str,
    timeout: int = 660,
) -> tuple[str, str, int]:
    """Run *cmd* in a shell and return ``(stdout, stderr, return_code)``.

    Args:
        cmd: Full shell command string.
        timeout: Maximum seconds to wait (default 660).

    Returns:
        Tuple of ``(stdout, stderr, return_code)``.

    Raises:
        subprocess.TimeoutExpired: If the process exceeds *timeout*.
    """
    proc = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.stdout, proc.stderr, proc.returncode


# ─── Token helpers ─────────────────────────────────────────────────────────────


def read_gateway_token_from_pod(
    docker_container: str = DEFAULT_DOCKER_CONTAINER,
    kubectl_namespace: str = DEFAULT_KUBECTL_NAMESPACE,
    kubectl_pod: str = DEFAULT_KUBECTL_POD,
    openclaw_config_path: str = DEFAULT_OPENCLAW_CONFIG_PATH,
    kubectl_container: str = DEFAULT_KUBECTL_CONTAINER,
) -> str:
    """Read the gateway auth token from the running OpenClaw pod.

    Executes a ``docker exec ... kubectl exec ... python3`` command to read
    ``gateway.auth.token`` from the pod's ``openclaw.json``.

    Args:
        docker_container: Name of the Docker container running k3s.
        kubectl_namespace: Kubernetes namespace of the OpenClaw pod.
        kubectl_pod: Name of the OpenClaw gateway pod.
        openclaw_config_path: Path to ``openclaw.json`` inside the pod.
        kubectl_container: Container name inside the pod (default ``agent``).

    Returns:
        The gateway auth token string.

    Raises:
        OpenClawNotFoundError: If the token cannot be read.
    """
    cmd = (
        f"docker exec {docker_container} "
        f"kubectl exec -n {kubectl_namespace} {kubectl_pod} "
        f"-c {kubectl_container} -- "
        f"python3 -c "
        f"\"import json; d=json.load(open('{openclaw_config_path}')); "
        f"print(d['gateway']['auth']['token'])\""
    )
    try:
        stdout, stderr, rc = run_subprocess(cmd, timeout=15)
        token = stdout.strip()
        if not token or rc != 0:
            raise OpenClawNotFoundError(
                f"Could not read gateway auth token from pod '{kubectl_pod}'. "
                f"stderr={stderr.strip()!r}"
            )
        return token
    except subprocess.TimeoutExpired as e:
        raise OpenClawNotFoundError(
            f"Timed out reading gateway auth token from pod '{kubectl_pod}'."
        ) from e


def read_skill_env_from_pod(
    skill_name: str = "twg",
    docker_container: str = DEFAULT_DOCKER_CONTAINER,
    kubectl_namespace: str = DEFAULT_KUBECTL_NAMESPACE,
    kubectl_pod: str = DEFAULT_KUBECTL_POD,
    kubectl_container: str = DEFAULT_KUBECTL_CONTAINER,
    openclaw_config_path: str = DEFAULT_OPENCLAW_CONFIG_PATH,
) -> dict:  # type: ignore[type-arg]
    """Read skill env vars from the OpenClaw pod config.

    Reads ``skills.entries.<skill_name>.env`` from ``openclaw.json`` in the pod.
    These env vars are configured in the openclaw config but NOT automatically
    injected into the agent subprocess environment in ``--local`` mode.

    Args:
        skill_name: Name of the skill (default ``"twg"``).
        docker_container: Docker container running k3s.
        kubectl_namespace: Kubernetes namespace.
        kubectl_pod: Pod name.
        kubectl_container: Container name.
        openclaw_config_path: Path to ``openclaw.json`` inside the pod.

    Returns:
        Dict of env var name → value. Empty dict if skill not found.
    """
    cmd = (
        f"docker exec {docker_container} "
        f"kubectl exec -n {kubectl_namespace} {kubectl_pod} "
        f"-c {kubectl_container} -- "
        f"python3 -c "
        f"\"import json; d=json.load(open('{openclaw_config_path}')); "
        f"print(json.dumps(d.get('skills',{{}}).get('entries',{{}}).get('{skill_name}',{{}}).get('env',{{}})))\""
    )
    try:
        stdout, _, rc = run_subprocess(cmd, timeout=15)
        if rc == 0 and stdout.strip():
            return json.loads(stdout.strip())
    except Exception:
        pass
    return {}


def read_gateway_token_from_config(
    openclaw_json_path: str = DEFAULT_OPENCLAW_CONFIG_PATH,
) -> str:
    """Read the gateway auth token from a local ``openclaw.json`` file.

    Args:
        openclaw_json_path: Path to a local ``openclaw.json`` file.

    Returns:
        The gateway auth token string.

    Raises:
        OpenClawNotFoundError: If the file cannot be read or token is missing.
    """
    try:
        config = json.loads(Path(openclaw_json_path).read_text(encoding="utf-8"))
        token = config["gateway"]["auth"]["token"]
        if not token:
            raise OpenClawNotFoundError(
                f"Empty gateway auth token in '{openclaw_json_path}'."
            )
        return str(token)
    except (OSError, KeyError, json.JSONDecodeError) as e:
        raise OpenClawNotFoundError(
            f"Could not read gateway auth token from '{openclaw_json_path}': {e}"
        ) from e


# ─── Availability checks ──────────────────────────────────────────────────────


def check_docker_available() -> None:
    """Verify that the ``docker`` binary is in PATH.

    Raises:
        OpenClawNotFoundError: If ``docker`` is not found.
    """
    if not shutil.which("docker"):
        raise OpenClawNotFoundError(
            "'docker' not found in PATH. Install Docker Desktop or Docker CLI."
        )


def check_gateway_reachable(
    gateway_url: str = DEFAULT_GATEWAY_URL,
    timeout: float = 5.0,
) -> None:
    """Verify that the OpenClaw gateway WebSocket port is reachable.

    Attempts a raw TCP connection to the gateway host:port.

    Args:
        gateway_url: WebSocket URL of the gateway (e.g. ``ws://127.0.0.1:18789``).
        timeout: Seconds to wait for TCP connection.

    Raises:
        OpenClawNotFoundError: If the port is not reachable.
    """
    import socket

    # Parse host:port from ws:// or wss:// URL
    url = gateway_url.replace("ws://", "").replace("wss://", "")
    host, _, port_str = url.partition(":")
    port = int(port_str.split("/")[0]) if port_str else 18789

    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as e:
        raise OpenClawNotFoundError(
            f"OpenClaw gateway not reachable at {gateway_url}. "
            "Ensure OpenClaw is running via './run.sh start'. "
            f"Error: {e}"
        ) from e
