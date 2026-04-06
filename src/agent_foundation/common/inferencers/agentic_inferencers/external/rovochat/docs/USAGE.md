# RovoChat Inferencer — Usage Guide

Practical guide for using the RovoChat inferencer with examples.

---

## Quick Start

### Zero-Config (with existing JIRA env vars)

If your environment already has `JIRA_URL`, `JIRA_EMAIL`, and `JIRA_API_TOKEN`:

```python
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat import (
    RovoChatInferencer,
)

inferencer = RovoChatInferencer()
result = inferencer("What is Atlassian Rovo?")
print(result)
```

### Explicit Configuration

```python
inferencer = RovoChatInferencer(
    base_url="https://hello.atlassian.net",
    email="you@atlassian.com",
    api_token="your-api-token",
)
result = inferencer("How to check recent Confluence pages?")
```

---

## Authentication Modes

### 1. Basic Auth (Recommended)

Simplest setup. Uses your Atlassian email and API token via the site gateway.
Get an API token at: https://id.atlassian.com/manage-profile/security/api-tokens

```python
inferencer = RovoChatInferencer(
    base_url="https://hello.atlassian.net",
    email="you@atlassian.com",
    api_token="your-api-token",
)
```

Or via environment variables:
```bash
export ROVOCHAT_EMAIL="you@atlassian.com"
export ROVOCHAT_API_TOKEN="your-api-token"
export ROVOCHAT_BASE_URL="https://hello.atlassian.net"
```

Fallback env vars also work: `JIRA_EMAIL`, `JIRA_API_TOKEN`, `JIRA_URL`.

### 2. UCT Token

For staging or when you have a pre-generated User-Context Token:

```python
inferencer = RovoChatInferencer(
    base_url="https://convo-ai.us-east-1.staging.atl-paas.net",
    cloud_id="your-cloud-id",
    uct_token="eyJ...",
)
```

Mint a UCT via atlas CLI:
```bash
atlas asap token -c user-context-for-testing.json -a convo-ai
```

### 3. ASAP JWT

For service-to-service with a registered ASAP keypair:

```python
inferencer = RovoChatInferencer(
    base_url="https://convo-ai.us-east-1.staging.atl-paas.net",
    cloud_id="your-cloud-id",
    asap_issuer="micros/your-service",
    asap_private_key="-----BEGIN RSA PRIVATE KEY-----...",
    asap_key_id="your-key-id",
)
```

Requires `atlassian-jwt-auth` package: `pip install atlassian-jwt-auth`

---

## Streaming

### Async Streaming

```python
async for chunk in inferencer.ainfer_streaming("Explain Rovo agents"):
    print(chunk, end="", flush=True)
```

### Sync Streaming

```python
for chunk in inferencer.infer_streaming("Explain Rovo agents"):
    print(chunk, end="", flush=True)
```

---

## Multi-Turn Conversations

### Auto-Resume

After the first query, subsequent queries auto-resume the same conversation:

```python
r1 = inferencer("I'm building a Jira integration")
r2 = inferencer("What APIs should I use?")  # Same conversation!
```

### Explicit Session Management

```python
# Start new session
r1 = inferencer.new_session("I'm working on project X")

# Resume specific session
r2 = inferencer("Tell me more", session_id=r1.session_id)

# Force new session
r3 = inferencer("Different topic", new_session=True)

# Reset
inferencer.reset_session()
```

### SDKInferencerResponse

Get structured response with session info:

```python
response = await inferencer.ainfer(
    "What is Rovo?",
    return_sdk_response=True,
)
print(response.content)        # Response text
print(response.session_id)     # Conversation ID for resume
print(response.tokens_received) # Approximate token count
```

---

## Using the Client Directly

For lower-level control, use `RovoChatClient`:

```python
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat import (
    RovoChatAuth, RovoChatClient, RovoChatConfig,
)

auth = RovoChatAuth(email="you@atlassian.com", api_token="...")
config = RovoChatConfig(
    base_url="https://hello.atlassian.net",
    use_gateway=True,
)
client = RovoChatClient(config=config, auth=auth)

# Create conversation
conv = await client.create_conversation()

# Stream events
async for event in client.send_message_stream(conv.conversation_id, "Hello"):
    print(f"[{event.event_type}] {event.data}")

# Get full response
response = await client.send_message(conv.conversation_id, "Tell me about Rovo")
print(response.content)
print(f"Citations: {len(response.citations)}")
```

---

## Targeting a Specific Agent

```python
inferencer = RovoChatInferencer(
    base_url="https://hello.atlassian.net",
    email="you@atlassian.com",
    api_token="...",
    agent_named_id="your-agent-uuid",
)
```

Or override per-call:

```python
async for chunk in inferencer.ainfer_streaming(
    "Analyze this code",
    agent_named_id="code-review-agent-uuid",
):
    print(chunk, end="")
```

---

## Configuration Reference

### RovoChatInferencer Attributes

| Attribute | Default | Description |
|-----------|---------|-------------|
| `base_url` | From env or staging URL | RovoChat API base URL |
| `cloud_id` | From env or `""` | Atlassian Cloud ID (optional for gateway) |
| `email` | From env | Atlassian email for Basic Auth |
| `api_token` | From env | API token for Basic Auth |
| `uct_token` | From env | Pre-generated UCT token |
| `asap_issuer` | From env | ASAP token issuer |
| `asap_private_key` | From env | ASAP RSA private key |
| `asap_key_id` | From env | ASAP key identifier |
| `agent_named_id` | `""` | Target Rovo agent ID |
| `agent_id` | `""` | Alternative agent ID |
| `lanyard_config` | `""` | Lanyard config (staging only) |
| `product` | `"rovo"` | X-Product header value |
| `experience_id` | `"ai-mate"` | X-Experience-Id header value |
| `store_message` | `True` | Persist messages server-side |
| `citations_enabled` | `True` | Request citations |
| `auto_continue` | `True` | Auto-reply to clarification questions |
| `max_continuations` | `5` | Max auto-continue rounds |
| `total_timeout_seconds` | `1800` | Max total operation time |
| `idle_timeout_seconds` | `600` | Max idle time between chunks |

### Environment Variables

| Variable | Fallbacks | Description |
|----------|-----------|-------------|
| `ROVOCHAT_BASE_URL` | `JIRA_URL` | API base URL |
| `ROVOCHAT_CLOUD_ID` | — | Atlassian Cloud ID |
| `ROVOCHAT_EMAIL` | `JIRA_EMAIL`, `ATLASSIAN_EMAIL` | Account email |
| `ROVOCHAT_API_TOKEN` | `JIRA_API_TOKEN`, `ATLASSIAN_API_TOKEN` | API token |
| `ROVOCHAT_UCT_TOKEN` | — | Pre-generated UCT |
| `ROVOCHAT_ASAP_ISSUER` | — | ASAP issuer |
| `ROVOCHAT_ASAP_PRIVATE_KEY` | — | ASAP private key |
| `ROVOCHAT_ASAP_KEY_ID` | — | ASAP key ID |
| `ROVOCHAT_ASAP_AUDIENCE` | — | ASAP audience |

---

## Error Handling

```python
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat import (
    RovoChatInferencer,
    RovoChatAuthError,
    RovoChatConnectionError,
    RovoChatTimeoutError,
    RovoChatAPIError,
)

try:
    result = inferencer("query")
except RovoChatAuthError as e:
    print(f"Auth failed: {e}")
except RovoChatConnectionError as e:
    print(f"Cannot reach API: {e}")
except RovoChatTimeoutError as e:
    print(f"Timed out: {e}")
except RovoChatAPIError as e:
    print(f"API error (HTTP {e.status_code}): {e}")
```
