# RovoChat Standalone Inferencer — Investigation Report & Implementation Plan

**Date**: 2026-04-05
**Author**: Rovo Dev (AI-assisted investigation)
**Status**: ✅ Implemented & E2E tested against production

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Investigation: MetaMate Inferencer Pattern](#2-investigation-metamate-inferencer-pattern)
3. [Investigation: AgentFoundation Inferencer Framework](#3-investigation-agentfoundation-inferencer-framework)
4. [Investigation: RovoChat API (conversational-ai-platform)](#4-investigation-rovochat-api)
5. [Investigation: ACRA Python SDK & Auth Patterns](#5-investigation-acra-python-sdk--auth-patterns)
6. [Investigation: Live Testing Discoveries](#6-investigation-live-testing-discoveries)
7. [Gap Analysis & Key Challenges](#7-gap-analysis--key-challenges)
8. [Implementation Plan (As Executed)](#8-implementation-plan)
9. [Key Design Decisions](#9-key-design-decisions)
10. [Resolved Questions](#10-resolved-questions)

---

## 1. Executive Summary

**Objective**: Create a standalone RovoChat inferencer package under
`external/rovochat/` that can query Atlassian's RovoChat conversational AI
service with natural language queries.

**Approach**: Follow the MetaMate inferencer pattern — extend `StreamingInferencerBase`,
use async HTTP streaming (NDJSON) via `httpx`, and provide a clean standalone
package with no hard dependency on the conversational-ai-platform codebase.

**Result**: Fully working inferencer with 3 auth modes, gateway + direct URL
support, NDJSON streaming with event filtering, and zero-config capability
via environment variables. E2E tested against production (`hello.atlassian.net`).

**Key Discovery**: RovoChat is accessible via two paths:
1. **Direct** (staging): `https://convo-ai.us-east-1.staging.atl-paas.net/api/rovo/v1/chat/...`
2. **Gateway** (production): `https://<site>.atlassian.net/gateway/api/assist/rovo/v1/chat/...`

The gateway path with Basic Auth (`email:api_token`) is the simplest fully
automatic end-to-end flow — no token minting required.

---

## 2. Investigation: MetaMate Inferencer Pattern

### 2.1 Architecture Overview

The MetaMate inferencer at `external/metamate/` is a dual-interface integration:
```
external/metamate/
├── __init__.py                  # Package exports
├── metamate_sdk_inferencer.py   # Async SDK inferencer (StreamingInferencerBase)
├── metamate_cli_inferencer.py   # CLI subprocess inferencer (TerminalSessionInferencerBase)
├── common.py                   # Shared constants, enums, parsing utilities
├── types.py                    # All dataclass/enum type definitions
├── exceptions.py               # Custom exception hierarchy
├── query_metamate.py           # CLI-only entry point
├── adapters/                   # Higher-level research/knowledge adapters
└── clients/                    # Client abstraction layer (production, mock, fallback)
```

### 2.2 Key MetaMate Patterns We Followed

- **`StreamingInferencerBase`** extension with `_ainfer_streaming()` as the core primitive
- **Session management**: mapping `session_id` → conversation identifier
- **Auto-continuation**: detecting clarification questions and auto-replying
- **`SDKInferencerResponse`**: shared response type for structured returns
- **`common.py`** for constants, parsing, and continuation detection
- **`types.py`** for domain-specific dataclasses
- **`exceptions.py`** for custom exception hierarchy

### 2.3 Key MetaMate Patterns We Diverged From

- **No CLI inferencer**: MetaMate wraps a Buck build target subprocess. RovoChat
  has no CLI equivalent — HTTP is the natural interface.
- **No `clients/` subdirectory**: MetaMate has production/mock/fallback clients.
  Our single `client.py` is sufficient for a pure HTTP client.
- **No adapters**: MetaMate has research/knowledge/debug adapters. These can be
  added later for agent-specific workflows.
- **True streaming vs polling**: MetaMate polls `get_conversation_for_stream()`.
  We stream NDJSON lines directly from HTTP response.

---

## 3. Investigation: AgentFoundation Inferencer Framework

### 3.1 Class Hierarchy

```
Debuggable (rich_python_utils)
  └── InferencerBase (abstract)
        ├── _infer() [abstract]
        ├── infer(), iter_infer(), parallel_infer()
        ├── ainfer()
        └── StreamingInferencerBase
              ├── _ainfer_streaming() [abstract] — yields text chunks
              ├── ainfer_streaming() — with idle timeout + caching
              ├── infer_streaming() — sync bridge via thread + queue
              ├── _ainfer() — accumulates from ainfer_streaming()
              ├── new_session() / anew_session()
              └── resume_session() / aresume_session()
```

### 3.2 Required Implementation

A new external inferencer must implement:
1. `_ainfer_streaming(prompt, **kwargs) -> AsyncIterator[str]` — yield text chunks
2. `_infer(inference_input, inference_config, **kwargs)` — sync inference

### 3.3 SDKInferencerResponse

```python
@attrs
class SDKInferencerResponse:
    content: str
    session_id: Optional[str]
    tool_uses: int
    tokens_received: int
    raw_response: Optional[Any]
```

---

## 4. Investigation: RovoChat API (conversational-ai-platform)

### 4.1 Technology Stack

- **Backend**: Kotlin on Spring Boot
- **Source**: `atlassian_packages/conversational-ai-platform/`
- **Controller**: `modules/product/rovo/rovo-impl/.../rest/RovoChatV1Controller.kt`

### 4.2 API Endpoints — Two Path Variants

#### Direct Path (staging/internal)
```
POST /api/rovo/v1/chat/conversation
POST /api/rovo/v1/chat/conversation/{id}/message/stream
```

#### Gateway Path (production, via Atlassian site)
```
POST /gateway/api/assist/rovo/v1/chat/conversation
POST /gateway/api/assist/rovo/v1/chat/conversation/{id}/message/stream
```

### 4.3 Request Format

```http
POST /gateway/api/assist/rovo/v1/chat/conversation/{id}/message/stream
Content-Type: application/json
Authorization: Basic base64(email:api_token)
x-cloudid: <cloud-id>          # optional via gateway
x-product: rovo
x-experience-id: ai-mate

{
  "content": {
    "version": 1,
    "type": "doc",
    "content": [{
      "type": "paragraph",
      "content": [{"type": "text", "text": "Your message"}]
    }]
  },
  "mimeType": "text/adf",
  "store_message": true,
  "citations_enabled": true,
  "context": {}
}
```

### 4.4 NDJSON Response Event Types (Discovered via Live Testing)

| Event Type | Content? | Description |
|-----------|----------|-------------|
| `ANSWER_PART` | ✅ Yes | Incremental text chunks (majority of events) |
| `FINAL_RESPONSE` | ✅ Yes | Complete response with sources/citations |
| `TRACE` | ❌ No | Search queries, document lookups (metadata) |
| `HEART_BEAT` | ❌ No | Keep-alive pings |
| `CONVERSATION_CHANNEL_DATA` | ❌ No | Conversation metadata |

**Critical finding**: `TRACE` and `HEART_BEAT` events contain a `message.content`
field, but it holds metadata (search queries, document titles) — NOT response text.
The inferencer must filter these non-content event types to avoid polluting output.

### 4.5 ADF (Atlassian Document Format)

Minimal text message:
```json
{
  "version": 1,
  "type": "doc",
  "content": [{
    "type": "paragraph",
    "content": [{"type": "text", "text": "Hello, RovoChat!"}]
  }]
}
```

---

## 5. Investigation: ACRA Python SDK & Auth Patterns

### 5.1 Three Authentication Modes

| Mode | Mechanism | When to Use |
|------|-----------|-------------|
| **Basic Auth** | `Authorization: Basic base64(email:api_token)` | Production via gateway — simplest, no token minting |
| **UCT Token** | `User-Context: <jwt>` + `Authorization: Bearer <jwt>` | Staging with pre-minted token |
| **ASAP JWT** | `Authorization: Bearer <asap-jwt>` | Service-to-service with registered keypair |

### 5.2 UCT Token Minting (for staging)

UCT tokens can be minted via:
- `atlas asap token -c user-context-for-testing.json -a convo-ai`
- Browser DevTools (copy `User-Context` header from Rovo Chat API calls)
- `./gradlew setUserContext` in the conversational-ai-platform dev environment

### 5.3 Environment Variable Resolution

The inferencer resolves credentials from env vars with fallback chains:

| Parameter | Primary | Fallbacks |
|-----------|---------|-----------|
| `email` | `ROVOCHAT_EMAIL` | `JIRA_EMAIL` → `ATLASSIAN_EMAIL` |
| `api_token` | `ROVOCHAT_API_TOKEN` | `JIRA_API_TOKEN` → `ATLASSIAN_API_TOKEN` |
| `base_url` | `ROVOCHAT_BASE_URL` | `JIRA_URL` (extracts site domain) |
| `cloud_id` | `ROVOCHAT_CLOUD_ID` | *(none — optional for gateway)* |

Uses `get__()` from `rich_python_utils.common_utils.map_helper` for multi-key lookups.

---

## 6. Investigation: Live Testing Discoveries

### 6.1 Authentication Journey

| Attempt | Auth Method | URL | Result |
|---------|-------------|-----|--------|
| 1 | ASAP (`micros/tony-dev`) | Staging direct | ❌ 401 — key not registered |
| 2 | UCT from `user-context-for-testing.json` | Staging direct | ❌ 403 — Rovo not enabled on staging cloud |
| 3 | Basic Auth (`email:api_token`) | Production gateway | ✅ **Success!** |

### 6.2 Streaming Behavior (from production)

- **Query**: "how to obtain recent Confluence pages of a person through CLI or API?"
- **Response**: 3,886 chars, comprehensive answer with CQL examples and curl patterns
- **Events**: ~1,000 total (982 `ANSWER_PART`, 16 `TRACE`, 3 `HEART_BEAT`, 1 `FINAL_RESPONSE`, 1 `CONVERSATION_CHANNEL_DATA`)
- **Streaming**: Incremental chunks (each `ANSWER_PART` is a new text fragment, not accumulated)
- **Multi-turn**: Follow-up messages in same conversation retain context

### 6.3 Critical Bug Fixed: TRACE Text Leaking

Initial implementation extracted text from ALL events. `TRACE` events contain
search queries and document titles in `message.content`, causing metadata to
leak into response text (e.g., "Searching: confluence api recent pages").

Fix: Added `_NON_CONTENT_EVENT_TYPES` filter in `extract_text_from_event()`.

### 6.4 Gateway Discovery

The production path uses `/gateway/api/assist/...` (not `/api/rovo/v1/...`),
with lowercase headers (`x-cloudid`, `x-product`). The `cloud_id` is optional
via gateway (inferred from site URL). This was discovered from `aifc.py` in the
conversational-ai-platform's dev scripts.

---

## 7. Gap Analysis & Key Challenges

### 7.1 Differences from MetaMate Pattern

| Aspect | MetaMate | RovoChat |
|--------|----------|----------|
| **Transport** | GraphQL SDK (synchronous, polled) | REST HTTP (async, NDJSON streaming) |
| **Auth** | API key + optional CAT token | Basic Auth / UCT / ASAP (3 modes) |
| **Streaming** | Poll `get_conversation_for_stream()` | Stream NDJSON lines directly |
| **SDK Dependency** | `msl.metamate.cli.metamate_graphql` | None — pure HTTP via `httpx` |
| **Message Format** | Plain text prompt | ADF (Atlassian Document Format) |
| **Session ID** | `conversation_uuid` + `conversation_fbid` | `conversationId` (single ID) |
| **URL Pattern** | Single endpoint | Gateway + Direct (two variants) |

### 7.2 Challenges Solved

1. **NDJSON event filtering**: `TRACE`/`HEART_BEAT` carry metadata, not response text
2. **ADF construction**: Transparent wrapping of plain text into ADF documents
3. **Gateway vs Direct**: Auto-detected based on base URL domain
4. **Auth flexibility**: Three modes with env var fallback chains
5. **Delta logic**: Handles both incremental and accumulated text patterns

---

## 8. Implementation Plan (As Executed)

### 8.1 Final File Structure

```
external/rovochat/
├── __init__.py                  # Package exports (96 lines)
├── exceptions.py                # Exception hierarchy (82 lines)
├── types.py                     # Config, StreamEvent, Response types (132 lines)
├── common.py                    # Constants, env vars, ADF, NDJSON parsing (380 lines)
├── auth.py                      # Basic Auth + UCT + ASAP (260 lines)
├── client.py                    # Gateway + Direct HTTP client (460 lines)
├── rovochat_inferencer.py       # StreamingInferencerBase implementation (400 lines)
└── docs/
    ├── INVESTIGATION_AND_PLAN.md # This document
    ├── ARCHITECTURE.md           # System architecture and framework integration
    └── USAGE.md                  # Practical usage guide with examples
```

### 8.2 Dependencies

- `httpx` — async HTTP client with streaming (soft dependency)
- `attrs` — class definition (used throughout AgentFoundation)
- `rich_python_utils` — `get__()` for multi-key env var lookup; `_run_async()` for sync wrapper
- `atlassian-jwt-auth` — ASAP token generation (soft/optional, only for ASAP mode)
- Standard library: `asyncio`, `json`, `logging`, `uuid`, `time`, `dataclasses`, `base64`, `os`

---

## 9. Key Design Decisions

1. **Three auth modes**: Basic Auth is simplest (zero token minting). UCT for staging. ASAP for service-to-service. Priority: Basic > UCT > ASAP.
2. **Gateway auto-detection**: If Basic Auth + `.atlassian.net` URL → gateway mode enabled automatically.
3. **HTTP streaming (not polling)**: NDJSON lines streamed directly — more efficient than MetaMate's polling.
4. **Non-content event filtering**: `TRACE`, `HEART_BEAT`, `CONVERSATION_CHANNEL_DATA` filtered from text extraction.
5. **ADF abstraction**: Users pass plain strings; ADF wrapping is transparent.
6. **Session = Conversation ID**: `session_id` maps to `conversationId`.
7. **Env var fallback chains**: `ROVOCHAT_*` → `JIRA_*` → `ATLASSIAN_*` using `get__()`.
8. **Zero-config capable**: With `JIRA_URL` + `JIRA_EMAIL` + `JIRA_API_TOKEN` env vars, the inferencer works with no constructor arguments.

---

## 10. Resolved Questions

| Question (from original plan) | Answer |
|-------------------------------|--------|
| Base URL? | Staging: `https://convo-ai.us-east-1.staging.atl-paas.net`. Production: via gateway on `https://<site>.atlassian.net` |
| Lanyard Config? | Only needed for direct staging path. Not needed for gateway. |
| NDJSON Event Types? | `ANSWER_PART` (text), `FINAL_RESPONSE` (complete), `TRACE` (metadata), `HEART_BEAT` (ping), `CONVERSATION_CHANNEL_DATA` (meta) |
| Agent Named ID? | Optional. Empty string → default Rovo agent. |
| Rate Limiting? | Not observed during testing. |
| Cloud ID required? | Optional via gateway (inferred from site URL). Required for direct path. |
| Dependencies hard or soft? | `httpx` is soft (ImportError at runtime). `atlassian-jwt-auth` is soft (only for ASAP mode). `rich_python_utils` and `attrs` are hard (framework deps). |
