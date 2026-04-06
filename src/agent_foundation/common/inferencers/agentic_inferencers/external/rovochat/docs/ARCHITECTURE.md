# RovoChat Inferencer — Architecture & Framework Integration

This document describes how the RovoChat inferencer integrates with the
AgentFoundation framework and how it connects to the Atlassian RovoChat service.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Code                                    │
│                                                                     │
│  inferencer = RovoChatInferencer(email="...", api_token="...")       │
│  result = inferencer("How to check recent Confluence pages?")       │
│  async for chunk in inferencer.ainfer_streaming("Explain Rovo"):    │
│      print(chunk)                                                   │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    RovoChatInferencer                                 │
│                 (StreamingInferencerBase)                             │
│                                                                      │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │  _ainfer_stream  │  │  _ainfer()   │  │  _infer()             │  │
│  │  _ing()          │  │  session     │  │  sync via _run_async  │  │
│  │  yields text     │  │  management  │  │                        │  │
│  │  deltas          │  │  + SDK resp  │  │                        │  │
│  └────────┬─────────┘  └──────┬───────┘  └───────────┬────────────┘  │
│           │                   │                      │               │
│           └───────────────────┴──────────────────────┘               │
│                               │                                      │
│                               ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │                      RovoChatClient                            │   │
│  │                                                                │   │
│  │  create_conversation() ──► POST /conversation                 │   │
│  │  send_message_stream() ──► POST /conversation/{id}/msg/stream │   │
│  │  send_message()        ──► (accumulates stream)               │   │
│  │                                                                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────┐ │   │
│  │  │ RovoChatAuth │  │RovoChatConfig│  │  common.py helpers   │ │   │
│  │  │ Basic/UCT/   │  │ URL, gateway│  │  ADF, NDJSON, events │ │   │
│  │  │ ASAP         │  │ cloud_id    │  │  continuation detect │ │   │
│  │  └──────┬───────┘  └──────┬──────┘  └───────────────────────┘ │   │
│  └─────────┼─────────────────┼───────────────────────────────────┘   │
│            │                 │                                        │
└────────────┼─────────────────┼────────────────────────────────────────┘
             │                 │
             ▼                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          HTTP (httpx)                                  │
│                                                                        │
│  Gateway Mode (production):                                            │
│    https://hello.atlassian.net/gateway/api/assist/rovo/v1/chat/...     │
│    Authorization: Basic base64(email:token)                            │
│    x-cloudid, x-product, x-experience-id (lowercase)                  │
│                                                                        │
│  Direct Mode (staging):                                                │
│    https://convo-ai.us-east-1.staging.atl-paas.net/api/rovo/v1/chat/  │
│    Authorization: Bearer <token>                                       │
│    User-Context: <token>                                               │
│    Atl-Cloudid, X-Product, X-Experience-Id (capitalized)               │
│    Lanyard-Config: <config-id>                                         │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  Atlassian RovoChat Service                            │
│              (conversational-ai-platform)                              │
│                                                                        │
│  Kotlin/Spring Boot backend with NDJSON streaming                      │
│  Endpoints: conversation CRUD, message/stream, realtime WebSocket      │
│  Uses ADF for message format, returns NDJSON events                    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Class Hierarchy

```
rich_python_utils.Debuggable
  └── InferencerBase (abstract)
        │   _infer() [abstract]
        │   infer(), iter_infer(), parallel_infer()
        │   ainfer(), aiter_infer(), aparallel_infer()
        │
        └── StreamingInferencerBase
              │   _ainfer_streaming() [abstract]
              │   ainfer_streaming() — with idle timeout + caching
              │   infer_streaming() — sync bridge via thread + queue
              │   _ainfer() — accumulates from ainfer_streaming()
              │   session management (new/resume/reset)
              │
              └── RovoChatInferencer  ◄── THIS PACKAGE
                    │
                    ├── _ainfer_streaming()  → creates RovoChatClient,
                    │                          streams NDJSON, yields deltas
                    ├── _ainfer()            → session management,
                    │                          SDKInferencerResponse support
                    └── _infer()            → sync wrapper via _run_async()
```

---

## Module Dependency Graph

```
__init__.py ──imports──► rovochat_inferencer.py
                              │
                              ├──► client.py
                              │       ├──► auth.py
                              │       ├──► common.py
                              │       ├──► types.py
                              │       └──► exceptions.py
                              │
                              ├──► auth.py
                              │       ├──► common.py (env var constants)
                              │       └──► exceptions.py
                              │
                              ├──► common.py
                              │       └──► types.py (StreamEvent)
                              │
                              └──► types.py (RovoChatConfig)

External dependencies:
  ├── httpx                     (soft — in client.py)
  ├── attrs                     (hard — in rovochat_inferencer.py)
  ├── rich_python_utils.get__   (hard — in auth.py, rovochat_inferencer.py)
  ├── StreamingInferencerBase   (hard — in rovochat_inferencer.py)
  ├── SDKInferencerResponse     (hard — in rovochat_inferencer.py)
  └── atlassian-jwt-auth        (soft — in auth.py, ASAP mode only)
```

---

## Data Flow: Streaming Query

```
User: inferencer("How to check Confluence pages?")
  │
  ├─1─► _infer() calls _run_async(_ainfer())
  │
  ├─2─► _ainfer() resolves session (new/resume conversation)
  │     calls super()._ainfer() which calls _ainfer_streaming()
  │
  ├─3─► _ainfer_streaming():
  │     ├── Creates RovoChatClient with RovoChatAuth + RovoChatConfig
  │     ├── client.create_conversation() → POST, gets conversation_id
  │     ├── client.send_message_stream(conv_id, prompt):
  │     │   ├── Wraps text in ADF via build_adf_message()
  │     │   ├── Sends POST with NDJSON Accept header
  │     │   └── Yields StreamEvent objects from response lines
  │     │
  │     ├── For each StreamEvent:
  │     │   ├── Skip if event_type in _NON_CONTENT_EVENT_TYPES
  │     │   ├── extract_text_from_event() → text or None
  │     │   ├── Compute delta (incremental or accumulated)
  │     │   └── yield delta
  │     │
  │     └── If auto_continue and needs_continuation(text):
  │         └── Send AUTO_CONTINUE_REPLY, continue streaming
  │
  ├─4─► _ainfer() accumulates all deltas into response text
  │     Optionally wraps in SDKInferencerResponse
  │
  └─5─► Returns response string (or SDKInferencerResponse)
```

---

## Authentication Modes

```
                    ┌────────────────────────┐
                    │     RovoChatAuth       │
                    │                        │
                    │  email + api_token ─────┼──► Basic Auth header
                    │       (Priority 1)     │    Authorization: Basic ...
                    │                        │
                    │  uct_token ─────────────┼──► Bearer + User-Context
                    │       (Priority 2)     │    Authorization: Bearer ...
                    │                        │    User-Context: ...
                    │                        │
                    │  asap_issuer + key ─────┼──► ASAP JWT generation
                    │       (Priority 3)     │    (via atlassian-jwt-auth)
                    │                        │
                    │  Environment fallback: │
                    │  ROVOCHAT_* ──────────►│
                    │  JIRA_* ──────────────►│    (via get__() multi-key)
                    │  ATLASSIAN_* ─────────►│
                    └────────────────────────┘
```

---

## Connection to Original Codebases

### conversational-ai-platform (Kotlin)

The RovoChat inferencer is a **pure HTTP client** that calls the same REST API
that the Rovo Chat UI uses in the browser. It does NOT depend on any code from
the conversational-ai-platform repository.

Key files in conversational-ai-platform that define the API contract:
- `modules/product/rovo/rovo-impl/.../rest/RovoChatV1Controller.kt` — REST controller
- `modules/product/rovo/rovo-api/.../rest/RovoChatMessageStreamRequest.kt` — Request model
- `http/rovo/rovo_chat_with_agent_uct.http` — HTTP examples
- `scripts/devloop/aifc.py` — Python dev script showing gateway auth pattern

### acra-python / acra-python-lab

Referenced for:
- ASAP token generation patterns (`nemo/utils/ai_gateway.py`)
- HTTP client patterns (`httpx.AsyncClient` usage)
- Environment variable naming conventions (`JIRA_EMAIL`, etc.)

### AgentFoundation Framework

The inferencer extends:
- `StreamingInferencerBase` — provides streaming, sync bridge, session management
- `SDKInferencerResponse` — shared response type across external inferencers
- Uses `_run_async()` from `rich_python_utils` for sync wrapper
- Uses `get__()` from `rich_python_utils` for multi-key env var lookups
