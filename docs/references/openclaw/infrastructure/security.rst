.. _infrastructure-security:

========
Security
========

OpenClaw implements multiple security layers to protect user data and prevent
unauthorized access.

Authentication
==============

Gateway Auth Token
------------------

All gateway HTTP and WebSocket requests require an authentication token:

- Set via ``gateway.authToken`` in ``openclaw.json``
- Or via ``OPENCLAW_GATEWAY_AUTH_TOKEN`` environment variable
- Token is verified on every request

Credentials stored at ``~/.openclaw/credentials/`` for web provider OAuth.

Execution Approvals
===================

The exec approval system prevents agents from running dangerous commands
without explicit user consent:

- Configured via ``approvals`` section in ``openclaw.json``
- Commands can be auto-approved, require approval, or be denied
- The approval UI prompts the user via the Control UI or CLI

Sandbox Isolation
=================

Agents can run in sandboxed environments:

- **Docker sandbox**: Isolated container with controlled filesystem access
- **Mounted paths**: Configurable read/write paths into the sandbox
- **Tool restrictions**: Sandbox mode restricts available tools

Sandbox Dockerfiles:

- ``Dockerfile.sandbox`` — Basic sandbox
- ``Dockerfile.sandbox-browser`` — Sandbox with browser (Playwright)
- ``Dockerfile.sandbox-common`` — Shared base image

Rate Limiting
=============

- Auth profile cooldown tracking prevents API key abuse
- Per-session request serialization prevents concurrent agent runs
- Queue system prevents message flooding

Content Security Policy
========================

The gateway sets CSP headers for the Control UI to prevent XSS:

- Restricts script sources
- Prevents inline script execution
- Controls resource loading origins

Secrets Management
==================

- ``.secrets.baseline`` — detect-secrets baseline for pre-commit scanning
- Environment variables for sensitive configuration
- API keys stored in auth profiles (not in config file plaintext)
- Credential files at ``~/.openclaw/credentials/`` with restricted permissions

Security Reporting
==================

See ``SECURITY.md`` in the repository root for the vulnerability reporting
policy and security contact information.
