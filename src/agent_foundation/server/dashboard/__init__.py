"""Generic dashboard framework (backend).

Transport-agnostic schema + atomic JSON-sidecar store for the Dashboard
``tool_type``. The host app (OpenTeam) owns the WebSocket/FastAPI transport and
mounts thin routers that delegate to these (and the experiment_hub) services.

Import from the submodules directly (no module-scope side effects):
  from agent_foundation.server.dashboard.dashboard_protocol import DashboardManifest
  from agent_foundation.server.dashboard.dashboard_store import JsonSidecarStore
"""
