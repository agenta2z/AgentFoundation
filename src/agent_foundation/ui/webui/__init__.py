"""DEPRECATED — standalone RankEvolve-webui demo copy.

This package is a reference-only snapshot of RankEvolve's WebUI. It is NOT used
by OpenTeam and is retained only for parity comparison during the Experiment Hub
port. Do not add new dependencies on it.

The Experiment Hub now lives, first-class, in:
  * Frontend: ``agent_foundation/ui/react-shared/src/dashboards/experiment_hub/``
    (generic Dashboard framework in ``react-shared/src/dashboard/``)
  * Backend:  ``agent_foundation/experiment_hub/`` (OpenTeam-agnostic services +
    ``HubController``), surfaced via OpenTeam's REST routers + WebSocket
    ``dashboard_*`` protocol.

Slated for removal once Experiment Hub parity is confirmed end-to-end.
"""
