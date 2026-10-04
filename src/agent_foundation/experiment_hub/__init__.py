"""Experiment Hub — the concrete ML Dashboard, ported from RankEvolve.

OpenTeam-agnostic library: takes an injected ``session_dir`` + an
``emit_event(session_id, event)`` callback (no FastAPI import here — AF
``server`` is transport-agnostic; the thin REST routers live in OpenTeam and
call these services). Populated across Phase 3 (backend) and consumed by the
``resources/tools/experiment_hub`` Dashboard tool.
"""
