"""Preflight tests for the task topology.

These tests are fast (< 60s each), do NOT spawn LLM subprocesses, and verify
configuration correctness BEFORE the multi-hour integration tests run. Use
them as a gate: if any preflight fails, the integration test is guaranteed
to be wasted compute.

Layout:
  preflight/
    test_yaml_smoke.py                 # YAML loads + topology instantiates
    test_workspace_final_deliverables.py  # outer workspace uses two-axis outputs/
    ...

Run all preflights:
  pytest test/openteam/resources/tools/task/preflight -v
"""
