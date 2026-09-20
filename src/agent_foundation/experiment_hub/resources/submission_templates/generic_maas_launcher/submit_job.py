# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Skeleton: MaaS one-shot launcher.

Wraps ``app-layer main fire-app -d <config>`` style launches. The agent
should:

1. Replace the stubbed ``run_fire_app`` with the real subprocess
   invocation that mirrors the user's reference command.
2. Treat ``--enable-flags enable_foo,enable_bar`` as **canonical config
   field names already resolved by the hub**. Build the overrides dict
   directly as ``{name: True for name in received_names}``. Do NOT define
   any HYPOTHESIS_FLAG_MAP / FLAG_MAP table or any ID-to-name translation
   logic — the names arrive pre-resolved.
3. Print ``FLOW_URI: <url>`` and ``MAST_JOB: <name>`` as bare ``print()``
   calls so the runner regex matches.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hub submission script (MaaS launcher)")
    p.add_argument(
        "--enable-flags",
        default="",
        help=(
            "Comma-separated CONFIG FIELD NAMES already resolved by the "
            "hub (e.g. 'enable_remove_detach,enable_universal_loss'). "
            "The script sets each to True on the model's main config."
        ),
    )
    p.add_argument("--experiment-name", required=True)
    return p.parse_args()


def build_config_overrides(enable_flags: list[str]) -> dict[str, Any]:
    """Build the model-config overrides dict from pre-resolved field names.

    The names in ``enable_flags`` are canonical config field names — the
    hub resolved them upstream (see SubmitExperimentConfirm.js). The
    script must NOT translate, look up, or otherwise re-derive them.
    """
    return {name: True for name in enable_flags}


def run_fire_app(experiment_name: str, overrides: dict[str, Any]) -> str:
    """REPLACE with real ``app-layer main fire-app -d <config>`` call.

    Should return the FBLearner flow URI (full URL).
    """
    raise NotImplementedError("Replace run_fire_app with the real fire-app launch.")


def main() -> int:
    args = parse_args()
    enable_flags = [f.strip() for f in args.enable_flags.split(",") if f.strip()]
    overrides = build_config_overrides(enable_flags)
    try:
        flow_uri = run_fire_app(args.experiment_name, overrides)
    except Exception as e:
        print(f"[error] launch failed: {e}", file=sys.stderr)
        return 1
    print(f"FLOW_URI: {flow_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
