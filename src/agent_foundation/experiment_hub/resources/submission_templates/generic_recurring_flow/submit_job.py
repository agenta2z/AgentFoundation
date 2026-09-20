# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Skeleton: recurring-flow submission entry point.

This is a TEMPLATE — the agent reads it for shape only. Real fbcode
imports are stubbed out with ``# fbcode: ...`` comments so this file
parses outside the buck environment. The agent should:

1. Replace the stubs with the user's team's actual recurring-flow
   dispatcher (commonly ``cfr_main_mtml.recurring`` or similar).
2. Treat ``--enable-flags enable_foo,enable_bar`` as **canonical config
   field names already resolved by the hub**. Build the overrides dict
   directly as ``{name: True for name in received_names}``. Do NOT define
   any HYPOTHESIS_FLAG_MAP / FLAG_MAP table or any ID-to-name translation
   logic — the names arrive pre-resolved.
3. Print ``FLOW_URI: <url>`` and ``MAST_JOB: <name>`` as bare ``print()``
   calls (NOT logger.info — the runner regex anchors to ^FLOW_URI:).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hub submission script (recurring flow)")
    p.add_argument(
        "--enable-flags",
        default="",
        help=(
            "Comma-separated CONFIG FIELD NAMES already resolved by the "
            "hub (e.g. 'enable_remove_detach,enable_universal_loss'). "
            "The script sets each to True on the model's main config."
        ),
    )
    p.add_argument(
        "--experiment-name",
        required=True,
        help="Human-readable experiment name; substituted into FBLearner job name.",
    )
    return p.parse_args()


def build_config_overrides(enable_flags: list[str]) -> dict[str, Any]:
    """Build the model-config overrides dict from pre-resolved field names.

    The names in ``enable_flags`` are canonical config field names — the
    hub resolved them upstream (see SubmitExperimentConfirm.js). The
    script must NOT translate, look up, or otherwise re-derive them.
    """
    return {name: True for name in enable_flags}


def schedule_recurring_flow(experiment_name: str, overrides: dict[str, Any]) -> str:
    """REPLACE with the real recurring-flow dispatcher call.

    Should return the FBLearner flow URI (full URL) so the runner can
    capture it via ``print(f"FLOW_URI: {url}")`` below.
    """
    # fbcode: from cfr_main_mtml.recurring import RecurringFlowScheduler
    # fbcode: scheduler = RecurringFlowScheduler(experiment_name=experiment_name)
    # fbcode: scheduler.apply_overrides(overrides)
    # fbcode: flow = scheduler.dispatch()
    # fbcode: return flow.uri
    raise NotImplementedError(
        "Replace schedule_recurring_flow with the real dispatcher."
    )


def main() -> int:
    args = parse_args()
    enable_flags = [f.strip() for f in args.enable_flags.split(",") if f.strip()]
    overrides = build_config_overrides(enable_flags)
    try:
        flow_uri = schedule_recurring_flow(args.experiment_name, overrides)
    except Exception as e:
        print(f"[error] dispatch failed: {e}", file=sys.stderr)
        return 1
    # Contract lines — bare print, single token after the ':' so the
    # runner regex matches.
    print(f"FLOW_URI: {flow_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
