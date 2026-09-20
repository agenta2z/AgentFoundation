#!/usr/bin/env fbpython
"""Scan a task tree and emit a correlation report.

For each `<root>/**/logs/session/<Class>-<hex8>.jsonl`, walks the file in order
and pairs InferenceInput → InferenceResponse. When the JSONL was produced under
``RESEARCH_PROPOSE__VERBOSE_CORRELATION=1`` the pair carries a matching
``call_<hex>`` filename segment (set via ``parts_file_namer`` in
``inferencer_base._call_correlation_kwargs``) so the pairing is filename-based
and unambiguous. For uninstrumented runs the script falls back to line-ordering
and reports ``call_id=null`` (still pairs correctly, just no defence against
interleaved guardrail-retries).

Also surfaces three v5-Phase-1 structured-event timelines per inferencer
instance:
  * RoleSwitch / RoleSwitchComplete  (multi_flow_dual_inferencer.py)
  * WorkspaceReassign                (inferencer_base.py _workspace.setter)
  * GuardrailRetry                   (inferencer_base.py _recovery_wrapper)
  * ToolUsePhase                     (streaming_inferencer_base.py)

Usage:
    fbpython analyze_correlation.py <task_root> [--out <csv_path>]

Output: `<task_root>/correlation_report.csv` (or the path passed via --out)
columns: ``instance_id, class, round, role, call_id, input_parts,
response_parts, retry_of, anomaly``.

Anomalies flagged:
  * CALL_ID_MISMATCH         input call_id != response call_id
  * INPUT_WITHOUT_RESPONSE   trailing input with no response (still-running
                             OR structural failure)
  * RESPONSE_WITHOUT_INPUT   leading response with no preceding input (should
                             never happen in normal flow)
  * AMBIGUOUS_ORDERING       uninstrumented run with >1 input or >1 response
                             between markers (cannot pair safely by ordering)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROUND_RE = re.compile(r"/round_(\d+)/")
# Recognise the role from the deepest matching segment; preserved as a list so
# panelist_NN / fix / aggregator can be disambiguated.
_ROLE_SEGMENTS = (
    "review",
    "fix",
    "aggregator",
    "breakdown",
    "guardrail",
    "panelist",
    "worker",
    "flow",
)
CALL_ID_RE = re.compile(r"call_([0-9a-f]{8})")


def _parse_role_from_path(path: str) -> str:
    """Return a slash-joined role hint extracted from the dir path.

    e.g. ``…/worker_00/children/round_02/children/review/children/panelist_00/…``
    → ``"worker_00/review/panelist_00"``.
    """
    parts = path.split(os.sep)
    keep = []
    for seg in parts:
        for kw in _ROLE_SEGMENTS:
            if seg == kw or seg.startswith(kw + "_"):
                keep.append(seg)
                break
    return "/".join(keep)


def _extract_call_id(parts_ref: Optional[Any]) -> Optional[str]:
    """Extract `call_<hex>` from a parts-file path. None when uninstrumented."""
    if not parts_ref:
        return None
    if isinstance(parts_ref, dict):
        parts_ref = parts_ref.get("__parts_file__")
    if not isinstance(parts_ref, str):
        return None
    m = CALL_ID_RE.search(parts_ref)
    return m.group(1) if m else None


def _input_parts_str(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("__parts_file__") or "")
    return ""


def _response_parts_str(item: Any) -> str:
    """Response is a dict whose values are each a {__parts_file__:…} ref;
    join the parts files into a single string for the CSV row."""
    if not isinstance(item, dict):
        return ""
    refs = []
    for key, value in item.items():
        if isinstance(value, dict) and "__parts_file__" in value:
            refs.append(f"{key}={value['__parts_file__']}")
    return ";".join(refs)


def _response_call_id(item: Any) -> Optional[str]:
    """Take the call_id from any of the response leaf parts files (they all share it)."""
    if not isinstance(item, dict):
        return None
    for value in item.values():
        if isinstance(value, dict):
            cid = _extract_call_id(value.get("__parts_file__"))
            if cid:
                return cid
    return None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _scan_jsonl(jsonl: Path) -> List[Dict[str, Any]]:
    """Return list of correlation rows for one .jsonl session file."""
    stem = jsonl.stem  # e.g. "ClaudeCodeCliInferencer-de7e1399"
    try:
        cls, hex8 = stem.rsplit("-", 1)
    except ValueError:
        cls, hex8 = stem, ""
    round_match = ROUND_RE.search(str(jsonl))
    rnd = round_match.group(1) if round_match else ""
    role = _parse_role_from_path(str(jsonl.parent.parent))  # strip /logs/session/

    rows: List[Dict[str, Any]] = []
    pending_input: Optional[Dict[str, Any]] = None
    pending_retry_of: Optional[str] = None

    lines = _read_jsonl(jsonl)
    for obj in lines:
        t = obj.get("type")
        item = obj.get("item")
        time_str = obj.get("time", "")
        if t == "GuardrailRetry":
            if isinstance(item, dict):
                pending_retry_of = item.get("parent_call_id")
        elif t == "InferenceInput":
            if pending_input is not None:
                # Two inputs in a row without a response — flag the first.
                rows.append(
                    {
                        "instance_id": hex8,
                        "class": cls,
                        "round": rnd,
                        "role": role,
                        "call_id": pending_input["call_id"],
                        "time_in": pending_input["time"],
                        "input_parts": pending_input["parts"],
                        "time_out": "",
                        "response_parts": "",
                        "retry_of": pending_input.get("retry_of", ""),
                        "anomaly": "INPUT_WITHOUT_RESPONSE",
                    }
                )
            pending_input = {
                "time": time_str,
                "call_id": _extract_call_id(_input_parts_str(item)),
                "parts": _input_parts_str(item),
                "retry_of": pending_retry_of or "",
            }
            pending_retry_of = None
        elif t == "InferenceResponse":
            if pending_input is None:
                rows.append(
                    {
                        "instance_id": hex8,
                        "class": cls,
                        "round": rnd,
                        "role": role,
                        "call_id": "",
                        "time_in": "",
                        "input_parts": "",
                        "time_out": time_str,
                        "response_parts": _response_parts_str(item),
                        "retry_of": "",
                        "anomaly": "RESPONSE_WITHOUT_INPUT",
                    }
                )
                continue
            in_cid = pending_input["call_id"]
            out_cid = _response_call_id(item)
            anomaly = ""
            if in_cid and out_cid and in_cid != out_cid:
                anomaly = "CALL_ID_MISMATCH"
            rows.append(
                {
                    "instance_id": hex8,
                    "class": cls,
                    "round": rnd,
                    "role": role,
                    "call_id": in_cid or out_cid or "",
                    "time_in": pending_input["time"],
                    "input_parts": pending_input["parts"],
                    "time_out": time_str,
                    "response_parts": _response_parts_str(item),
                    "retry_of": pending_input.get("retry_of", ""),
                    "anomaly": anomaly,
                }
            )
            pending_input = None

    if pending_input is not None:
        rows.append(
            {
                "instance_id": hex8,
                "class": cls,
                "round": rnd,
                "role": role,
                "call_id": pending_input["call_id"],
                "time_in": pending_input["time"],
                "input_parts": pending_input["parts"],
                "time_out": "",
                "response_parts": "",
                "retry_of": pending_input.get("retry_of", ""),
                "anomaly": "INPUT_WITHOUT_RESPONSE",
            }
        )
    return rows


def _scan_events(root: Path) -> Dict[str, List[Tuple[str, str, Any]]]:
    """Collect RoleSwitch / WorkspaceReassign / ToolUsePhase timelines.

    Returns {instance_id: [(time, type, item), …]}.
    """
    out: Dict[str, List[Tuple[str, str, Any]]] = {}
    for jsonl in root.rglob("logs/session/*.jsonl"):
        stem = jsonl.stem
        try:
            _cls, hex8 = stem.rsplit("-", 1)
        except ValueError:
            continue
        for obj in _read_jsonl(jsonl):
            t = obj.get("type")
            if t in (
                "RoleSwitch",
                "RoleSwitchComplete",
                "WorkspaceReassign",
                "ToolUsePhase",
            ):
                out.setdefault(hex8, []).append(
                    (obj.get("time", ""), t, obj.get("item"))
                )
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "task_root", help="Path to a task root (the dir containing children/ and logs/)"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Defaults to <task_root>/correlation_report.csv",
    )
    parser.add_argument(
        "--events-out",
        default=None,
        help="Optional JSON path to dump RoleSwitch/WorkspaceReassign/ToolUsePhase timelines. "
        "Defaults to <task_root>/correlation_events.json",
    )
    args = parser.parse_args(argv)

    root = Path(args.task_root).resolve()
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 2
    out_csv = Path(args.out) if args.out else root / "correlation_report.csv"
    out_json = (
        Path(args.events_out) if args.events_out else root / "correlation_events.json"
    )

    rows: List[Dict[str, Any]] = []
    jsonl_count = 0
    for jsonl in sorted(root.rglob("logs/session/*.jsonl")):
        jsonl_count += 1
        rows.extend(_scan_jsonl(jsonl))

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "instance_id",
                "class",
                "round",
                "role",
                "call_id",
                "time_in",
                "input_parts",
                "time_out",
                "response_parts",
                "retry_of",
                "anomaly",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    events = _scan_events(root)
    with out_json.open("w", encoding="utf-8") as f:
        # JSON output: keyed by instance_id → list of {time,type,item}
        f.write(
            json.dumps(
                {
                    k: [{"time": t, "type": ty, "item": it} for (t, ty, it) in v]
                    for k, v in events.items()
                },
                indent=2,
                default=str,
            )
        )

    n = len(rows)
    n_with_callid = sum(1 for r in rows if r["call_id"])
    n_anomalies = sum(1 for r in rows if r["anomaly"])
    n_retries = sum(1 for r in rows if r["retry_of"])

    print(f"Scanned {jsonl_count} session.jsonl file(s)")
    print(f"Total invocations:  {n}")
    print(f"With call_id:       {n_with_callid}  ({n_with_callid * 100 // max(n, 1)}%)")
    print(f"Marked as retry:    {n_retries}")
    print(f"Anomalies:          {n_anomalies}")
    if n_anomalies:
        # Group anomaly counts by kind
        from collections import Counter

        kinds = Counter(r["anomaly"] for r in rows if r["anomaly"])
        for kind, count in sorted(kinds.items()):
            print(f"  {kind}: {count}")
    print()
    print(f"Wrote correlation report: {out_csv}")
    print(f"Wrote events timeline:    {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
