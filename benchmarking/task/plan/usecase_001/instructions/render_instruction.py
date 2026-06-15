#!/usr/bin/env python3
"""
render_instruction.py — render the judge-instruction template into a verbatim,
ready-to-paste prompt by filling in placeholders from a usecase's plan_sources.yaml.

Generic across usecases. Refuses to overwrite an existing output file unless --force
is passed (to protect the historical 3way/4way artifacts).

Usage:
    python3 render_instruction.py [--usecase-dir <dir>] [--out <path>] [--force]
                                  [--axes "axis1,axis2,..."] [--paths-style {benchmark|original}]

Defaults:
    --usecase-dir : parent of this script
    --out         : instructions/judge_instruction.rendered.md
    --axes        : "plan depth, comprehensiveness, correctness, elegance (design quality)"
    --paths-style : "benchmark"  (use plans/ copies; choose "original" for original paths)

Examples:
    # Render the current N-plan instruction (auto-detects N from plan_sources.yaml)
    python3 render_instruction.py

    # Render with original on-disk plan paths (less reproducible; matches the 3-way pattern)
    python3 render_instruction.py --paths-style original --out instructions/_scratch.md

    # Custom rubric axes
    python3 render_instruction.py --axes "depth, rigor, novelty, elegance"
"""

from __future__ import annotations

import argparse
import datetime as _dt
import pathlib
import sys

try:
    import yaml
except ImportError:
    sys.exit("ERROR: PyYAML is required. pip install pyyaml")


WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
        7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Render judge-instruction template.")
    ap.add_argument("--usecase-dir", type=pathlib.Path, default=None,
                    help="Path to the usecase directory (default: parent of this script).")
    ap.add_argument("--template", type=pathlib.Path, default=None,
                    help="Path to template (default: <usecase>/instructions/judge_instruction.template.md).")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="Output path (default: <usecase>/instructions/judge_instruction.rendered.md).")
    ap.add_argument("--axes", type=str,
                    default="plan depth, comprehensiveness, correctness, elegance (design quality)",
                    help="Comma-separated top-weighted axes string.")
    ap.add_argument("--paths-style", choices=("benchmark", "original"), default="benchmark",
                    help="Which path to list for each plan (default: benchmark copies in plans/).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite output file if it exists (default: refuse, to protect historical artifacts).")
    args = ap.parse_args()

    usecase = (args.usecase_dir or pathlib.Path(__file__).resolve().parent.parent).resolve()
    template_path = (args.template or (usecase / "instructions" / "judge_instruction.template.md")).resolve()
    out_path = (args.out or (usecase / "instructions" / "judge_instruction.rendered.md")).resolve()
    plan_sources_path = (usecase / "plan_sources.yaml").resolve()

    for p in (template_path, plan_sources_path):
        if not p.is_file():
            sys.exit(f"ERROR: required file not found: {p}")

    if out_path.exists() and not args.force:
        sys.exit(f"ERROR: refusing to overwrite existing {out_path}; pass --force or choose a different --out path.")

    # Load plan list
    ps = yaml.safe_load(plan_sources_path.read_text())
    plans = ps.get("plans", [])
    if not plans:
        sys.exit("ERROR: no plans found in plan_sources.yaml::plans")

    n = len(plans)
    if n not in WORD:
        sys.exit(f"ERROR: N={n} plans is outside the supported English-word range (1-10); extend WORD dict.")

    # Build the bullet list
    bullets = []
    for p in plans:
        if args.paths_style == "benchmark":
            path = p["benchmark_copy"]
            prefix = f"  - {usecase.name}/{path}"
        else:
            path = p["original_path"]
            prefix = f"  {path}"
        bullets.append(prefix)
    plan_list_bullets = "\n".join(bullets)

    # Render
    tmpl = template_path.read_text()
    rendered = (tmpl
                .replace("{{N_PLANS}}", str(n))
                .replace("{{N_PLANS_WORD}}", WORD[n])
                .replace("{{PLAN_LIST_BULLETS}}", plan_list_bullets)
                .replace("{{TOP_WEIGHTED_AXES_CSV}}", args.axes)
                .replace("{{USECASE_ID}}", usecase.name)
                .replace("{{RENDERED_AT}}", _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")))

    out_path.write_text(rendered)
    print(f"OK — wrote {out_path}")
    print(f"     plans:        {n} ({WORD[n]})")
    print(f"     paths-style:  {args.paths_style}")
    print(f"     axes:         {args.axes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
