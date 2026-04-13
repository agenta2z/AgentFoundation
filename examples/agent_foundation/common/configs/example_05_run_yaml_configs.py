#!/usr/bin/env python3
"""Example 05: Load and run every YAML config in the yaml_configs/ folder.

This is the "gallery" example: each .yaml file in yaml_configs/ defines an
inferencer composition.  This script loads each one, instantiates the object
tree, and runs a test query to show the composition in action.

Run it:
    python example_05_run_yaml_configs.py

Or run a specific config:
    python example_05_run_yaml_configs.py 03_chain_pipeline

Expected output:
================

================================================================
  YAML Config Gallery — load a file, get a working object
================================================================

--- yaml_configs/01_simple.yaml ---
  _target_: MockLLM
  model_name: claude-haiku
  response_prefix: "[Haiku] "

  Created: MockLLM(model_name='claude-haiku')
  Query:   'Explain recursion in one sentence'
  Output:  '[Haiku] Explain recursion in one sentence'

--- yaml_configs/02_nested.yaml ---
  ...ReviewerInferencer wrapping a MockLLM...
  Created: ReviewerInferencer with base=MockLLM(model_name='sonnet')
  Query:   'Explain recursion in one sentence'
    Step 1 -- base generates:  'Draft: Explain recursion in one sentence'
    Step 2 -- reviewer checks: 'REVIEW of [Draft: ...]: Check for factual errors'
  Output:  'REVIEW of [Draft: Explain recursion in one sentence]: Check for factual errors'

  ...and so on for each YAML file...
"""

import os
import sys
import warnings
from pathlib import Path

_script_dir = os.path.dirname(os.path.abspath(__file__))
_core_root = os.path.normpath(os.path.join(_script_dir, "..", "..", "..", "..", ".."))
for _sub in ("AgentFoundation/src", "RichPythonUtils/src"):
    _p = os.path.normpath(os.path.join(_core_root, _sub))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

warnings.filterwarnings("ignore")

from typing import Any
from attr import attrib, attrs
from rich_python_utils.config_utils import instantiate, load_config, register


# ---------------------------------------------------------------------------
# Mock inferencer classes (same ones used in example_04)
# ---------------------------------------------------------------------------

@register("MockLLM", category="inferencer")
@attrs
class MockLLM:
    """Simulates an LLM: echoes input with a prefix."""
    model_name: str = attrib(default="mock-model")
    response_prefix: str = attrib(default="")
    _secret_key: str = attrib(default="demo-key")
    _call_count: int = attrib(default=0, init=False)

    def infer(self, prompt: str) -> str:
        self._call_count += 1
        return f"{self.response_prefix}{prompt}"

    def __repr__(self):
        return f"MockLLM(model_name={self.model_name!r})"


@register("ReviewerInferencer", category="inferencer")
@attrs
class ReviewerInferencer:
    """Wraps a base inferencer and adds a review step."""
    base: Any = attrib(default=None)
    review_prompt: str = attrib(default="Please review")

    def infer(self, prompt: str) -> str:
        base_output = self.base.infer(prompt) if self.base else prompt
        print(f"    Step 1 -- base generates:  {base_output!r}")
        review = f"REVIEW of [{base_output}]: {self.review_prompt}"
        print(f"    Step 2 -- reviewer checks: {review!r}")
        return review

    def __repr__(self):
        return f"ReviewerInferencer(base={self.base!r})"


@register("ChainInferencer", category="inferencer")
@attrs
class ChainInferencer:
    """Chains inferencers in sequence: output of one feeds the next."""
    steps: list = attrib(factory=list)

    def infer(self, prompt: str) -> str:
        result = prompt
        for i, step in enumerate(self.steps, 1):
            result = step.infer(result)
            name = getattr(step, "model_name", type(step).__name__)
            print(f"    Step {i} ({name}): {result}")
        return result

    def __repr__(self):
        return f"ChainInferencer(steps={self.steps!r})"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def describe_object(obj, indent=2):
    """Print a human-readable description of the instantiated object."""
    prefix = " " * indent
    cls_name = type(obj).__name__

    if cls_name == "MockLLM":
        print(f"{prefix}Created: {obj}")
    elif cls_name == "ReviewerInferencer":
        print(f"{prefix}Created: ReviewerInferencer with base={obj.base!r}")
    elif cls_name == "ChainInferencer":
        step_names = []
        for s in obj.steps:
            if hasattr(s, "model_name"):
                step_names.append(s.model_name)
            else:
                step_names.append(type(s).__name__)
        print(f"{prefix}Created: ChainInferencer with {len(obj.steps)} steps: {step_names}")
    else:
        print(f"{prefix}Created: {cls_name}")


def run_one(yaml_path: Path, query: str):
    """Load a YAML config, instantiate, describe, and run a query."""
    rel = yaml_path.relative_to(yaml_path.parent.parent)
    print(f"\n--- {rel} ---")

    # Show the YAML contents
    for line in yaml_path.read_text().strip().splitlines():
        if not line.strip().startswith("#"):
            print(f"  {line}")

    print()

    # Load and instantiate
    cfg = load_config(str(yaml_path))
    obj = instantiate(cfg)
    describe_object(obj)

    # Run a test query
    print(f"  Query:   {query!r}")
    result = obj.infer(query)
    print(f"  Output:  {result!r}")


def main():
    yaml_dir = Path(_script_dir) / "yaml_configs"

    # Optional: run a specific config by name
    filter_name = sys.argv[1] if len(sys.argv) > 1 else None

    yamls = sorted(yaml_dir.glob("*.yaml"))
    if filter_name:
        yamls = [y for y in yamls if filter_name in y.stem]
        if not yamls:
            print(f"No YAML config matching '{filter_name}'. Available:")
            for y in sorted(yaml_dir.glob("*.yaml")):
                print(f"  {y.stem}")
            return

    print("=" * 64)
    print("  YAML Config Gallery -- load a file, get a working object")
    print("=" * 64)

    query = "Explain recursion in one sentence"

    for yaml_path in yamls:
        run_one(yaml_path, query)

    print(f"\n{'=' * 64}")
    print(f"  Ran {len(yamls)} YAML configs. Each produced a real, callable object.")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
