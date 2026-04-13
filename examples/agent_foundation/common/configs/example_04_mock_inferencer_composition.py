#!/usr/bin/env python3
"""Example 04: Compose mock inferencers via YAML — the real-world use case.

This is what the system was built for: defining inferencer pipelines in YAML
instead of writing Python wiring code.  Uses @attrs mock inferencers that
simulate LLM behavior (with canned responses) to show the composition effect.

Demonstrates:
- Registering real @attrs inferencers
- YAML config that composes a DualInferencer-like pattern
- String shorthand for simple components
- How attrs underscore-stripping works in YAML
- How init=False fields are auto-filtered

Expected output:
================

=== 1. Simple mock inferencer from YAML ===
  mock_claude.yaml:
    _target_: MockLLM
    model_name: claude-test
    response_prefix: "Claude says: "

  obj = MockLLM(model_name='claude-test')
  Query: 'What is Python?'
  Response: 'Claude says: What is Python?'

=== 2. Compose: reviewer wraps a base inferencer ===
  reviewer.yaml:
    _target_: ReviewerInferencer
    base: MockLLM
    review_prompt: "Review this for accuracy"

  Query: 'The sky is green'
  Step 1 — base generates:  'I agree: The sky is green'
  Step 2 — reviewer checks: 'REVIEW of [I agree: The sky is green]: Review this for accuracy'
  Final: 'REVIEW of [I agree: The sky is green]: Review this for accuracy'

=== 3. Full pipeline: translate -> generate -> review ===
  full_pipeline.yaml:
    _target_: ChainInferencer
    steps:
      - _target_: MockLLM
        model_name: translator
        response_prefix: "[Translated] "
      - _target_: MockLLM
        model_name: generator
        response_prefix: "[Generated] "
      - _target_: MockLLM
        model_name: reviewer
        response_prefix: "[Reviewed] "

  Query: 'Explain quantum computing'
    -> Step 1 (translator): [Translated] Explain quantum computing
    -> Step 2 (generator):  [Generated] [Translated] Explain quantum computing
    -> Step 3 (reviewer):   [Reviewed] [Generated] [Translated] Explain quantum computing

=== 4. attrs features: underscore-stripping + init=False filtering ===
  Config with _secret_key -> YAML key is 'secret_key' (underscore stripped)
  Config with init=False field -> auto-filtered with warning
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, List, Optional

_script_dir = os.path.dirname(os.path.abspath(__file__))
_core_root = os.path.normpath(os.path.join(_script_dir, "..", "..", "..", "..", ".."))
for _sub in ("AgentFoundation/src", "RichPythonUtils/src"):
    _p = os.path.normpath(os.path.join(_core_root, _sub))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from attr import attrib, attrs

from rich_python_utils.config_utils import (
    instantiate,
    load_config,
    register,
)
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Mock inferencer classes (simulate LLM behavior with canned responses)
# ---------------------------------------------------------------------------

@register("MockLLM", category="inferencer")
@attrs
class MockLLM:
    """Simulates an LLM: echoes the input with a configurable prefix.

    Demonstrates:
    - @attrs class with attrib() fields
    - _secret_key with underscore-stripping (init param = 'secret_key')
    - _call_count with init=False (internal state, filtered from YAML)
    """
    model_name: str = attrib(default="mock-model")
    response_prefix: str = attrib(default="")
    _secret_key: str = attrib(default="demo-key-123")
    _call_count: int = attrib(default=0, init=False)

    def infer(self, prompt: str) -> str:
        self._call_count += 1
        return f"{self.response_prefix}{prompt}"

    def __repr__(self):
        return f"MockLLM(model_name={self.model_name!r})"


@register("ReviewerInferencer", category="inferencer")
@attrs
class ReviewerInferencer:
    """Wraps a base inferencer and adds a review step.

    Demonstrates nested composition: this inferencer CONTAINS another inferencer.
    """
    base: Any = attrib(default=None)
    review_prompt: str = attrib(default="Please review")

    def infer(self, prompt: str) -> str:
        # Step 1: let the base inferencer generate
        base_output = self.base.infer(prompt) if self.base else prompt
        print(f"  Step 1 — base generates:  {base_output!r}")

        # Step 2: review the output
        review = f"REVIEW of [{base_output}]: {self.review_prompt}"
        print(f"  Step 2 — reviewer checks: {review!r}")
        return review


@register("ChainInferencer", category="inferencer")
@attrs
class ChainInferencer:
    """Chains multiple inferencers in sequence — output of one feeds the next.

    Demonstrates list-based composition from YAML.
    """
    steps: list = attrib(factory=list)

    def infer(self, prompt: str) -> str:
        result = prompt
        for i, step in enumerate(self.steps, 1):
            result = step.infer(result)
            name = getattr(step, "model_name", type(step).__name__)
            print(f"    -> Step {i} ({name}): {result}")
        return result


def separator(title):
    print(f"\n{'=' * 3} {title} {'=' * 3}")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ── 1. Simple mock inferencer ────────────────────────
        separator("1. Simple mock inferencer from YAML")

        yaml_text = """\
_target_: MockLLM
model_name: claude-test
response_prefix: "Claude says: "
"""
        yaml_path = tmpdir / "mock_claude.yaml"
        yaml_path.write_text(yaml_text)

        print(f"  mock_claude.yaml:")
        for line in yaml_text.strip().splitlines():
            print(f"    {line}")
        print()

        cfg = load_config(str(yaml_path))
        llm = instantiate(cfg)
        print(f"  obj = {llm}")

        query = "What is Python?"
        print(f"  Query: {query!r}")
        print(f"  Response: {llm.infer(query)!r}")

        # ── 2. Nested composition ────────────────────────────
        separator("2. Compose: reviewer wraps a base inferencer")

        yaml_text = """\
_target_: ReviewerInferencer
base:
  _target_: MockLLM
  model_name: base-model
  response_prefix: "I agree: "
review_prompt: "Review this for accuracy"
"""
        yaml_path = tmpdir / "reviewer.yaml"
        yaml_path.write_text(yaml_text)

        print(f"  reviewer.yaml:")
        for line in yaml_text.strip().splitlines():
            print(f"    {line}")
        print()

        cfg = load_config(str(yaml_path))
        reviewer = instantiate(cfg)

        query = "The sky is green"
        print(f"  Query: {query!r}")
        result = reviewer.infer(query)
        print(f"  Final: {result!r}")

        # ── 3. Full chain pipeline ───────────────────────────
        separator("3. Full pipeline: translate -> generate -> review")

        yaml_text = """\
_target_: ChainInferencer
steps:
  - _target_: MockLLM
    model_name: translator
    response_prefix: "[Translated] "
  - _target_: MockLLM
    model_name: generator
    response_prefix: "[Generated] "
  - _target_: MockLLM
    model_name: reviewer
    response_prefix: "[Reviewed] "
"""
        yaml_path = tmpdir / "full_pipeline.yaml"
        yaml_path.write_text(yaml_text)

        print(f"  full_pipeline.yaml:")
        for line in yaml_text.strip().splitlines():
            print(f"    {line}")
        print()

        cfg = load_config(str(yaml_path))
        chain = instantiate(cfg)

        query = "Explain quantum computing"
        print(f"  Query: {query!r}")
        result = chain.infer(query)

        # ── 4. attrs features demo ───────────────────────────
        separator("4. attrs features: underscore-stripping + init=False filtering")

        # Underscore stripping: _secret_key -> YAML key 'secret_key'
        yaml_text = """\
_target_: MockLLM
model_name: secret-model
secret_key: my-api-key-xyz
"""
        yaml_path = tmpdir / "with_secret.yaml"
        yaml_path.write_text(yaml_text)

        cfg = load_config(str(yaml_path))
        llm = instantiate(cfg)
        print(f"  Config with secret_key: 'my-api-key-xyz'")
        print(f"  obj._secret_key = {llm._secret_key!r}  (underscore auto-stripped in YAML)")

        # init=False filtering: _call_count would be rejected
        print()
        import logging
        logging.basicConfig(level=logging.WARNING, format="  %(levelname)s: %(message)s")

        cfg = OmegaConf.create({
            "_target_": f"{__name__}.MockLLM",
            "model_name": "test",
            "_call_count": 999,  # init=False field — will be filtered with warning
        })
        llm = instantiate(cfg)
        print(f"  obj._call_count = {llm._call_count}  (init=False field was filtered, default 0 used)")

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Key Takeaways:")
    print("  1. _target_ + kwargs in YAML -> real Python objects")
    print("  2. Nested _target_ blocks -> composed object trees")
    print("  3. String shorthand: 'MockLLM' -> auto-expands to full object")
    print("  4. attrs underscore-stripping: _secret_key -> YAML key 'secret_key'")
    print("  5. init=False fields auto-filtered with a warning")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
