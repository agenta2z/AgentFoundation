#!/usr/bin/env python3
"""Example 02: Instantiate Python objects from dict configs (no YAML files needed).

This shows the core concept: a dict with ``_target_`` + kwargs -> a real Python
object.  Progressively builds from simple to nested to shorthand.

Expected output:
================

=== 1. Simple: one object from a dict ===
  Config: {'_target_': 'Translator', 'lang': 'French'}
  Result: Translator(lang='French')
  Translate 'hello' -> '[French] hello'

=== 2. Nested: parent object containing a child object ===
  Config: {'_target_': 'Pipeline', 'steps': [{'_target_': 'Translator', 'lang': 'Spanish'}, ...]}
  Result: Pipeline with 2 steps
  Run pipeline on 'good morning':
    Step 1 (Translator): [Spanish] good morning
    Step 2 (Uppercaser): GOOD MORNING
  Final: GOOD MORNING

=== 3. String shorthand: 'Uppercaser' expands automatically ===
  Config: {'_target_': 'Pipeline', 'first': 'Translator', 'second': 'Uppercaser'}
  Result: Pipeline with first=Translator(lang=''), second=Uppercaser()
    -> Both were instantiated from shorthand strings!

=== 4. Overrides: same config, different parameters ===
  Default:  Translator(lang='')
  Override: Translator(lang='Japanese')
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_core_root = os.path.normpath(os.path.join(_script_dir, "..", "..", "..", "..", ".."))
for _sub in ("AgentFoundation/src", "RichPythonUtils/src"):
    _p = os.path.normpath(os.path.join(_core_root, _sub))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from omegaconf import OmegaConf

from rich_python_utils.config_utils import (
    _reset_registry,
    instantiate,
    register,
    register_class,
)


# ---------------------------------------------------------------------------
# Simple mock classes (imagine these are real inferencers)
# ---------------------------------------------------------------------------

@register("Translator")
class Translator:
    """Simulates translation by prepending [lang]."""
    def __init__(self, lang: str = ""):
        self.lang = lang

    def __repr__(self):
        return f"Translator(lang={self.lang!r})"

    def process(self, text: str) -> str:
        return f"[{self.lang}] {text}" if self.lang else text


@register("Uppercaser")
class Uppercaser:
    """Converts text to uppercase."""
    def __repr__(self):
        return "Uppercaser()"

    def process(self, text: str) -> str:
        return text.upper()


@register("Pipeline")
class Pipeline:
    """Chains multiple processors together."""
    def __init__(self, steps=None, first=None, second=None):
        # Accept either a list of steps or named first/second
        if steps:
            self.steps = steps
        else:
            self.steps = [s for s in [first, second] if s is not None]

    def run(self, text: str) -> str:
        result = text
        for i, step in enumerate(self.steps, 1):
            result = step.process(result)
            print(f"    Step {i} ({type(step).__name__}): {result}")
        return result


def separator(title: str):
    print(f"\n{'=' * 3} {title} {'=' * 3}")


def main():
    # ── 1. Simple instantiation ──────────────────────────────
    separator("1. Simple: one object from a dict")

    config = {"_target_": "Translator", "lang": "French"}
    print(f"  Config: {config}")

    obj = instantiate(OmegaConf.create(config))
    print(f"  Result: {obj}")
    print(f"  Translate 'hello' -> {obj.process('hello')!r}")

    # ── 2. Nested instantiation ──────────────────────────────
    separator("2. Nested: parent object containing child objects")

    config = {
        "_target_": "Pipeline",
        "steps": [
            {"_target_": "Translator", "lang": "Spanish"},
            {"_target_": "Uppercaser"},
        ],
    }
    print(f"  Config: {config}")

    pipeline = instantiate(OmegaConf.create(config))
    print(f"  Result: Pipeline with {len(pipeline.steps)} steps")
    print(f"  Run pipeline on 'good morning':")
    final = pipeline.run("good morning")
    print(f"  Final: {final}")

    # ── 3. String shorthand ──────────────────────────────────
    separator("3. String shorthand: 'Uppercaser' expands automatically")

    config = {
        "_target_": "Pipeline",
        "first": "Translator",     # shorthand — expands to {_target_: Translator}
        "second": "Uppercaser",    # shorthand — same
    }
    print(f"  Config: {config}")

    pipeline = instantiate(OmegaConf.create(config))
    print(f"  Result: Pipeline with first={pipeline.steps[0]}, second={pipeline.steps[1]}")
    print(f"    -> Both were instantiated from shorthand strings!")

    # ── 4. Overrides ─────────────────────────────────────────
    separator("4. Overrides: same config, different parameters")

    base_config = OmegaConf.create({"_target_": "Translator"})

    default_obj = instantiate(base_config)
    print(f"  Default:  {default_obj}")

    override_obj = instantiate(
        OmegaConf.merge(base_config, {"lang": "Japanese"})
    )
    print(f"  Override: {override_obj}")


if __name__ == "__main__":
    main()
