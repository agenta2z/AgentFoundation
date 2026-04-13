#!/usr/bin/env python3
"""Example 03: Load and instantiate objects from YAML config files.

This is the main use case: define object compositions in YAML, then
instantiate them with one function call.  Shows:
- Loading a YAML file with aliases
- Nested object composition (inferencer containing another inferencer)
- String shorthand in YAML
- Overrides at load time

The example creates temporary YAML files to be self-contained.

Expected output:
================

=== 1. Simple YAML -> object ===
  translator.yaml:
    _target_: Translator
    lang: German

  Loaded config: {'_target_': 'Translator', 'lang': 'German'}
  Instantiated: Translator(lang='German')
  Test: 'hello' -> '[German] hello'

=== 2. Nested YAML -> composed objects ===
  pipeline.yaml:
    _target_: Pipeline
    steps:
      - _target_: Translator
        lang: Italian
      - _target_: Uppercaser

  Instantiated: Pipeline with 2 steps
  Run 'buongiorno':
    Step 1 (Translator): [Italian] buongiorno
    Step 2 (Uppercaser): [ITALIAN] BUONGIORNO
  Final: [ITALIAN] BUONGIORNO

=== 3. String shorthand in YAML ===
  shorthand_pipeline.yaml:
    _target_: Pipeline
    first: Translator        # ← shorthand!
    second: Uppercaser       # ← shorthand!

  Both 'Translator' and 'Uppercaser' were auto-expanded to real objects.
  Pipeline steps: [Translator(lang=''), Uppercaser()]

=== 4. Override at load time ===
  Base YAML has lang: German
  Override with lang: Korean at load time
  Result: Translator(lang='Korean')

=== 5. ${path:...} resolver ===
  data_config.yaml:
    data_dir: ${path:data/input}

  Resolved relative to YAML file location:
  data_dir = /absolute/path/to/tmpdir/data/input
"""

import os
import sys
import tempfile
from pathlib import Path

_script_dir = os.path.dirname(os.path.abspath(__file__))
_core_root = os.path.normpath(os.path.join(_script_dir, "..", "..", "..", "..", ".."))
for _sub in ("AgentFoundation/src", "RichPythonUtils/src"):
    _p = os.path.normpath(os.path.join(_core_root, _sub))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from rich_python_utils.config_utils import (
    _reset_registry,
    instantiate,
    load_config,
    register,
)


# ---------------------------------------------------------------------------
# Same mock classes as Example 02 (re-registered here for standalone use)
# ---------------------------------------------------------------------------

@register("Translator")
class Translator:
    def __init__(self, lang=""):
        self.lang = lang

    def __repr__(self):
        return f"Translator(lang={self.lang!r})"

    def process(self, text):
        return f"[{self.lang}] {text}" if self.lang else text


@register("Uppercaser")
class Uppercaser:
    def __repr__(self):
        return "Uppercaser()"

    def process(self, text):
        return text.upper()


@register("Pipeline")
class Pipeline:
    def __init__(self, steps=None, first=None, second=None):
        if steps:
            self.steps = steps
        else:
            self.steps = [s for s in [first, second] if s is not None]

    def run(self, text):
        result = text
        for i, step in enumerate(self.steps, 1):
            result = step.process(result)
            print(f"    Step {i} ({type(step).__name__}): {result}")
        return result


def separator(title):
    print(f"\n{'=' * 3} {title} {'=' * 3}")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ── 1. Simple YAML ───────────────────────────────────
        separator("1. Simple YAML -> object")

        yaml_text = "_target_: Translator\nlang: German\n"
        yaml_path = tmpdir / "translator.yaml"
        yaml_path.write_text(yaml_text)

        print(f"  translator.yaml:")
        for line in yaml_text.strip().splitlines():
            print(f"    {line}")
        print()

        cfg = load_config(str(yaml_path))
        print(f"  Loaded config: {dict(cfg)}")

        obj = instantiate(cfg)
        print(f"  Instantiated: {obj}")
        print(f"  Test: 'hello' -> {obj.process('hello')!r}")

        # ── 2. Nested YAML ───────────────────────────────────
        separator("2. Nested YAML -> composed objects")

        yaml_text = """\
_target_: Pipeline
steps:
  - _target_: Translator
    lang: Italian
  - _target_: Uppercaser
"""
        yaml_path = tmpdir / "pipeline.yaml"
        yaml_path.write_text(yaml_text)

        print(f"  pipeline.yaml:")
        for line in yaml_text.strip().splitlines():
            print(f"    {line}")
        print()

        cfg = load_config(str(yaml_path))
        pipeline = instantiate(cfg)
        print(f"  Instantiated: Pipeline with {len(pipeline.steps)} steps")
        print(f"  Run 'buongiorno':")
        final = pipeline.run("buongiorno")
        print(f"  Final: {final}")

        # ── 3. String shorthand ──────────────────────────────
        separator("3. String shorthand in YAML")

        yaml_text = """\
_target_: Pipeline
first: Translator
second: Uppercaser
"""
        yaml_path = tmpdir / "shorthand_pipeline.yaml"
        yaml_path.write_text(yaml_text)

        print(f"  shorthand_pipeline.yaml:")
        for line in yaml_text.strip().splitlines():
            print(f"    {line}")
        print()

        cfg = load_config(str(yaml_path))
        pipeline = instantiate(cfg)
        print(f"  Both 'Translator' and 'Uppercaser' were auto-expanded to real objects.")
        print(f"  Pipeline steps: {pipeline.steps}")

        # ── 4. Override at load time ─────────────────────────
        separator("4. Override at load time")

        yaml_path = tmpdir / "translator.yaml"  # reuse from step 1
        print(f"  Base YAML has lang: German")

        cfg = load_config(str(yaml_path), overrides={"lang": "Korean"})
        obj = instantiate(cfg)
        print(f"  Override with lang: Korean at load time")
        print(f"  Result: {obj}")

        # ── 5. ${path:...} resolver ──────────────────────────
        separator("5. ${path:...} resolver")

        yaml_text = "data_dir: ${path:data/input}\n"
        yaml_path = tmpdir / "data_config.yaml"
        yaml_path.write_text(yaml_text)

        print(f"  data_config.yaml:")
        print(f"    data_dir: ${{path:data/input}}")
        print()

        cfg = load_config(str(yaml_path))
        expected = str((tmpdir / "data" / "input").resolve())
        print(f"  Resolved relative to YAML file location:")
        print(f"  data_dir = {cfg.data_dir}")
        assert cfg.data_dir == expected, f"Expected {expected}, got {cfg.data_dir}"


if __name__ == "__main__":
    main()
