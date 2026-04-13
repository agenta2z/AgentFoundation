#!/usr/bin/env python3
"""Example 01: Registry Basics — register, resolve, and list target aliases.

This example shows how the target registry maps short alias names to full
Python import paths, making YAML configs human-readable.

Expected output:
================

=== 1. Register aliases (3 ways) ===
  Decorator:  @register('Echo') on EchoInferencer
  Imperative: register_class(LoudEcho, 'LoudEcho')
  String-only: register_alias('QuietEcho', 'example_01_registry_basics.QuietEchoInferencer')

=== 2. Resolve aliases -> full import paths ===
  resolve_target('Echo')      -> example_01_registry_basics.EchoInferencer
  resolve_target('LoudEcho')  -> example_01_registry_basics.LoudEchoInferencer
  resolve_target('QuietEcho') -> example_01_registry_basics.QuietEchoInferencer

=== 3. Full dotted paths still work (no registration needed) ===
  resolve_target('os.path.join') -> os.path.join

=== 4. List all registered aliases ===
  All: {'Echo': '...EchoInferencer', 'LoudEcho': '...LoudEchoInferencer', ...}

=== 5. List by category ===
  Category 'inferencer': {'Echo': ..., 'LoudEcho': ...}
  Category 'config':     {'QuietEcho': ...}

=== 6. Unknown alias -> helpful error ===
  KeyError: "Unknown target alias: 'NonExistent'. Registered: ['Echo', 'LoudEcho', 'QuietEcho']. ..."
"""

import os
import sys

# --- Path setup (makes script runnable standalone) ---
_script_dir = os.path.dirname(os.path.abspath(__file__))
_core_root = os.path.normpath(os.path.join(_script_dir, "..", "..", "..", "..", ".."))
for _sub in ("AgentFoundation/src", "RichPythonUtils/src"):
    _p = os.path.normpath(os.path.join(_core_root, _sub))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from rich_python_utils.config_utils import (
    register,
    register_alias,
    register_class,
    resolve_target,
    list_registered,
    _reset_registry,
)


# ---------------------------------------------------------------------------
# Sample classes (stand-ins for real inferencers)
# ---------------------------------------------------------------------------

@register("Echo", category="inferencer")
class EchoInferencer:
    """Simply echoes back the input."""
    def __init__(self, prefix=""):
        self.prefix = prefix

    def infer(self, text):
        return f"{self.prefix}{text}"


class LoudEchoInferencer:
    """Echoes back in UPPERCASE."""
    def __init__(self, volume=10):
        self.volume = volume

    def infer(self, text):
        return text.upper()


class QuietEchoInferencer:
    """Echoes back in lowercase."""
    def infer(self, text):
        return text.lower()


def separator(title: str):
    print(f"\n{'=' * 3} {title} {'=' * 3}")


def main():
    # ── 1. Three ways to register ────────────────────────────
    separator("1. Register aliases (3 ways)")

    # Way 1: @register decorator (already applied above)
    print(f"  Decorator:   @register('Echo') on EchoInferencer")

    # Way 2: Imperative registration
    register_class(LoudEchoInferencer, "LoudEcho", category="inferencer")
    print(f"  Imperative:  register_class(LoudEcho, 'LoudEcho')")

    # Way 3: String-only (no import needed — great for bulk registration)
    register_alias(
        "QuietEcho",
        f"{__name__}.QuietEchoInferencer",
        category="config",  # different category for demo
    )
    print(f"  String-only: register_alias('QuietEcho', '{__name__}.QuietEchoInferencer')")

    # ── 2. Resolve aliases to full import paths ──────────────
    separator("2. Resolve aliases -> full import paths")
    for alias in ("Echo", "LoudEcho", "QuietEcho"):
        path = resolve_target(alias)
        print(f"  resolve_target({alias!r:12s}) -> {path}")

    # ── 3. Full paths pass through ───────────────────────────
    separator("3. Full dotted paths still work (no registration needed)")
    path = resolve_target("os.path.join")
    print(f"  resolve_target('os.path.join') -> {path}")

    # ── 4. List all ──────────────────────────────────────────
    separator("4. List all registered aliases")
    all_aliases = list_registered()
    print(f"  All: {all_aliases}")

    # ── 5. List by category ──────────────────────────────────
    separator("5. List by category")
    print(f"  Category 'inferencer': {list_registered('inferencer')}")
    print(f"  Category 'config':     {list_registered('config')}")

    # ── 6. Error handling ────────────────────────────────────
    separator("6. Unknown alias -> helpful error")
    try:
        resolve_target("NonExistent")
    except KeyError as e:
        print(f"  KeyError: {e}")


if __name__ == "__main__":
    main()
