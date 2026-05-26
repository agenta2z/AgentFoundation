"""High-level prompt helpers for CLI interaction.

Each function lazily imports its Rich / prompt_toolkit dependency so
the rest of the codebase can ``import agent_foundation.ui.cli`` without
pulling in heavy terminal libraries.
"""
from __future__ import annotations

from typing import List, Optional


def ask_confirm(prompt: str = "Continue?", default: bool = False) -> bool:
    """Yes/no confirmation using Rich's ``Confirm.ask``.

    Falls back to plain ``input()`` when Rich is not installed.
    """
    try:
        from rich.prompt import Confirm
        return Confirm.ask(prompt, default=default)
    except ImportError:
        suffix = " [Y/n]" if default else " [y/N]"
        answer = input(prompt + suffix + " ").strip().lower()
        if not answer:
            return default
        return answer in ("y", "yes")


def ask_single_choice(prompt: str, options: List[str]) -> str:
    """Display a numbered list of *options* and return the selected value.

    Falls back to plain ``input()`` when Rich is not installed.
    """
    try:
        from rich.console import Console
        from rich.prompt import IntPrompt
        from agent_foundation.ui.cli.theme import ThemeManager

        console = Console(theme=ThemeManager.get_theme())
        console.print(f"\n[bold]{prompt}[/bold]")
        for idx, opt in enumerate(options, 1):
            console.print(f"  [cyan]{idx}.[/cyan] {opt}")

        choice = IntPrompt.ask(
            "Select",
            choices=[str(i) for i in range(1, len(options) + 1)],
            default=1,
        )
        return options[choice - 1]
    except ImportError:
        print(f"\n{prompt}")
        for idx, opt in enumerate(options, 1):
            print(f"  {idx}. {opt}")
        while True:
            raw = input("Select [1]: ").strip()
            if not raw:
                return options[0]
            try:
                num = int(raw)
                if 1 <= num <= len(options):
                    return options[num - 1]
            except ValueError:
                pass
            print(f"Please enter a number between 1 and {len(options)}.")


def ask_text(prompt: str = "> ", multiline: bool = False) -> str:
    """Collect free-form text input via prompt_toolkit.

    Falls back to plain ``input()`` when prompt_toolkit is not installed.

    Parameters
    ----------
    prompt:
        The prompt string shown to the user.
    multiline:
        If *True*, enable multiline editing (submit with Alt+Enter).
    """
    try:
        from prompt_toolkit import PromptSession

        session = PromptSession()
        return session.prompt(prompt, multiline=multiline)
    except ImportError:
        return input(prompt)
