"""Rich-powered terminal interactive transport.

``RichTerminalInteractive`` extends ``RichInteractiveBase`` to provide a
fully Rich-rendered CLI experience: panels for agent/user messages,
input-mode-aware prompting (choices, confirmations, free text), and
streaming token display via ``StreamingPanel``.

All Rich / prompt_toolkit imports live inside method bodies so that
importing this module does **not** eagerly pull in the libraries.
"""
from __future__ import annotations

import asyncio
from typing import Any, Iterable, Optional

from attr import attrs, attrib

from agent_foundation.ui.interactive_base import InteractionFlags
from agent_foundation.ui.rich_interactive_base import RichInteractiveBase


@attrs
class RichTerminalInteractive(RichInteractiveBase):
    """Terminal transport with Rich rendering and structured input modes."""

    pending_message: str = attrib(default="Awaiting further input ...")

    # -- InteractiveBase abstract method implementations ----------------------

    def _send_response(
        self,
        response: Any,
        flag: InteractionFlags = InteractionFlags.TurnCompleted,
    ) -> None:
        from rich.console import Console
        from rich.panel import Panel
        from rich.markdown import Markdown
        from agent_foundation.ui.cli.theme import ThemeManager, COLORS
        from agent_foundation.ui.input_modes import InputMode

        console = Console(theme=ThemeManager.get_theme())
        text = str(response)

        # Decide border color based on input mode context
        if (
            self._current_input_mode is not None
            and hasattr(self._current_input_mode, "mode")
            and self._current_input_mode.mode == InputMode.PRESS_TO_CONTINUE
        ):
            border = COLORS["info"]
        else:
            border = COLORS["assistant_border"]

        console.print(
            Panel(
                Markdown(text),
                title=self.system_name,
                border_style=border,
            )
        )

    def _get_input(self) -> Any:
        from agent_foundation.ui.input_modes import InputMode

        mode_cfg = self._pending_input_mode

        if mode_cfg is None:
            return self._collect_free_text()

        mode = mode_cfg.mode

        if mode == InputMode.PRESS_TO_CONTINUE:
            return self._collect_press_to_continue(mode_cfg)
        elif mode == InputMode.SINGLE_CHOICE:
            return self._collect_single_choice(mode_cfg)
        elif mode == InputMode.MULTIPLE_CHOICE:
            return self._collect_multiple_choice(mode_cfg)
        elif mode == InputMode.EXACT_STRING:
            return self._collect_exact_string(mode_cfg)
        else:
            return self._collect_free_text(mode_cfg)

    def reset_input(self, flag: InteractionFlags) -> None:
        pass

    def _send_pending_message(self) -> None:
        try:
            from rich.console import Console
            from agent_foundation.ui.cli.theme import ThemeManager

            console = Console(theme=ThemeManager.get_theme())
            console.print(f"\n[dim]{self.pending_message}[/dim]\n")
        except ImportError:
            print(f"\n{self.pending_message}\n")

    # -- input collectors (lazy imports) --------------------------------------

    def _collect_free_text(self, mode_cfg=None) -> str:
        from agent_foundation.ui.cli.prompts import ask_text

        prompt_text = f"{self.user_name}: "
        if mode_cfg and mode_cfg.prompt:
            prompt_text = mode_cfg.prompt + " "
        return ask_text(prompt_text)

    def _collect_press_to_continue(self, mode_cfg) -> str:
        prompt_text = mode_cfg.prompt or "Press Enter to continue..."
        input(prompt_text)
        return ""

    def _collect_single_choice(self, mode_cfg):
        from rich.console import Console
        from agent_foundation.ui.cli.theme import ThemeManager

        console = Console(theme=ThemeManager.get_theme())
        prompt_text = mode_cfg.prompt or "Choose one:"
        console.print(f"\n[bold]{prompt_text}[/bold]")

        for idx, opt in enumerate(mode_cfg.options):
            desc = f"  [dim]- {opt.description}[/dim]" if opt.description else ""
            console.print(f"  [cyan]{idx + 1}.[/cyan] {opt.label}{desc}")

        if mode_cfg.allow_custom:
            console.print(f"  [cyan]{len(mode_cfg.options) + 1}.[/cyan] [dim](custom)[/dim]")

        while True:
            raw = input("Select: ").strip()
            if not raw:
                continue
            try:
                num = int(raw)
                if 1 <= num <= len(mode_cfg.options):
                    chosen = mode_cfg.options[num - 1]
                    if chosen.follow_up_prompt:
                        follow_up = input(chosen.follow_up_prompt + " ").strip()
                        return {"choice_index": num - 1, "follow_up_value": follow_up}
                    return {"choice_index": num - 1}
                elif mode_cfg.allow_custom and num == len(mode_cfg.options) + 1:
                    custom = input("Enter custom value: ").strip()
                    return {"custom_text": custom}
            except ValueError:
                if mode_cfg.allow_custom:
                    return {"custom_text": raw}
            console.print("[red]Invalid selection.[/red]")

    def _collect_multiple_choice(self, mode_cfg):
        from rich.console import Console
        from agent_foundation.ui.cli.theme import ThemeManager

        console = Console(theme=ThemeManager.get_theme())
        prompt_text = mode_cfg.prompt or "Choose one or more (comma-separated):"
        console.print(f"\n[bold]{prompt_text}[/bold]")

        for idx, opt in enumerate(mode_cfg.options):
            desc = f"  [dim]- {opt.description}[/dim]" if opt.description else ""
            console.print(f"  [cyan]{idx + 1}.[/cyan] {opt.label}{desc}")

        if mode_cfg.show_select_all:
            console.print(f"  [cyan]A.[/cyan] {mode_cfg.select_all_text}")

        raw = input("Select: ").strip()
        if raw.upper() == "A" and mode_cfg.show_select_all:
            return {
                "selections": [{"choice_index": i} for i in range(len(mode_cfg.options))]
            }

        selections = []
        for part in raw.split(","):
            part = part.strip()
            try:
                num = int(part)
                if 1 <= num <= len(mode_cfg.options):
                    selections.append({"choice_index": num - 1})
            except ValueError:
                if part:
                    selections.append({"custom_text": part})
        return {"selections": selections}

    def _collect_exact_string(self, mode_cfg) -> str:
        prompt_text = mode_cfg.prompt or f"Type '{mode_cfg.expected_string}' to confirm: "
        return input(prompt_text)

    # -- streaming support ----------------------------------------------------

    def stream_token_batches(
        self,
        token_stream: Iterable[str],
        session_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> str:
        """Render tokens in a Rich Live panel as they arrive.

        Parameters
        ----------
        token_stream:
            Iterable (or generator) yielding string tokens.
        session_id:
            Optional session identifier (currently informational).
        title:
            Title shown on the streaming panel; defaults to ``self.system_name``.

        Returns
        -------
        str
            The fully accumulated text.
        """
        from agent_foundation.ui.cli.streaming import StreamingPanel

        panel_title = title or self.system_name
        with StreamingPanel(title=panel_title) as panel:
            for token in token_stream:
                panel.append(token)
        return panel._buffer

    # -- async wrappers -------------------------------------------------------

    async def asend_response(self, response, flag=InteractionFlags.TurnCompleted, **kwargs):
        """Async wrapper -- offloads to thread since Rich is synchronous."""
        await asyncio.to_thread(self.send_response, response, flag, **kwargs)

    async def aget_input(self):
        """Async wrapper -- offloads blocking input to a thread."""
        return await asyncio.to_thread(self.get_input)
