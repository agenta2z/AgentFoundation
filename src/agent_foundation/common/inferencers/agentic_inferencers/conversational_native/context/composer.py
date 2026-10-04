"""TurnComposer — renders the three lanes AgentFoundation contributes.

* L1 ``session``      — appended to the vendor system prompt once per session.
* L2 ``turn``         — per-turn host context (``<af_context>``) around the
                         SOP state (``turn_state``).
* L3 ``state_update`` — appended to an AF tool result when SOP state changed.

Rendering goes through the same TemplateManager machinery as the classic
prompt (templated feed values resolved first, shared ``conversation/sections``
rendered with the same feed), so SOP wording is identical in both modes.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from typing import Any, Mapping, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_feed import (
    resolve_feed,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_sections import (
    render_sop_sections,
    sections_used_by,
)

NATIVE_ROOT_SPACE = "conversation_native"
NATIVE_TEMPLATE_TYPE = "main"
L1_KEY = "session"
L2_KEY = "turn"
L2_STATE_KEY = "turn_state"
L3_KEY = "state_update"

_HOST_TAGS = ("af_context", "af_state_update")
_BLOCK_RE = re.compile(
    r"<(af_context|af_state_update)\b[^>]*>.*?</\1>\s*", re.DOTALL | re.IGNORECASE
)
_EXCESS_BLANK_RE = re.compile(r"\n{3,}")
# What a status-and-next-step state update does not repeat: which SOP is
# active and what the full active-SOP block states around it.
_ACTIVE_SOP_IDENTITY_KEYS = (
    "sop_name",
    "sop_instance_id",
    "sop_description",
    "workflow_target_path",
    "session_root_path",
)
_SURROUNDING_IDENTITY_KEYS = ("paused_sop", "inprogress_sops", "catalog_changes")


def new_nonce() -> str:
    return secrets.token_hex(4)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def neutralize_host_tags(text: str) -> str:
    """Make user text / tool output unable to impersonate host blocks."""
    for tag in _HOST_TAGS:
        text = re.sub(rf"<(/?){tag}", rf"&lt;\1{tag}", text, flags=re.IGNORECASE)
    return text


def strip_host_blocks(text: str) -> str:
    """Remove host blocks the model echoed back from user-visible text."""
    return _BLOCK_RE.sub("", text)


def sop_identity(feed: Mapping[str, Any]) -> str:
    """Digest of the SOP state a status-and-next-step update leaves out
    (``_ACTIVE_SOP_IDENTITY_KEYS`` while an SOP is active, and
    ``_SURROUNDING_IDENTITY_KEYS``)."""
    active = bool(feed.get("sop_active"))
    keys = _SURROUNDING_IDENTITY_KEYS + (_ACTIVE_SOP_IDENTITY_KEYS if active else ())
    values = {key: str(feed.get(key) or "") for key in keys}
    digest = json.dumps([active, values], sort_keys=True)
    return text_hash(digest)[:16]


def drifted_parts(recorded: str, current: str) -> tuple[bool, bool]:
    """``(instructions_changed, tools_changed)`` between two L1 core hashes;
    a hash of another form (an older record) counts as both changed."""
    old_instructions, sep, old_tools = recorded.partition(".")
    new_instructions, _, new_tools = current.partition(".")
    if not sep:
        return True, True
    return old_instructions != new_instructions, old_tools != new_tools


def _tidy(text: str) -> str:
    return _EXCESS_BLANK_RE.sub("\n\n", text).strip() + "\n"


class TurnComposer:
    def __init__(self, template_manager: Any, render_string: Any) -> None:
        """``template_manager`` resolves ``conversation_native/main/*`` and the
        shared ``conversation/sections/*``; ``render_string(template, ctx)``
        renders an arbitrary template string (for templated feed values)."""
        self._tm = template_manager
        self._render_string = render_string

    def render(self, key: str, feed: Mapping[str, Any], *, resolve: bool = True) -> str:
        """Render ``key`` with ``feed``: templated feed values resolved and
        the shared sections it references rendered first, unless not
        ``resolve`` (for feeds of rendered text, which must not be rendered
        again)."""
        raw = self._tm.get_raw_template(
            key,
            active_template_type=NATIVE_TEMPLATE_TYPE,
            active_template_root_space=NATIVE_ROOT_SPACE,
        )
        if not raw or raw == self._tm.default_template:
            raise LookupError(
                f"Native template '{NATIVE_ROOT_SPACE}/{NATIVE_TEMPLATE_TYPE}/{key}' "
                "was not found on any template root."
            )
        resolved = dict(feed)
        if resolve:
            resolved = resolve_feed(resolved, self._render_string)
            sections = sections_used_by(raw)
            if sections:
                resolved.update(render_sop_sections(self._tm, resolved, sections))
        text = self._tm(
            key,
            feed=resolved,
            active_template_type=NATIVE_TEMPLATE_TYPE,
            active_template_root_space=NATIVE_ROOT_SPACE,
        )
        return _tidy(text)

    def session_instructions(
        self, feed: Mapping[str, Any], *, tool_manifest: str = ""
    ) -> tuple[str, str]:
        """Render L1; returns ``(text, core_hash)``.

        The core hash covers the instructions and ``tool_manifest`` (the AF
        tool set's fingerprint: a tool change is drift too), as
        ``"<instructions>.<tool_manifest>"`` so a drift can tell which one
        changed (``drifted_parts``). It ignores the catalog snapshot and the
        nonce: a catalog change (delivered later as ``catalog_changes`` in L2)
        is not instruction drift."""
        text = self.render(L1_KEY, feed)
        core_feed = dict(feed, available_sops="", nonce="")
        core_text = self.render(L1_KEY, core_feed)
        return text, f"{text_hash(core_text)}.{tool_manifest}"

    def turn_state(self, feed: Mapping[str, Any]) -> tuple[str, str]:
        """Render the SOP state an L2 carries; returns ``(text, state_hash)``.
        The state hash reads ``"<sop_identity>.<content hash>"`` so a later
        state update can tell whether the agent already holds this SOP's full
        block."""
        text = self.render(L2_STATE_KEY, feed)
        return text, f"{sop_identity(feed)}.{text_hash(text)}"

    def turn_context(
        self, feed: Mapping[str, Any], *, state: Optional[str] = None
    ) -> tuple[str, str]:
        """Render L2 around ``state`` (default: the SOP state rendered from
        ``feed``) and ``feed["notices"]``; returns ``(text, state_hash)`` of
        the full SOP state, whatever ``state`` carries. The hash ignores the
        turn/generation/origin attributes and the notices, so it only changes
        when the SOP state does."""
        full_state, state_hash = self.turn_state(feed)
        text = self.wrap_turn_state(feed, full_state if state is None else state)
        return text, state_hash

    def wrap_turn_state(self, feed: Mapping[str, Any], state: str) -> str:
        """Render L2 around an already rendered SOP ``state``."""
        return self.render(L2_KEY, dict(feed, turn_state=state), resolve=False)

    def state_update(self, feed: Mapping[str, Any]) -> str:
        """Render L3 (plan §3.1): the full active-SOP block when the SOP
        identity differs from that of ``feed["delivered_state"]`` (the state
        hash of the latest L2 or L3 the agent received; "" when unknown),
        else only the active SOP's status and next step."""
        delivered, sep, _ = str(feed.get("delivered_state") or "").partition(".")
        progress_only = bool(sep) and delivered == sop_identity(feed)
        return self.render(L3_KEY, dict(feed, sop_progress_only=progress_only))
