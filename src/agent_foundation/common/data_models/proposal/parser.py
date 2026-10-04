"""Proposal parser — read/write ``ProposalIndex`` from workspace artifacts.

Three strategies in priority order:
    A. Read ``outputs/proposals.json`` sidecar (fast path, AF-native).
    B. Extract ``proposal_index`` JSON fence from markdown (reuses
       ``_extract_json_block`` from ``flow_parsers``).
    C. Regex-parse a Priority Ranking Table from markdown (last resort,
       recovers ``id``, ``rank``, ``title`` only).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .model import Proposal, ProposalGroup, ProposalIndex

_logger = logging.getLogger(__name__)

_SIDECAR_NAME = "proposals.json"

_FENCE_RE = re.compile(r"```json\s+proposal_index\b[^\n]*\n([\s\S]*?)\n\s*```")

_TABLE_ROW_RE = re.compile(
    r"^\s*\|\s*(\d+)\s*\|\s*([\w-]+)\s*\|\s*(.+?)\s*\|",
    re.MULTILINE,
)

# Matches a hub-shaped hypothesis id (``H12``) so the producer-boundary adapter
# can canonicalise it to the AF proposal id form (``P12``). Anchored + numeric
# tail so genuinely AF-native ids (``P12``, ``P1-a``) are left untouched, making
# the rewrite idempotent.
_H_ID_RE = re.compile(r"^H(\d+)([A-Za-z0-9_-]*)$")


# ---------------------------------------------------------------------------
# Producer-boundary adapter (correction #16)
# ---------------------------------------------------------------------------


def _canonicalize_id(value: Any) -> str:
    """Rewrite a single hub ``H#`` id to the canonical AF ``P#`` id.

    Idempotent: ids that are not hub-shaped (already ``P#``, or any non-``H#``
    token) pass through unchanged. Non-strings are coerced via ``str``.
    """
    s = str(value)
    m = _H_ID_RE.match(s)
    if m is None:
        return s
    return f"P{m.group(1)}{m.group(2)}"


def _canonicalize_id_list(values: Any) -> list[str]:
    """Apply :func:`_canonicalize_id` across a scalar-or-list of ids."""
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [_canonicalize_id(v) for v in values]
    return [_canonicalize_id(values)]


def canonicalize_proposal_index_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise an externally-produced proposal-index dict to AF-canonical form.

    This is the single producer-side boundary adapter (correction #16). The
    Experiment Hub the framework feeds is RankEvolve-shaped (H-ids: ``phases`` /
    ``hypothesis_ids`` / ``combo_constraints`` / ``H#``), while AF pins
    **proposal ids (``P#``) as canonical end-to-end**. Applying the rename once
    here — where the producer writes the sidecar — means every downstream read
    site sees canonical P-ids and never needs a runtime P→H translation.

    Translations applied (each idempotent for already-canonical input):
      * top-level ``phases`` → ``groups``
      * ``combo_constraints`` → ``constraints``
      * within batches: ``hypothesis_ids`` → ``proposal_ids``
      * within constraints: ``hypothesis_ids`` → ``proposal_ids``
      * every id token ``H#`` → ``P#`` (proposal ids, batch/constraint id lists,
        ``requires_ids``, and per-proposal ``dependencies``)

    The input dict is not mutated; a new normalised dict is returned. Unknown
    keys are preserved untouched so the adapter is forward-compatible.
    """
    if not isinstance(raw, dict):
        return raw

    out: dict[str, Any] = dict(raw)

    # phases -> groups (only adopt the alias when the canonical key is absent so
    # an AF-native dict that already has `groups` is never clobbered).
    groups_src = out.pop("phases", None)
    if "groups" not in out and groups_src is not None:
        out["groups"] = groups_src
    elif groups_src is not None:
        # Both present (degenerate); prefer canonical `groups`, drop `phases`.
        pass

    # Flat-dialect fold (Fix 1): a recovered/aggregator fence may emit a FLAT
    # index — a top-level ``proposals[]`` with NO ``groups``/``phases`` (observed
    # axis-namespaced shape: ``{totals, phase_summary, proposals:[{id, phase,
    # title, cost, depends_on, ...}]}``). ``ProposalIndex.from_dict`` reads only
    # ``groups[].proposals[]``, so without this fold such a fence parses to an
    # EMPTY index (Defect C, the empty-widget bug). This is the producer-boundary
    # "be liberal in what you accept" half.
    if not out.get("groups") and isinstance(out.get("proposals"), list):
        out = _fold_flat_proposal_index(out)

    out["groups"] = [_canonicalize_group(g) for g in out.get("groups", []) or []]

    # combo_constraints -> constraints (same non-clobber rule).
    combo_src = out.pop("combo_constraints", None)
    if "constraints" not in out and combo_src is not None:
        out["constraints"] = combo_src
    out["constraints"] = [
        _canonicalize_constraint(c) for c in out.get("constraints", []) or []
    ]

    return out


def _canonicalize_group(group: Any) -> dict[str, Any]:
    """Canonicalise one group/phase entry (proposals + batches)."""
    if not isinstance(group, dict):
        return group
    g: dict[str, Any] = dict(group)
    g["proposals"] = [_canonicalize_proposal(p) for p in g.get("proposals", []) or []]
    if g.get("batches"):
        g["batches"] = [_canonicalize_batch(b) for b in g["batches"] or []]
    return g


def _canonicalize_proposal(proposal: Any) -> dict[str, Any]:
    """Canonicalise a single proposal's id + id-bearing reference fields."""
    if not isinstance(proposal, dict):
        return proposal
    p: dict[str, Any] = dict(proposal)
    if "id" in p:
        p["id"] = _canonicalize_id(p["id"])
    if p.get("dependencies"):
        p["dependencies"] = _canonicalize_id_list(p["dependencies"])
    return p


def _canonicalize_batch(batch: Any) -> dict[str, Any]:
    """Canonicalise a batch: ``hypothesis_ids`` → ``proposal_ids`` (P#)."""
    if not isinstance(batch, dict):
        return batch
    b: dict[str, Any] = dict(batch)
    ids_src = b.pop("hypothesis_ids", None)
    if "proposal_ids" not in b and ids_src is not None:
        b["proposal_ids"] = ids_src
    if b.get("proposal_ids"):
        b["proposal_ids"] = _canonicalize_id_list(b["proposal_ids"])
    return b


def _canonicalize_constraint(constraint: Any) -> dict[str, Any]:
    """Canonicalise a constraint: ``hypothesis_ids`` → ``proposal_ids`` (P#)."""
    if not isinstance(constraint, dict):
        return constraint
    c: dict[str, Any] = dict(constraint)
    ids_src = c.pop("hypothesis_ids", None)
    if "proposal_ids" not in c and ids_src is not None:
        c["proposal_ids"] = ids_src
    if c.get("proposal_ids"):
        c["proposal_ids"] = _canonicalize_id_list(c["proposal_ids"])
    if c.get("requires_ids"):
        c["requires_ids"] = _canonicalize_id_list(c["requires_ids"])
    return c


# ---------------------------------------------------------------------------
# Flat-dialect fold (Fix 1) — a top-level ``proposals[]`` with no groups/phases
# ---------------------------------------------------------------------------


def _coerce_int(value: Any, default: int) -> int:
    """Best-effort int coercion (tolerates ``None``/str/float); ``default`` on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _split_phase_label(key: Any) -> tuple[int | None, str]:
    """Parse a ``phase_summary`` key like ``"0_measurement"`` → ``(0, "measurement")``.

    A key with no leading integer (e.g. ``"Quick Wins"``) → ``(None, <key>)`` so
    the caller falls back to a synthetic ``"Phase N"`` label.
    """
    s = str(key)
    m = re.match(r"^(\d+)[_\s-]*(.*)$", s)
    if not m:
        return None, s
    label = m.group(2).replace("_", " ").strip()
    ph = int(m.group(1))
    return ph, (label or f"Phase {ph}")


def _map_flat_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    """Map a FLAT-dialect proposal to the canonical ``Proposal`` field names.

    ``cost`` → ``complexity``, ``depends_on`` → ``dependencies``, ``title`` →
    ``summary`` (fallback only). Non-schema fields (``axis``/``risk``/
    ``feasibility``/``cost``) are preserved under ``metadata`` so nothing is lost;
    ``Proposal.from_dict`` ignores any leftover unknown top-level keys. The input
    is not mutated.
    """
    q: dict[str, Any] = dict(proposal)
    if "complexity" not in q and q.get("cost") is not None:
        q["complexity"] = q["cost"]
    if "dependencies" not in q and q.get("depends_on") is not None:
        q["dependencies"] = q["depends_on"]
    if not q.get("summary") and q.get("title"):
        q["summary"] = q["title"]
    meta = dict(q.get("metadata") or {})
    for extra in ("axis", "risk", "feasibility", "cost"):
        if extra in q and extra not in meta:
            meta[extra] = q[extra]
    if meta:
        q["metadata"] = meta
    return q


def _fold_flat_proposal_index(raw: dict[str, Any]) -> dict[str, Any]:
    """Fold a FLAT proposal-index dialect into the canonical grouped shape (Fix 1).

    A recovered/aggregator fence sometimes emits a flat index — a top-level
    ``proposals[]`` and NO ``groups``/``phases`` (observed:
    ``{version, totals, phase_summary, proposals:[{id, phase, title, cost,
    depends_on, ...}]}``). ``ProposalIndex.from_dict`` reads only
    ``groups[].proposals[]``, so without this fold such a fence parses to an
    EMPTY index (Defect C).

    Groups proposals by their integer ``phase`` (default ``0``; first-seen order
    preserved), maps flat field names via :func:`_map_flat_proposal`, derives
    group labels from ``phase_summary`` keys when present, and sets
    ``total_count`` from ``totals.all`` else the proposal count. Emits a WARNING
    whenever it fires so format drift is always surfaced. The input is not
    mutated. Only the caller's "no groups + list ``proposals``" guard invokes it.
    """
    out: dict[str, Any] = dict(raw)
    flat = [p for p in (out.pop("proposals", []) or []) if isinstance(p, dict)]
    _logger.warning(
        "proposal_index not in the expected grouped schema; folding flat dialect "
        "as a fallback (top-level keys: %s)",
        sorted(k for k in raw if k != "proposals"),
    )

    phase_labels: dict[int, str] = {}
    phase_summary = out.get("phase_summary")
    if isinstance(phase_summary, dict):
        for k in phase_summary:
            ph, label = _split_phase_label(k)
            if ph is not None:
                phase_labels.setdefault(ph, label)

    grouped: dict[int, list[dict[str, Any]]] = {}
    order: list[int] = []
    for p in flat:
        ph = _coerce_int(p.get("phase"), 0)
        if ph not in grouped:
            grouped[ph] = []
            order.append(ph)
        grouped[ph].append(_map_flat_proposal(p))

    out["groups"] = [
        {
            "phase": ph,
            "label": phase_labels.get(ph, f"Phase {ph}"),
            "proposals": grouped[ph],
        }
        for ph in order
    ]

    if "total_count" not in out:
        totals = out.get("totals")
        if isinstance(totals, dict) and totals.get("all") is not None:
            out["total_count"] = _coerce_int(totals.get("all"), len(flat))
        else:
            out["total_count"] = len(flat)

    return out


# ---------------------------------------------------------------------------
# Consumer-boundary adapter (AF → hub dialect)
# ---------------------------------------------------------------------------


def canonicalize_proposal_index_to_hub_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert an AF-canonical proposal-index dict to the hub dialect.

    Inverse of :func:`canonicalize_proposal_index_dict`. Used at the single
    OpenStartup hub boundary (``tool_dispatcher.py::create_experiment_hub`` /
    ``open_experiment_hub`` / ``_init_dashboard_state``) to normalise the
    payload once so every downstream hub consumer speaks the hub dialect
    natively (``open_experiment_hub`` validation, ``group_selected_by_batch``,
    FE ``MultiChoiceComboView``, ``ProposalSelectionWidget``'s existing rich
    path). Keeps the widget's Phase-1 JS normaliser as a defensive fallback
    (idempotent when it sees already-hub data).

    Translations applied (each idempotent for already-hub input):
      * top-level ``groups`` → ``phases``
      * ``constraints`` → ``combo_constraints``
      * within batches: ``proposal_ids`` → ``hypothesis_ids``
      * within constraints: ``proposal_ids`` → ``hypothesis_ids``

    **Intentionally does NOT rewrite id tokens** (no ``P#`` → ``H#``). AF pins
    proposal ids as ``P#`` end-to-end and the hub treats ids as opaque strings;
    keeping ``P#`` in the hub payload gives users a consistent id across
    surfaces (in-chat card matches Selection tab card matches Review&Combo
    row). The hub→AF direction does rewrite (``H#`` → ``P#``) because that
    direction owns id canonicalisation.

    The input dict is not mutated; a new normalised dict is returned. Unknown
    keys are preserved untouched so the adapter is forward-compatible.
    Proposals inside groups pass through by SPREAD (never a whitelist) so
    hub-only fields like ``includes`` / ``probability`` / ``theme`` /
    ``source_workers`` / ``notes`` survive when they're already present.
    """
    if not isinstance(raw, dict):
        return raw

    out: dict[str, Any] = dict(raw)

    # groups -> phases (only adopt the alias when the canonical key is absent
    # so an already-hub dict with `phases` is never clobbered).
    phases_src = out.pop("groups", None)
    if "phases" not in out and phases_src is not None:
        out["phases"] = phases_src

    out["phases"] = [_to_hub_phase(g) for g in out.get("phases", []) or []]

    # constraints -> combo_constraints (same non-clobber rule).
    combo_src = out.pop("constraints", None)
    if "combo_constraints" not in out and combo_src is not None:
        out["combo_constraints"] = combo_src
    out["combo_constraints"] = [
        _to_hub_constraint(c) for c in out.get("combo_constraints", []) or []
    ]

    return out


def _to_hub_phase(group: Any) -> dict[str, Any]:
    """Convert one AF-canonical group entry to its hub `phase` form."""
    if not isinstance(group, dict):
        return group
    g: dict[str, Any] = dict(group)
    # Proposals: SPREAD by shallow copy so any hub-only fields already present
    # (e.g., a caller that fed a partially-hub payload) survive. Non-dict
    # entries pass through unchanged.
    proposals = g.get("proposals", []) or []
    g["proposals"] = [dict(p) if isinstance(p, dict) else p for p in proposals]
    if g.get("batches"):
        g["batches"] = [_to_hub_batch(b) for b in g["batches"] or []]
    return g


def _to_hub_batch(batch: Any) -> dict[str, Any]:
    """Convert a batch: ``proposal_ids`` → ``hypothesis_ids``.

    Ids inside the list are NOT rewritten (AF ``P#`` stays ``P#``).
    """
    if not isinstance(batch, dict):
        return batch
    b: dict[str, Any] = dict(batch)
    ids_src = b.pop("proposal_ids", None)
    if "hypothesis_ids" not in b and ids_src is not None:
        b["hypothesis_ids"] = list(ids_src)
    elif b.get("hypothesis_ids"):
        b["hypothesis_ids"] = list(b["hypothesis_ids"])
    return b


def _to_hub_constraint(constraint: Any) -> dict[str, Any]:
    """Convert a constraint: ``proposal_ids`` → ``hypothesis_ids``.

    Ids inside id-lists are NOT rewritten (AF ``P#`` stays ``P#``).
    """
    if not isinstance(constraint, dict):
        return constraint
    c: dict[str, Any] = dict(constraint)
    ids_src = c.pop("proposal_ids", None)
    if "hypothesis_ids" not in c and ids_src is not None:
        c["hypothesis_ids"] = list(ids_src)
    elif c.get("hypothesis_ids"):
        c["hypothesis_ids"] = list(c["hypothesis_ids"])
    # requires_ids: preserve as-is (no rename, no id rewrite).
    if c.get("requires_ids"):
        c["requires_ids"] = list(c["requires_ids"])
    return c


# ---------------------------------------------------------------------------
# Enrichment: resolve per-proposal `proposal_file` to an absolute path
# ---------------------------------------------------------------------------


def attach_proposal_file_abs(
    proposals: dict[str, Any], proposals_json_path: str | Path
) -> None:
    """Mutate ``proposals`` in place: set ``p['proposal_file_abs']`` for each proposal.

    For each proposal whose ``proposal_file`` names a file that exists on disk under
    ``dirname(proposals_json_path)``, attach the resolved absolute path as
    ``proposal_file_abs``. The frontend fetches this via ``GET /api/view/<abs>``
    (whose allowed bases include ``runtime_root``, where these docs live) to render
    the full ``P{N}.md`` in the proposal-selection widget.

    ``proposals_json_path`` MUST be the path to the proposals.JSON FILE (not its
    parent dir) — the base for relative ``proposal_file`` values is
    ``Path(proposals_json_path).parent``. Passing a directory would misplace the
    base by one level and every containment check would fail.

    Guards:
      * Non-dict ``proposals`` → no-op.
      * Non-list ``groups``/``phases`` → skipped (never a ``dict + list`` TypeError).
      * Non-dict entries in the tree → skipped.
      * Missing/empty ``proposal_file`` on a proposal → key omitted.
      * Path traversal (``../..``) after symlink resolution → key omitted.
      * File does not exist → key omitted.

    Absent ``proposal_file_abs`` on a proposal → the widget gracefully falls back
    to its inline (JSON-derived) detail fields. This helper NEVER raises on
    malformed input, so callers may invoke it inside a narrow try/except without
    widening exception handling.

    Handles both dialects (``groups`` for AF-native, ``phases`` for hub-canonicalized)
    defensively — current callers all feed ``groups``, but the helper lives in this
    module where the canonicalizer's ``phases`` output could later be fed in.
    """
    if not isinstance(proposals, dict):
        return
    try:
        base = Path(str(proposals_json_path)).parent.resolve()
    except (OSError, ValueError):
        return
    for key in ("groups", "phases"):
        groups = proposals.get(key)
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            for p in group.get("proposals") or []:
                if not isinstance(p, dict):
                    continue
                rel = p.get("proposal_file")
                if not rel:
                    continue
                try:
                    cand = (base / rel).resolve()
                    cand.relative_to(base)
                except (OSError, ValueError):
                    continue
                if cand.is_file():
                    p["proposal_file_abs"] = str(cand)


# ---------------------------------------------------------------------------
# D6 — Defensive normalizer for LLM-authored ``summary`` fields
# ---------------------------------------------------------------------------


def _looks_like_bad_summary(s: str) -> bool:
    """A ``summary`` is 'bad' when it can't serve as a real one-line preview.

    * Starts with ``|`` → an accidentally-pasted markdown table row.
    * Starts and ends with ``**`` → a bold-only "heading" fragment.
    * Ends with ``…`` or ``...`` → the LLM truncated its own output.
    """
    stripped = (s or "").strip()
    if not stripped:
        return False
    if stripped.startswith("|"):
        return True
    if stripped.startswith("**") and stripped.endswith("**"):
        return True
    if stripped.endswith("…") or stripped.endswith("..."):
        return True
    return False


def _first_prose_sentence(text: str, cap: int = 200) -> str:
    """First sentence of ``text``, stopping at ``. `` / ``! `` / ``? `` / newline.

    Trimmed and capped at ``cap`` chars so an untruncated ``problem`` field still
    produces a valid one-liner even if it happens to lack sentence terminators.
    """
    t = (text or "").strip()
    if not t:
        return ""
    best = len(t)
    for stop in (". ", "! ", "? ", "\n"):
        idx = t.find(stop)
        if idx > 0 and idx + 1 < best:
            best = idx + 1
    return t[:best].strip()[:cap]


def _normalize_proposal_summary(proposal: dict[str, Any]) -> None:
    """Mutate a proposal dict in place: replace a bad ``summary`` with a derived one.

    Priority for the replacement value: first prose sentence of ``problem``, else
    ``title``. Also emits ``logger.warning`` when ANY prose field ends in the
    truncation marker — surfaces LLM misbehavior for triage without persisting
    garbage into the widget's collapsed-card preview.
    """
    for prose_field in ("summary", "problem", "approach", "notes"):
        val = proposal.get(prose_field)
        if isinstance(val, str):
            s = val.rstrip()
            if s.endswith("…") or s.endswith("..."):
                _logger.warning(
                    "Proposal %s: %r field ends with truncation marker; "
                    "LLM output was truncated at the source",
                    proposal.get("id", "?"),
                    prose_field,
                )
    summary = proposal.get("summary", "")
    if not isinstance(summary, str) or not _looks_like_bad_summary(summary):
        return
    derived = _first_prose_sentence(proposal.get("problem", ""))
    if not derived:
        derived = str(proposal.get("title", "") or "").strip()
    if derived:
        proposal["summary"] = derived


def _normalize_proposal_index_summaries(data: dict[str, Any]) -> None:
    """Walk a proposal-index dict and normalize every proposal's ``summary``.

    Handles both dialects (``groups``/``phases``). Called at the parser boundary
    (``parse_proposal_file`` + ``_strategy_b``) so the sidecar the widget reads
    always has a clean one-liner for its collapsed-card preview.
    """
    if not isinstance(data, dict):
        return
    for key in ("groups", "phases"):
        groups = data.get(key)
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            for p in group.get("proposals") or []:
                if isinstance(p, dict):
                    _normalize_proposal_summary(p)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_proposals(workspace: Path) -> ProposalIndex | None:
    """Try Strategy A → B → C. Returns ``None`` only if all three fail."""
    result = _strategy_a(workspace)
    if result is not None:
        return result

    md_path = _find_markdown(workspace)
    if md_path is not None:
        text = md_path.read_text(encoding="utf-8", errors="replace")
        result = _strategy_b(text)
        if result is not None:
            return result
        result = _strategy_c(text)
        if result is not None:
            return result

    return None


def parse_proposal_file(path: Path) -> ProposalIndex | None:
    """Strategy A: load ``proposals.json`` sidecar directly."""
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        _normalize_proposal_index_summaries(data)
        return ProposalIndex.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        _logger.warning("Failed to parse %s: %s", path, exc)
        return None


def parse_proposal_index_from_text(text: str) -> ProposalIndex | None:
    """Extract ``ProposalIndex`` from text containing a JSON fence."""
    return _strategy_b(text)


def write_proposal_index(path: Path, index: ProposalIndex) -> None:
    """Atomic write: tmp → fsync → rename. Never produces partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = index.to_dict()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _strategy_a(workspace: Path) -> ProposalIndex | None:
    """Read sidecar ``outputs/proposals.json``."""
    for candidate in (
        workspace / "outputs" / _SIDECAR_NAME,
        workspace / _SIDECAR_NAME,
    ):
        result = parse_proposal_file(candidate)
        if result is not None:
            return result
    return None


def _strategy_b(text: str) -> ProposalIndex | None:
    """Extract ``proposal_index`` JSON fence from markdown.

    This is the producer boundary: the aggregator's LLM output enters here as a
    raw fence. Whether the model emits the AF-native (``groups`` / ``P#``) or
    the hub-shaped (``phases`` / ``hypothesis_ids`` / ``H#``) dialect, the dict
    is canonicalised to AF proposal ids via
    :func:`canonicalize_proposal_index_dict` before it becomes a
    :class:`ProposalIndex` — so the sidecar BTA writes (and every downstream
    read) is already P-id canonical (correction #16).
    """
    m = _FENCE_RE.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        if not isinstance(data, dict):
            return None
        canonical = canonicalize_proposal_index_dict(data)
        _normalize_proposal_index_summaries(canonical)
        return ProposalIndex.from_dict(canonical)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        _logger.warning("Malformed proposal_index fence: %s", exc)
        return None


def _strategy_c(text: str) -> ProposalIndex | None:
    """Regex-parse a Priority Ranking Table. Recovers ``id``, ``rank``, ``title``."""
    rows = _TABLE_ROW_RE.findall(text)
    if not rows:
        return None
    proposals: list[Proposal] = []
    for rank_str, pid, title in rows:
        try:
            rank = int(rank_str)
        except ValueError:
            continue
        proposals.append(Proposal(id=pid.strip(), rank=rank, title=title.strip()))
    if not proposals:
        return None
    return ProposalIndex(
        version="1",
        total_count=len(proposals),
        groups=[ProposalGroup(phase=1, label="Recovered", proposals=proposals)],
        warnings=["parsed-from-ranking-table-only"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_markdown(workspace: Path) -> Path | None:
    """Locate the aggregator's markdown output."""
    for name in ("unified_plan.md", "output.md", "final_result.md"):
        for subdir in ("outputs", "."):  # Part 2: final_deliverables/ retired
            candidate = workspace / subdir / name
            if candidate.is_file():
                return candidate
    return None


def make_empty_index(
    source_workspace: str = "",
    warnings: list[str] | None = None,
) -> ProposalIndex:
    """Create an empty index with metadata (used when extraction fails)."""
    return ProposalIndex(
        version="1",
        created_at=datetime.now(timezone.utc).isoformat(),
        source_workspace=source_workspace,
        total_count=0,
        warnings=warnings or [],
    )
