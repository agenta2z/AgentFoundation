/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * normalizeProposalData — JS dialect normalizer for the proposal-selection
 * widget. Converts either dialect the widget may receive into a single canonical
 * hub-dialect shape, so the rich `phases` layout + `multiTaskHelpers` helpers
 * work unchanged regardless of the caller.
 *
 * The AF-native producer (`ProposalIndex.to_dict()` in
 * `common/data_models/proposal/model.py`) emits `groups` / `proposal_ids` /
 * `constraints`, while the RankEvolve/hub subsystem
 * (`experiment_hub/proposal_models.py::ProposalSelectionData.to_dict()`) emits
 * `phases` / `hypothesis_ids` / `combo_constraints`. This function is the
 * JS-side inverse of the Python `canonicalize_proposal_index_dict`
 * (`common/data_models/proposal/parser.py:68`) — where Python canonicalises
 * hub→AF for on-disk `proposals.json`, this canonicalises AF→hub for the
 * widget's in-memory rendering. Both directions preserve semantics losslessly.
 *
 * Canonical output shape:
 *   {
 *     phases: [{
 *       phase, label, description,
 *       proposals: [ { ...spread of all input fields, one_line_summary, slots } ],
 *       batches:   [ { id, label, timeline, hypothesis_ids } ],
 *     }],
 *     combo_constraints: [
 *       { id, kind, hypothesis_ids, requires_ids, requires_any_of, label, reason, severity }
 *     ],
 *     total_count,
 *     themes,
 *   }
 *
 * Rules:
 *   • Alias priority ALWAYS canonical-first (`phases ?? groups`,
 *     `hypothesis_ids ?? proposal_ids`, `combo_constraints ?? constraints`) so
 *     re-normalisation is a fixed point (idempotent).
 *   • Proposal fields survive by SPREAD `{...p}` (never whitelist) — hub-only
 *     fields like `includes`, `probability`, `theme`, `source_workers`, `notes`,
 *     `_overrideMeta`, `deprioritized` all flow through untouched.
 *   • `slots` exposed defensively: AF `Proposal` keeps them in `metadata`
 *     (per `model.py:4-6` docstring — "Domain-specific fields (probability,
 *     slots, batches) go in `Proposal.metadata` or in subclasses.").
 *   • IDs never rewritten (no P#→H#) — user-visible ids stay consistent across
 *     surfaces; the hub treats ids as opaque strings.
 *   • Malformed / null / empty input → empty canonical shape (safe render).
 *   • Options-only input (no proposals structure) → synthesised single phase
 *     from `{label, value, description}` triples so any legacy caller with just
 *     a bare option list still renders a card list.
 */

/**
 * @typedef {Object} NormalizedBatch
 * @property {string} id
 * @property {string} label
 * @property {string} [timeline]
 * @property {string[]} hypothesis_ids
 */
/**
 * @typedef {Object} NormalizedConstraint
 * @property {string} id
 * @property {string} kind
 * @property {string[]} hypothesis_ids
 * @property {string[]} [requires_ids]
 * @property {boolean} [requires_any_of]
 * @property {string} [label]
 * @property {string} [reason]
 * @property {string} [severity]
 */
/**
 * @typedef {Object} NormalizedPhase
 * @property {number} [phase]
 * @property {string} label
 * @property {string} [description]
 * @property {Object[]} proposals
 * @property {NormalizedBatch[]} batches
 */
/**
 * @typedef {Object} NormalizedProposals
 * @property {NormalizedPhase[]} phases
 * @property {NormalizedConstraint[]} combo_constraints
 * @property {number} total_count
 * @property {string[]} themes
 */

const EMPTY = Object.freeze({
  phases: [],
  combo_constraints: [],
  total_count: 0,
  themes: [],
});

/**
 * Normalize a raw proposals payload into the hub-canonical shape the widget
 * renders. Pure function; safe on null/undefined/malformed input.
 *
 * @param {Object|null|undefined} rawProposals
 * @param {Object[]|null|undefined} [options] — options-only fallback list
 * @returns {NormalizedProposals}
 */
export function normalizeProposalData(rawProposals, options) {
  if (rawProposals && typeof rawProposals === 'object' && !Array.isArray(rawProposals)) {
    const groupsSrc = _prefer(rawProposals.phases, rawProposals.groups);
    const groups = Array.isArray(groupsSrc) ? groupsSrc : [];
    const phases = groups.map(_normalizePhase).filter(Boolean);

    const constraintsSrc = _prefer(
      rawProposals.combo_constraints,
      rawProposals.constraints,
    );
    const constraints = Array.isArray(constraintsSrc) ? constraintsSrc : [];
    const combo_constraints = constraints.map(_normalizeConstraint).filter(Boolean);

    const total_count = Number.isFinite(rawProposals.total_count)
      ? rawProposals.total_count
      : phases.reduce((s, ph) => s + (ph.proposals || []).length, 0);

    const themes = Array.isArray(rawProposals.themes) ? rawProposals.themes : [];

    if (phases.length > 0 || combo_constraints.length > 0 || total_count > 0) {
      return { phases, combo_constraints, total_count, themes };
    }
  }

  // Options-only fallback: synthesize a single phase from an option list so the
  // rich renderer always has something to walk (replaces the deleted
  // `_optionsToProposals` path from the widget).
  if (Array.isArray(options) && options.length > 0) {
    const synth = _phaseFromOptions(options);
    if (synth) {
      return {
        phases: [synth],
        combo_constraints: [],
        total_count: synth.proposals.length,
        themes: [],
      };
    }
  }

  return { ...EMPTY };
}

/** Canonical-first `a ?? b` — but null-safe (avoids `?? []` promoting `undefined`). */
function _prefer(a, b) {
  if (a !== null && a !== undefined) return a;
  return b;
}

function _normalizePhase(g) {
  if (!g || typeof g !== 'object') return null;
  const proposals = Array.isArray(g.proposals)
    ? g.proposals.map(_normalizeProposal).filter(Boolean)
    : [];
  const batches = Array.isArray(g.batches)
    ? g.batches.map(_normalizeBatch).filter(Boolean)
    : [];
  return {
    phase: g.phase,
    label: g.label || (g.phase != null ? `Phase ${g.phase}` : ''),
    description: g.description || '',
    proposals,
    batches,
  };
}

function _normalizeProposal(p) {
  if (!p || typeof p !== 'object' || p.id == null) return null;
  // Spread first so hub-only fields (includes, probability, theme,
  // source_workers, notes, _overrideMeta, deprioritized, cross_refs,
  // dependencies) all survive unchanged. Then override for id/summary/slots.
  const out = { ...p, id: String(p.id) };
  if (!out.one_line_summary && out.summary) {
    out.one_line_summary = out.summary;
  }
  // AF Proposal keeps slots in metadata (per model.py:4-6). Expose defensively
  // at top level so the widget's slot chip + computeBlockedIds see it either
  // way. Never present in this project's AF data today → empty array → ME is a
  // data-driven no-op.
  const slots = p.slots ?? p.metadata?.slots ?? [];
  out.slots = Array.isArray(slots) ? slots : [];
  return out;
}

function _normalizeBatch(b) {
  if (!b || typeof b !== 'object') return null;
  const idsSrc = _prefer(b.hypothesis_ids, b.proposal_ids);
  return {
    id: b.id,
    label: b.label || '',
    timeline: b.timeline || '',
    hypothesis_ids: Array.isArray(idsSrc) ? idsSrc.map(String) : [],
  };
}

function _normalizeConstraint(c) {
  if (!c || typeof c !== 'object') return null;
  const idsSrc = _prefer(c.hypothesis_ids, c.proposal_ids);
  const out = {
    id: c.id,
    kind: c.kind,
    hypothesis_ids: Array.isArray(idsSrc) ? idsSrc.map(String) : [],
  };
  if (Array.isArray(c.requires_ids)) out.requires_ids = c.requires_ids.map(String);
  if (c.requires_any_of != null) out.requires_any_of = !!c.requires_any_of;
  if (c.label) out.label = c.label;
  if (c.reason) out.reason = c.reason;
  if (c.severity) out.severity = c.severity;
  return out;
}

function _phaseFromOptions(options) {
  const proposals = [];
  for (const o of options) {
    if (!o) continue;
    const id = o.value != null ? String(o.value) : o.label != null ? String(o.label) : null;
    if (id === null || id === '') continue;
    proposals.push({
      id,
      title: o.label || id,
      summary: o.description || '',
      one_line_summary: o.description || o.label || id,
      slots: [],
    });
  }
  if (proposals.length === 0) return null;
  return {
    phase: undefined,
    label: '',
    description: '',
    proposals,
    batches: [],
  };
}

export default normalizeProposalData;
