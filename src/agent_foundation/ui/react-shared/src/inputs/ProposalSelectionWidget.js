/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * ProposalSelectionWidget — review and select from a ranked/grouped proposal
 * set produced by research-propose (Phase 3) for Phase 3b review.
 *
 * Consumes AgentFoundation's generic ProposalIndex shape, passed through the
 * conversation tool's input_mode metadata:
 *   config.input_mode.metadata.proposals = {
 *     total_count, groups: [{ phase, label, description,
 *       proposals: [{ id, rank, title, summary, impact, complexity, dependencies }] }] }
 * When no structured proposals are present it degrades to the flat option list
 * (config.input_mode.options), so it is always at least as capable as the
 * multiple-choice widget.
 *
 * Emits onSubmit({ selected_proposals: [<id>...], total_available: <N> }),
 * which the backend (ConversationalInferencer._handle_conversation_tool)
 * captures into the SOP output variable (e.g. selected_proposal_ids).
 */

import React, { useMemo, useState } from 'react';

function _flattenProposals(proposals) {
  // ProposalIndex.to_dict() shape → list of { id, title, ...meta, groupLabel }.
  const groups = (proposals && proposals.groups) || [];
  const flat = [];
  groups.forEach((g) => {
    const groupLabel = g.label || (g.phase != null ? `Phase ${g.phase}` : '');
    (g.proposals || []).forEach((p) => {
      if (p && p.id != null && String(p.id).length > 0) {
        flat.push({ ...p, id: String(p.id), groupLabel });
      }
    });
  });
  return flat;
}

function _optionsToProposals(options) {
  // Fallback: backend already derived one option per proposal (value=id).
  return (options || [])
    .filter((o) => o && (o.value != null || o.label != null))
    .map((o) => ({
      id: String(o.value != null ? o.value : o.label),
      title: o.label || String(o.value),
      summary: o.description || '',
      groupLabel: '',
    }));
}

export default function ProposalSelectionWidget({ config, onSubmit }) {
  const inputMode = config?.input_mode || config || {};
  const metadata = inputMode.metadata || config?.metadata || {};
  const prompt =
    inputMode.prompt || config?.prompt || config?.title || 'Select proposals to advance';
  const allowZero = metadata.allow_zero ?? false;

  const items = useMemo(() => {
    const fromProposals = _flattenProposals(metadata.proposals);
    if (fromProposals.length > 0) return fromProposals;
    return _optionsToProposals(inputMode.options || config?.options || config?.choices);
  }, [metadata.proposals, inputMode.options, config]);

  const preselected = useMemo(() => {
    const raw = metadata.preselected_ids;
    if (!raw) return new Set();
    const ids = Array.isArray(raw) ? raw : String(raw).split(',');
    return new Set(ids.map((s) => String(s).trim()).filter(Boolean));
  }, [metadata.preselected_ids]);

  const [selected, setSelected] = useState(() => new Set(preselected));

  const toggle = (id) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelected(next);
  };

  const allSelected = items.length > 0 && selected.size === items.length;
  const toggleAll = () => {
    setSelected(allSelected ? new Set() : new Set(items.map((p) => p.id)));
  };

  const handleSubmit = () => {
    // Preserve the proposal-list order rather than selection order.
    const ordered = items.filter((p) => selected.has(p.id)).map((p) => p.id);
    onSubmit({ selected_proposals: ordered, total_available: items.length });
  };

  const submitDisabled = !allowZero && selected.size === 0;

  return (
    <div className="widget widget-proposal-selection">
      {prompt && <div className="widget-prompt">{prompt}</div>}

      {items.length > 1 && (
        <label className="widget-checkbox widget-select-all">
          <input type="checkbox" checked={allSelected} onChange={toggleAll} />
          <span className="option-label">Select all ({items.length})</span>
        </label>
      )}

      <div className="widget-options widget-proposals">
        {items.map((p) => (
          <label
            key={p.id}
            className={`widget-checkbox proposal ${selected.has(p.id) ? 'checked' : ''}`}
          >
            <input
              type="checkbox"
              checked={selected.has(p.id)}
              onChange={() => toggle(p.id)}
            />
            <span className="proposal-main">
              <span className="option-label">
                {p.id}: {p.title}
              </span>
              {(p.impact || p.complexity) && (
                <span className="proposal-chips">
                  {p.impact && <span className="chip impact">{p.impact}</span>}
                  {p.complexity && (
                    <span className="chip complexity">{p.complexity}</span>
                  )}
                </span>
              )}
              {p.groupLabel && (
                <span className="proposal-group">{p.groupLabel}</span>
              )}
              {p.summary && (
                <span className="option-description">{p.summary}</span>
              )}
            </span>
          </label>
        ))}
      </div>

      <button
        className="widget-submit"
        onClick={handleSubmit}
        disabled={submitDisabled}
      >
        Advance {selected.size} proposal{selected.size === 1 ? '' : 's'}
      </button>
    </div>
  );
}
