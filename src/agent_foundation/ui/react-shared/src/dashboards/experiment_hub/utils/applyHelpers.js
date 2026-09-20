/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * applyHelpers — POST helpers for the Apply Reranks / Apply Combos / Apply
 * All flows in AccumulatedLearningsDrawer + MultiChoiceComboView.
 *
 * Ported from RankEvolve (utils/applyHelpers.js). These write per-session
 * sidecars (proposal_overrides.json / combos/current.json) — session-scoped
 * reads/writes that are NOT part of the injected HubApiClient mutation
 * surface, so they continue to talk to the OpenTeam REST routes directly.
 *
 * Both functions THROW on non-OK HTTP — callers wrap with try/catch.
 */

/**
 * POST hypothesis re-ranking overrides.
 * @throws {Error} on non-OK HTTP
 */
export async function postRerankApply(sessionId, rankings, deprioritize) {
  const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/proposal_overrides`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      rankings,
      deprioritize: deprioritize || [],
    }),
  });
  if (!res.ok) {
    throw new Error(`Apply rerank failed: HTTP ${res.status}`);
  }
  window.dispatchEvent(new CustomEvent('proposal_overrides_applied'));
  try {
    return await res.json();
  } catch (_) {
    return {};
  }
}

/**
 * POST combo overrides apply. Sends ALL provided combos (no client-side
 * eligibility filter — server tags each with `applyState`).
 * @throws {Error} on non-OK HTTP
 */
export async function postCombosApply(sessionId, multiTaskId, combos, sourceArchiveId, appliedBy) {
  const url = `/api/sessions/${encodeURIComponent(sessionId)}`
    + `/combo_overrides/${encodeURIComponent(multiTaskId)}/apply`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      combos,
      source_archive_id: sourceArchiveId || null,
      applied_by: appliedBy || 'ui',
    }),
  });
  if (!res.ok) {
    let detail = '';
    try {
      const body = await res.json();
      if (body && body.detail) {
        if (typeof body.detail === 'object' && body.detail.blocking_combos) {
          const lines = (body.detail.blocking_combos || []).map(b =>
            `${b.comboId}${b.unimplemented_hypotheses?.length
              ? ` — pending: ${b.unimplemented_hypotheses.join(', ')}`
              : ''}${b.config_not_ready ? ' — config not ready' : ''}`
          );
          detail = `Apply blocked by gate:\n${lines.join('\n')}`;
        } else {
          detail = String(body.detail);
        }
      }
    } catch (_) { /* keep generic */ }
    throw new Error(detail || `Apply combos failed: HTTP ${res.status}`);
  }
  window.dispatchEvent(new CustomEvent('combo_overrides_changed', {
    detail: { multi_task_id: multiTaskId, action: 'apply' },
  }));
  try {
    return await res.json();
  } catch (_) {
    return {};
  }
}
