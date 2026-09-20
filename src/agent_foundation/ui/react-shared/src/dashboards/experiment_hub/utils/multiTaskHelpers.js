/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * multiTaskHelpers — pure combo / status helpers extracted from RankEvolve's
 * `useSessionManager.js` (getComboKey / findMatchingSubmissions /
 * computeBlockedIds / checkComboConstraints / getMultiTaskStatus). They
 * operate purely on the hub dashboard-state shape, with no SessionContext
 * dependency.
 *
 * NOTE: `deriveStatusMap` lives in the generic dashboard framework
 * (`../../../dashboard/lifecycle`) and is re-exported here for convenience so
 * the ported views keep a single import site.
 */

import { deriveStatusMap } from '../../../dashboard';

export { deriveStatusMap };

/** Compute combo key from selected item IDs (sorted for grouping). */
export function getComboKey(itemIds) {
  return [...itemIds].sort().join(',');
}

/** Find submissions matching the current selection (for combo hints). */
export function findMatchingSubmissions(selectedIds, submissions) {
  const currentKey = getComboKey(selectedIds);
  const exact = (submissions || []).filter(s => s.comboKey === currentKey);
  const currentSet = new Set(selectedIds);
  const similar = (submissions || []).filter(s => {
    if (s.comboKey === currentKey) return false;
    const sSet = new Set(s.selectedItems);
    const overlap = [...currentSet].filter(id => sSet.has(id)).length;
    const diff = Math.max(currentSet.size, sSet.size) - overlap;
    return diff <= 2 && overlap >= 1;
  });
  return { exact, similar };
}

/**
 * Slot-based combo-conflict detection. Used by Review&Combo to disable
 * hypothesis checkboxes that conflict with the current selection.
 * Returns { blocked: Map<hid, reason>, includedBy: Map<hid, parentComboId> }.
 */
export function computeBlockedIds(selectedIds, allProposals) {
  const occupied = new Map();   // slot -> first H id occupying it
  const includedBy = new Map(); // hid -> parent combo id

  // Pass 1: collect slot occupancy from explicitly-selected H's
  for (const p of (allProposals || [])) {
    if (!selectedIds.has(p.id)) continue;
    (p.includes || []).forEach(hid => includedBy.set(hid, p.id));
    (p.slots || []).forEach(s => {
      if (!occupied.has(s)) occupied.set(s, p.id);
    });
  }
  // Pass 2: included-by H's also occupy their slots (transitive)
  for (const p of (allProposals || [])) {
    if (!includedBy.has(p.id)) continue;
    (p.slots || []).forEach(s => {
      if (!occupied.has(s)) occupied.set(s, includedBy.get(p.id));
    });
  }
  // Pass 3: every NON-selected, NON-included H gets blocked if any of its
  // slots is occupied
  const blocked = new Map();
  for (const p of (allProposals || [])) {
    if (selectedIds.has(p.id) || includedBy.has(p.id)) continue;
    for (const s of (p.slots || [])) {
      if (occupied.has(s) && occupied.get(s) !== p.id) {
        blocked.set(p.id, `Conflicts with ${occupied.get(s)} (slot: ${s}). Uncheck ${occupied.get(s)} to enable.`);
        break;
      }
    }
  }
  return { blocked, includedBy };
}

/**
 * Typed-constraint checker for top-level combo_constraints.
 *
 * Returns { errors: [], warnings: [], infos: [] } partitioned by severity.
 * Supported kinds:
 *   • ``requires`` / ``recommends`` — routed by severity (default: errors for
 *     requires-with-missing; always infos for recommends).
 *   • ``mutually_exclusive`` — routed by severity (default errors).
 *   • ``conflicts`` — **always advisory** (routed to ``infos`` regardless of
 *     ``severity``), because the proposal-selection widget is display-only for
 *     conflicts — conflict resolution happens later at task-execution time
 *     (sequential integration ordering). The original ``severity`` is carried
 *     on the entry so consumers can style the badge (error=red / warning=amber
 *     / info=grey) without changing the enforcement path.
 */
export function checkComboConstraints(selectedIds, comboConstraints) {
  const violations = { errors: [], warnings: [], infos: [] };
  const sel = selectedIds instanceof Set ? selectedIds : new Set(selectedIds || []);
  for (const c of (comboConstraints || [])) {
    const triggered = (c.hypothesis_ids || []).some(h => sel.has(h));
    if (!triggered) continue;
    const bucket = c.severity === 'warning' ? 'warnings'
                 : c.severity === 'info' ? 'infos'
                 : 'errors';
    if (c.kind === 'requires') {
      const reqs = c.requires_ids || [];
      const ok = c.requires_any_of
        ? reqs.some(r => sel.has(r))
        : reqs.every(r => sel.has(r));
      if (!ok) {
        violations[bucket].push({
          constraint: c,
          missing: reqs.filter(r => !sel.has(r)),
        });
      }
    } else if (c.kind === 'recommends') {
      const reqs = c.requires_ids || [];
      const ok = reqs.length === 0 || reqs.some(r => sel.has(r));
      if (!ok) violations.infos.push({ constraint: c, hint: c.label || c.reason || '' });
    } else if (c.kind === 'mutually_exclusive') {
      const set = new Set(c.hypothesis_ids || []);
      const present = [...sel].filter(id => set.has(id));
      if (present.length > 1) {
        violations[bucket].push({ constraint: c, conflicting: present });
      }
    } else if (c.kind === 'conflicts') {
      // Advisory only — never blocks submit regardless of severity. The
      // original severity is preserved on `constraint.severity` for badge
      // styling; the routing bucket is *always* `infos` so downstream
      // `submitDisabled = violations.errors.length > 0` logic is unaffected.
      const set = new Set(c.hypothesis_ids || []);
      const present = [...sel].filter(id => set.has(id));
      if (present.length >= 2) {
        violations.infos.push({
          constraint: c,
          conflicting: present,
          hint: c.label || c.reason || '',
        });
      }
    }
  }
  return violations;
}

/**
 * Compute aggregate status for a hub (for the dashboard status chip).
 * Returns { label, color, icon }.
 */
export function getMultiTaskStatus(task) {
  const queue = task.runQueue || [];
  const running = queue.filter(e => e.status === 'running');
  const done = queue.filter(e => e.status === 'completed' || e.status === 'error');

  const subs = task.submissions || [];
  const activeSub = subs.find(e => e.status === 'running' || e.status === 'submitted');
  const allSubsDone = subs.length > 0 && subs.every(e => e.status === 'completed' || e.status === 'error');
  const anySubDone = subs.some(e => e.status === 'completed');

  if (allSubsDone) return { label: 'Experiments Done', color: 'success', icon: '✅' };
  if (activeSub) return { label: 'Experiment Running', color: 'info', icon: '▶' };
  if (anySubDone) return { label: 'Experiment Done', color: 'success', icon: '✅' };
  if (done.length === queue.length && queue.length > 0) return { label: 'Review Ready', color: 'warning', icon: '🔬' };
  if (running.length > 0) return { label: `${done.length}/${queue.length}`, color: 'info', icon: '▶' };
  return { label: 'Queued', color: 'default', icon: '⏸' };
}
