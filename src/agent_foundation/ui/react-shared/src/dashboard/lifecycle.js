/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Dashboard lifecycle helpers — context-free reimplementations of the rules
 * RankEvolve kept in `useSessionManager` (evaluateViewRules / getPipelineStages
 * / getMultiTaskStatus / deriveStatusMap). They operate purely on a dashboard
 * state object + the (JS) ViewRegistry, with no SessionContext dependency.
 */

import { getView } from './ViewRegistry';

/** Map a run-queue-like list → { itemId: status } (e.g. for per-row badges). */
export function deriveStatusMap(runQueue) {
  const map = {};
  (runQueue || []).forEach((entry) => {
    const ids = (entry && entry.metadata
      && (entry.metadata.hypothesisIds || entry.metadata.proposalIds)) || [];
    ids.forEach((id) => { map[id] = entry.status; });
  });
  return map;
}

/**
 * Evaluate which views should be newly unlocked / auto-activated given the
 * current state. Returns ONLY newly-unlockable indices (never re-emits an
 * already-unlocked view), so a caller can dispatch without a re-render loop.
 * Auto-activate fires on the unlock transition only.
 */
export function evaluateViewRules(state, views) {
  const unlocked = new Set(state.unlockedViewIds || [0]);
  const viewsToUnlock = [];
  let viewToActivate = null;
  (views || []).forEach((v, i) => {
    const reg = getView(v.type);
    const unlockWhen = (reg && reg.unlockWhen) || (() => true);
    const autoActivateWhen = (reg && reg.autoActivateWhen) || (() => false);
    if (!unlocked.has(i) && unlockWhen(state)) {
      viewsToUnlock.push(i);
      if (autoActivateWhen(state)) viewToActivate = i;
    }
  });
  return { viewsToUnlock, viewToActivate };
}

/**
 * Per-stage pipeline statuses ('completed' | 'active' | 'pending').
 * Prefers a backend-/reducer-maintained `state.pipelineStatuses` (aligned to
 * `pipeline`); otherwise derives a sensible default from the active view.
 */
export function getPipelineStages(state, pipeline) {
  if (!pipeline || pipeline.length === 0) return null;
  if (Array.isArray(state.pipelineStatuses)
      && state.pipelineStatuses.length === pipeline.length) {
    return state.pipelineStatuses;
  }
  const active = Math.min(state.activeView || 0, pipeline.length - 1);
  return pipeline.map((_, i) => (i < active ? 'completed' : i === active ? 'active' : 'pending'));
}

/** Overall dashboard status chip ({label, color, icon}). */
export function getDashboardStatus(state) {
  if (state && state.statusInfo) return state.statusInfo;
  return { label: (state && state.status) || 'Active', color: 'default', icon: '' };
}
