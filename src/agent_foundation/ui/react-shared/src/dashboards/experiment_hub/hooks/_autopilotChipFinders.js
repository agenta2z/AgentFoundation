/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure-function chip-finding helpers used by useAutopilot. Operate purely on
 * the reducer `tasks`/runQueue shape — no React deps.
 *
 * Ported verbatim from RankEvolve (hooks/_autopilotChipFinders.js).
 */

/**
 * Find the implhyp-* outer chip whose creation timestamp is at or after the
 * launch's `launchedAt`. Scans the bound Hub's runQueue first (Phase B1
 * nesting), then falls back to the legacy top-level scan.
 */
export function findImplChipByLaunch(tasks, launchInfo) {
  if (!launchInfo) return null;
  const cutoff = launchInfo.launchedAt - 5_000;  // 5s grace for clock skew

  // Path 1 — nested under the Hub: scan the Hub's runQueue for an implhyp
  // wrapper entry.
  if (launchInfo.multiTaskId) {
    const hubTask = (tasks || {})[launchInfo.multiTaskId];
    if (hubTask && Array.isArray(hubTask.runQueue)) {
      for (const entry of hubTask.runQueue) {
        if (!entry || typeof entry.subTaskId !== 'string') continue;
        if (!entry.subTaskId.startsWith('implhyp-')) continue;
        const meta = entry.metadata || {};
        if (meta.implhyp_kind && meta.implhyp_kind !== 'wrapper') continue;
        if (meta.hub_id && meta.hub_id !== launchInfo.multiTaskId) continue;
        return {
          id: entry.subTaskId,
          status: entry.status,
          metadata: meta,
          workspacePath: entry.workspacePath,
          label: entry.label,
        };
      }
    }
  }

  // Path 2 — legacy top-level scan (standalone path; no hub_id).
  for (const t of Object.values(tasks || {})) {
    if (!t || typeof t.id !== 'string' || !t.id.startsWith('implhyp-')) continue;
    const meta = t.metadata || {};
    if (meta.tool_name !== 'implement_hypothesis') continue;
    if (launchInfo.multiTaskId && meta.hub_id !== launchInfo.multiTaskId) continue;
    const tsStr = t.createdAt || t.startedAt || t.startingAt;
    const ts = tsStr ? Date.parse(tsStr) : 0;
    if (ts && ts < cutoff) continue;
    return t;
  }
  return null;
}

/**
 * Find the agg_<ts>_<id> aggregator-only-refresh chip for `multiTaskId`
 * created at or after `sinceMs`.
 */
export function findRefreshChipForHub(tasks, multiTaskId, sinceMs) {
  if (!multiTaskId) return null;
  for (const t of Object.values(tasks || {})) {
    if (!t || typeof t.id !== 'string' || !t.id.startsWith('agg_')) continue;
    const meta = t.metadata || {};
    if (meta.multi_task_id && meta.multi_task_id !== multiTaskId) continue;
    const tsStr = t.createdAt || t.startedAt;
    const ts = tsStr ? Date.parse(tsStr) : 0;
    if (sinceMs && ts && ts < sinceMs - 5_000) continue;
    return t;
  }
  return null;
}
