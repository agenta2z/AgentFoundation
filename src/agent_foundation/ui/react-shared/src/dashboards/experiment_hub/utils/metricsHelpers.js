/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * metricsHelpers — UI-side helpers for reading per-epoch / final metrics
 * from a hub submission row, robust to schema drift between `finalMetrics`
 * (snake-case `ndcg_10`) and `epochTrajectory[i]` (no-underscore `ndcg10`),
 * and to the `result.metrics` zero-stub for combos lacking a backing CSV.
 *
 * Ported verbatim from RankEvolve (utils/metricsHelpers.js).
 * Pure functions — no React/MUI imports; tree-shakeable.
 */

const _METRIC_KEY_ALIASES = {
  ndcg_10: ['ndcg_10', 'ndcg10'],
  hr_10: ['hr_10', 'hr10'],
  mrr: ['mrr'],
};

/**
 * Read a metric from a row, trying both snake-case and no-underscore keys.
 */
export function pickMetric(row, name) {
  if (!row) return undefined;
  const aliases = _METRIC_KEY_ALIASES[name] || [name];
  for (const k of aliases) {
    if (row[k] != null) return row[k];
  }
  return undefined;
}

/**
 * True when `rows` is empty OR is the synth zero-stub
 * (`length === 1 && epoch === 0 && all metrics are 0`).
 */
export function isMetricsStub(rows) {
  if (!rows || rows.length === 0) return true;
  if (rows.length > 1) return false;
  const r = rows[0];
  const epoch = Number(r?.epoch ?? 0);
  const allZero = ['ndcg_10', 'hr_10', 'mrr'].every(
    k => Number(pickMetric(r, k) ?? 0) === 0
  );
  return epoch === 0 && allZero;
}

/**
 * Sparse-sample an epochTrajectory into ≤7 rows for tabular rendering.
 */
export function sampleTrajectoryForTable(traj, comboKey, status) {
  if (!traj || traj.length === 0) return [];
  const n = traj.length;
  let bestIdx = 0;
  let bestNdcg = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = Number(pickMetric(traj[i], 'ndcg_10') ?? -Infinity);
    if (v > bestNdcg) {
      bestNdcg = v;
      bestIdx = i;
    }
  }
  const indices = new Set([
    0,
    Math.min(4, n - 1),
    Math.floor(n / 4),
    Math.floor(n / 2),
    Math.floor((3 * n) / 4),
    n - 1,
    bestIdx,
  ]);
  return [...indices]
    .filter(i => i >= 0 && i < n)
    .sort((a, b) => a - b)
    .map(i => ({
      experiment: comboKey || 'combo',
      epoch: Number(traj[i].epoch ?? 0),
      ndcg_10: Number(pickMetric(traj[i], 'ndcg_10') ?? 0),
      hr_10: Number(pickMetric(traj[i], 'hr_10') ?? 0),
      mrr: Number(pickMetric(traj[i], 'mrr') ?? 0),
      status: status || 'completed',
    }));
}
