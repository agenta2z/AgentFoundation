/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * useAggregationSettings — localStorage-backed, per-session settings for the
 * LLM aggregator (Refresh Learnings → Aggregation settings…).
 *
 * Ported verbatim from RankEvolve (hooks/useAggregationSettings.js).
 */

import { useCallback, useState } from 'react';

const STORAGE_KEY = (sid) => `rankevolve_agg_settings_${sid}`;

const DEFAULTS = Object.freeze({
  minEpochs: 10,
  includeIncomparable: false,
  includeErrored: true,
  forceRefresh: false, // NEVER loaded from storage; always default
});

export function defaultAggregationSettings() {
  return { ...DEFAULTS };
}

export default function useAggregationSettings(sessionId) {
  const [settings, setSettings] = useState(() => {
    if (!sessionId) return { ...DEFAULTS };
    try {
      const raw = localStorage.getItem(STORAGE_KEY(sessionId));
      if (!raw) return { ...DEFAULTS };
      const parsed = JSON.parse(raw);
      return { ...DEFAULTS, ...parsed, forceRefresh: false };
    } catch {
      return { ...DEFAULTS };
    }
  });

  const save = useCallback((next) => {
    setSettings(next);
    if (!sessionId) return;
    try {
      const { forceRefresh: _drop, ...persisted } = next;
      localStorage.setItem(STORAGE_KEY(sessionId), JSON.stringify(persisted));
    } catch {
      // localStorage may be disabled / quota-exceeded; fail silently.
    }
  }, [sessionId]);

  return [settings, save];
}
