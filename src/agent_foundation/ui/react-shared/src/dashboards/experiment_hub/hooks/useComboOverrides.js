/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * useComboOverrides — fetches the per-hub combos sidecar and re-fetches when
 * the server broadcasts `combo_overrides_changed`.
 *
 * Ported from RankEvolve (hooks/useComboOverrides.js). Takes `sessionId` +
 * `multiTaskId` (hubId) as explicit args. Session/hub-scoped GET → talks to
 * the OpenTeam REST route directly.
 *
 * Returns: { activeCombos, log, generatedAt, flagMapWarnings, loading, refetch }
 */

import { useEffect, useState, useCallback } from 'react';

const EMPTY = { activeCombos: [], log: [], generatedAt: null, flagMapWarnings: [] };

export function useComboOverrides(sessionId, multiTaskId) {
  const [state, setState] = useState(EMPTY);
  const [loading, setLoading] = useState(false);

  const refetch = useCallback(async () => {
    if (!sessionId || !multiTaskId) {
      setState(EMPTY);
      return;
    }
    setLoading(true);
    try {
      const url = `/api/sessions/${encodeURIComponent(sessionId)}`
        + `/combo_overrides/${encodeURIComponent(multiTaskId)}`;
      const res = await fetch(url);
      if (!res.ok) {
        setState(EMPTY);
        return;
      }
      const data = await res.json();
      setState({
        activeCombos: Array.isArray(data?.active_combos) ? data.active_combos : [],
        log: Array.isArray(data?.applied_changes_log) ? data.applied_changes_log : [],
        generatedAt: data?.generatedAt || null,
        flagMapWarnings: Array.isArray(data?._hub_flag_map_legacy_warning)
          ? data._hub_flag_map_legacy_warning
          : [],
      });
    } catch (e) {
      console.warn('[useComboOverrides] fetch failed', e);
      setState(EMPTY);
    } finally {
      setLoading(false);
    }
  }, [sessionId, multiTaskId]);

  useEffect(() => { refetch(); }, [refetch]);

  useEffect(() => {
    const onChanged = (ev) => {
      const detail = ev?.detail || {};
      if (!multiTaskId || detail.multi_task_id === multiTaskId) {
        refetch();
      }
    };
    window.addEventListener('combo_overrides_changed', onChanged);
    return () => window.removeEventListener('combo_overrides_changed', onChanged);
  }, [refetch, multiTaskId]);

  return { ...state, loading, refetch };
}

export default useComboOverrides;
