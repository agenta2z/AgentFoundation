/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * useHypothesisImplementations — fetches the per-hub hypothesis
 * implementation status map. Mirrors the server's Apply-Combos preflight
 * gate so the UI's disabled state and the server's 422 always agree.
 *
 * Ported from RankEvolve (hooks/useHypothesisImplementations.js). Takes
 * `sessionId` + `multiTaskId` (hubId) as explicit args; session/hub-scoped
 * GET → direct REST. Refetches on `combo_overrides_changed` and
 * `submission_state` window events.
 *
 * Returns: { implementations, loading, error, refetch }
 */

import { useEffect, useState, useCallback } from 'react';

const EMPTY = { implementations: {}, error: null };

export function useHypothesisImplementations(sessionId, multiTaskId) {
  const [state, setState] = useState(EMPTY);
  const [loading, setLoading] = useState(false);

  const refetch = useCallback(async () => {
    if (!sessionId || !multiTaskId) {
      setState(EMPTY);
      return;
    }
    setLoading(true);
    try {
      const url = `/api/hubs/${encodeURIComponent(multiTaskId)}/hypothesis_implementations`
        + `?session_id=${encodeURIComponent(sessionId)}`;
      const res = await fetch(url);
      if (!res.ok) {
        const errMsg = `hypothesis_implementations fetch returned HTTP ${res.status}`;
        console.warn('[useHypothesisImplementations]', errMsg, url);
        setState({ implementations: {}, error: errMsg });
        return;
      }
      const data = await res.json();
      setState({
        implementations: (data && typeof data.implementations === 'object'
          && data.implementations !== null)
          ? data.implementations
          : {},
        error: null,
      });
    } catch (e) {
      const errMsg = `hypothesis_implementations fetch failed: ${e?.message || e}`;
      console.warn('[useHypothesisImplementations]', errMsg);
      setState({ implementations: {}, error: errMsg });
    } finally {
      setLoading(false);
    }
  }, [sessionId, multiTaskId]);

  useEffect(() => { refetch(); }, [refetch]);

  useEffect(() => {
    const onChanged = (ev) => {
      const detail = ev?.detail || {};
      if (!multiTaskId || detail.multi_task_id === multiTaskId
          || detail.multi_task_id === undefined) {
        refetch();
      }
    };
    window.addEventListener('combo_overrides_changed', onChanged);
    window.addEventListener('submission_state', onChanged);
    return () => {
      window.removeEventListener('combo_overrides_changed', onChanged);
      window.removeEventListener('submission_state', onChanged);
    };
  }, [refetch, multiTaskId]);

  return { ...state, loading, refetch };
}

export default useHypothesisImplementations;
