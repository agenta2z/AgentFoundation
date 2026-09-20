/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * useProposalOverrides — fetches the per-session proposal_overrides.json
 * sidecar and re-fetches on the `proposal_overrides_applied` window event.
 *
 * Ported from RankEvolve (hooks/useProposalOverrides.js). Takes `sessionId`
 * as an explicit arg (no SessionContext). The sidecar is a session-scoped
 * GET, so it talks to the OpenTeam REST route directly rather than through
 * the injected HubApiClient (which carries only mutating hub actions).
 */

import { useEffect, useState, useCallback } from 'react';

const EMPTY = { rankings: [], deprioritize: [], applied_changes_log: [] };

export function useProposalOverrides(sessionId) {
  const [overrides, setOverrides] = useState(EMPTY);
  const [loading, setLoading] = useState(false);

  const refetch = useCallback(async () => {
    if (!sessionId) {
      setOverrides(EMPTY);
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(`/api/sessions/${sessionId}/proposal_overrides`);
      if (!res.ok) {
        setOverrides(EMPTY);
        return;
      }
      const data = await res.json();
      setOverrides({
        rankings: data?.rankings || [],
        deprioritize: data?.deprioritize || [],
        applied_changes_log: data?.applied_changes_log || [],
      });
    } catch (e) {
      console.warn('[useProposalOverrides] fetch failed', e);
      setOverrides(EMPTY);
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    refetch();
  }, [refetch]);

  useEffect(() => {
    const onApplied = () => refetch();
    window.addEventListener('proposal_overrides_applied', onApplied);
    return () => window.removeEventListener('proposal_overrides_applied', onApplied);
  }, [refetch]);

  return { overrides, loading, refetch };
}

export default useProposalOverrides;
