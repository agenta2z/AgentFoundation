/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * useAutopilot — client-side state machine that drives Auto Mode.
 *
 * Phases:
 *   idle | awaiting_implement | awaiting_refresh | applying_combos
 *   | completed | stopped | error
 *
 * Ported from RankEvolve (hooks/useAutopilot.js). The one hard SessionContext
 * entanglement: it watched the reducer's `tasks` map for the implhyp / agg
 * chip transitions. Here, `tasks` is INJECTED as an explicit arg (the hub
 * dashboard reducer tracks per-hub chips inside its own `tasks` slice — see
 * reducer.js). `runImplementHypothesis` / `runExperimentCombos` are the
 * injected apiClient actions. The autopilot's server mirror + learnings fetch
 * + combo-apply are session-scoped REST calls (kept as direct fetch, matching
 * the read-hooks convention).
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { isComboApplyEligible } from '../utils/comboPendingPredicate';
import {
  findImplChipByLaunch,
  findRefreshChipForHub,
} from './_autopilotChipFinders';
import { useHypothesisImplementations } from './useHypothesisImplementations';

export { findImplChipByLaunch as _findImplChipByLaunch };
export { findRefreshChipForHub as _findRefreshChipForHub };

export const DEFAULT_AUTOPILOT_SETTINGS = Object.freeze({
  loopMode: false,
  maxIterations: 10,
  maxTotalExperiments: 50,
  maxWallTimeHours: 24,
  stopOnFailure: true,
  topKCombos: 5,
});

const INITIAL_STATE = Object.freeze({
  phase: 'idle',
  iteration: 0,
  totalImplemented: 0,
  startedAt: null,
  launchInfo: null,
  implTaskId: null,
  refreshTaskId: null,
  appliedComboIds: [],
  blockedCombos: [],
  error: null,
});

function _safePost(url, body) {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).catch((e) => {
    console.warn('[useAutopilot] sidecar mirror POST failed:', e);
    return null;
  });
}

export function useAutopilot({
  sessionId,
  multiTaskId,            // hub mid; required for refresh + apply combos
  autoMode,               // boolean
  settings,
  runImplementHypothesis, // injected apiClient action
  runExperimentCombos,    // injected apiClient action
  tasks,                  // INJECTED reducer tasks map (was useSession().tasks)
}) {
  const tasksMap = tasks || {};
  const { implementations } = useHypothesisImplementations(sessionId, multiTaskId);
  const [state, setState] = useState(INITIAL_STATE);
  const [newCombosFromLearnings, setNewCombosFromLearnings] = useState(null);
  const learningsFetchInFlight = useRef(false);

  const settingsRef = useRef(settings);
  useEffect(() => { settingsRef.current = settings; }, [settings]);

  // Mirror state to server (best-effort, write-through).
  useEffect(() => {
    if (!sessionId) return;
    _safePost(
      `/api/sessions/${encodeURIComponent(sessionId)}/_hub/autopilot/state`,
      { multi_task_id: multiTaskId, state },
    );
  }, [sessionId, multiTaskId, state]);

  // Disarm on Auto Mode toggle-off.
  useEffect(() => {
    if (!autoMode && state.phase !== 'idle') {
      setState(INITIAL_STATE);
      setNewCombosFromLearnings(null);
    }
  }, [autoMode, state.phase]);

  // Phase: awaiting_implement → awaiting_refresh
  useEffect(() => {
    if (state.phase !== 'awaiting_implement') return;
    if (!state.launchInfo) return;
    const chip = findImplChipByLaunch(tasksMap, state.launchInfo);
    if (!chip) return;
    if (chip.status === 'completed') {
      try {
        runExperimentCombos(multiTaskId, { aggregateOnly: true });
      } catch (e) {
        setState((s) => ({ ...s, phase: 'error', error: `refresh_dispatch_failed: ${e.message}` }));
        return;
      }
      const nImpl = state.launchInfo.selectedIds?.length || 0;
      setState((s) => ({
        ...s,
        phase: 'awaiting_refresh',
        implTaskId: chip.id,
        totalImplemented: s.totalImplemented + nImpl,
      }));
    } else if (chip.status === 'error') {
      setState((s) => ({ ...s, phase: 'error', error: 'implementation_failed' }));
    }
  }, [state.phase, state.launchInfo, tasksMap, multiTaskId, runExperimentCombos]);

  // Phase: awaiting_refresh → applying_combos
  useEffect(() => {
    if (state.phase !== 'awaiting_refresh') return;
    const refreshChip = findRefreshChipForHub(tasksMap, multiTaskId, state.startedAt || 0);
    if (!refreshChip) return;
    if (refreshChip.status === 'completed') {
      if (newCombosFromLearnings === null && !learningsFetchInFlight.current) {
        learningsFetchInFlight.current = true;
        fetch(`/api/sessions/${encodeURIComponent(sessionId)}/learnings`)
          .then((r) => (r.ok ? r.json() : null))
          .then((body) => {
            const combos = body?.actions?.newCombos
              || body?.learnings_actions?.newCombos
              || [];
            setNewCombosFromLearnings(Array.isArray(combos) ? combos : []);
          })
          .catch((e) => {
            console.warn('[useAutopilot] learnings fetch failed:', e);
            setNewCombosFromLearnings([]);
          })
          .finally(() => { learningsFetchInFlight.current = false; });
        return;  // wait for fetch; effect re-runs on state change
      }
      if (newCombosFromLearnings === null) return;  // still fetching
      const allCombos = newCombosFromLearnings || [];
      const eligible = [];
      const blocked = [];
      for (const c of allCombos) {
        const r = isComboApplyEligible(c, implementations);
        if (r.eligible) eligible.push(c);
        else blocked.push({ comboId: c.comboId, gating: r.gatingHypotheses });
      }
      const cappedEligible = eligible.slice(0, settingsRef.current.topKCombos);
      if (cappedEligible.length === 0) {
        setState((s) => ({
          ...s,
          phase: 'completed',
          error: 'no_eligible_combos',
          blockedCombos: blocked,
          refreshTaskId: refreshChip.id,
        }));
        return;
      }
      setState((s) => ({
        ...s,
        phase: 'applying_combos',
        refreshTaskId: refreshChip.id,
        blockedCombos: blocked,
      }));
      fetch(
        `/api/sessions/${encodeURIComponent(sessionId)}`
        + `/combo_overrides/${encodeURIComponent(multiTaskId)}/apply`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ combos: cappedEligible, applied_by: 'auto_mode' }),
        },
      ).then(async (res) => {
        if (!res.ok) {
          let msg = `apply_combos_failed (HTTP ${res.status})`;
          try {
            const body = await res.json();
            if (body?.detail) msg = JSON.stringify(body.detail).slice(0, 240);
          } catch { /* keep generic */ }
          setState((s) => ({ ...s, phase: 'error', error: msg }));
          return;
        }
        const appliedComboIds = cappedEligible.map((c) => c.comboId).filter(Boolean);
        setState((s) => ({ ...s, appliedComboIds }));
      }).catch((e) => {
        setState((s) => ({ ...s, phase: 'error', error: `apply_combos_failed: ${e.message}` }));
      });
    } else if (refreshChip.status === 'error') {
      setState((s) => ({ ...s, phase: 'error', error: 'refresh_failed' }));
    }
  }, [state.phase, state.startedAt, tasksMap, multiTaskId, sessionId, newCombosFromLearnings, implementations]);

  // Phase: applying_combos (after appliedComboIds set) → completed OR re-arm
  useEffect(() => {
    if (state.phase !== 'applying_combos') return;
    if (!state.appliedComboIds || state.appliedComboIds.length === 0) return;
    const s = settingsRef.current;
    if (!s.loopMode) {
      setState((cur) => ({ ...cur, phase: 'completed' }));
      return;
    }
    if (state.iteration + 1 >= s.maxIterations) {
      setState((cur) => ({ ...cur, phase: 'completed', error: 'max_iterations_reached' }));
      return;
    }
    if (state.totalImplemented >= s.maxTotalExperiments) {
      setState((cur) => ({ ...cur, phase: 'completed', error: 'max_total_experiments_reached' }));
      return;
    }
    const wallElapsedMs = Date.now() - (state.startedAt || Date.now());
    const wallCapMs = s.maxWallTimeHours * 3600_000;
    if (wallElapsedMs >= wallCapMs) {
      setState((cur) => ({ ...cur, phase: 'completed', error: 'max_wall_time_exceeded' }));
      return;
    }
    const appliedCombos = (newCombosFromLearnings || []).filter(
      (c) => state.appliedComboIds.includes(c.comboId),
    );
    const nextSel = [...new Set(
      appliedCombos.flatMap((c) => c.selectedItems || []),
    )].filter((h) => !implementations[h]?.implemented);
    if (nextSel.length === 0) {
      setState((cur) => ({ ...cur, phase: 'completed', error: 'no_new_to_implement' }));
      return;
    }
    const launchInfo = runImplementHypothesis(multiTaskId, {
      selectedIds: nextSel,
      maxBatchSize: state.launchInfo?.maxBatchSize,
      maxParallel: state.launchInfo?.maxParallel,
    });
    if (!launchInfo) {
      setState((cur) => ({ ...cur, phase: 'error', error: 'launch_failed' }));
      return;
    }
    setState((cur) => ({
      ...INITIAL_STATE,
      phase: 'awaiting_implement',
      iteration: cur.iteration + 1,
      totalImplemented: cur.totalImplemented,
      startedAt: cur.startedAt,
      launchInfo: {
        ...launchInfo,
        maxBatchSize: state.launchInfo?.maxBatchSize,
        maxParallel: state.launchInfo?.maxParallel,
      },
    }));
  }, [state.phase, state.appliedComboIds, state.iteration, state.totalImplemented, state.startedAt, multiTaskId, newCombosFromLearnings, implementations, runImplementHypothesis]);

  // Public API
  const arm = useCallback((launchInfo) => {
    if (!autoMode || !launchInfo) return;
    setNewCombosFromLearnings(null);  // re-fetch on next refresh
    setState({
      ...INITIAL_STATE,
      phase: 'awaiting_implement',
      iteration: 0,
      totalImplemented: 0,
      startedAt: Date.now(),
      launchInfo,
    });
  }, [autoMode]);

  const stop = useCallback((reason = 'user_clicked_stop') => {
    setState({ ...INITIAL_STATE, phase: 'stopped', error: reason });
  }, []);

  return { state, arm, stop, blockedCombos: state.blockedCombos };
}

export default useAutopilot;
