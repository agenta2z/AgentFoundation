/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * SelectionView — the Experiment Hub's "Selection" tab. Wraps the generic
 * `WidgetHostView`, hosting the unified `proposal_selection` widget. Layers
 * the hub-specific concerns the generic host exposes as injection seams:
 *
 *   - config:        sourced from `dashboardState.scenarioState.selectionSnapshot`
 *                    (the canonical view config also names the widget via the
 *                    Python `view_manifest`, threaded as `config.widgets`).
 *   - enrichConfig:  incremental-submit locking — derive `_submittedIds` /
 *                    `_failedHypothesisIds` / `_implementationStatus` from the
 *                    hub's runQueue so completed Hs lock + failed Hs re-appear.
 *   - renderHeader:  batch-summary strip (hypotheses → batch queue entries).
 *   - onWidgetSubmit: route the selection through
 *                    `apiClient.runImplementHypothesis(hubId, {selectedIds,…})`
 *                    and auto-jump to the Implementation tab.
 *
 * NOTE on Auto Mode: RankEvolve's ProposalSelectionWidget embedded the
 * autopilot state machine inline (the one hard SessionContext entanglement).
 * The autopilot hook is ported (hooks/useAutopilot, with `tasks` injected) but
 * is NOT wired into this faithful first cut — the in-chat widget submit path
 * here is the single canonical launch. Auto Mode is deferred with this TODO:
 *   TODO(experiment-hub Auto Mode): surface the Auto Mode toggle + status pill
 *   by mounting useAutopilot({ sessionId, multiTaskId: hubId, autoMode,
 *   settings, runImplementHypothesis: apiClient.runImplementHypothesis,
 *   runExperimentCombos: apiClient.runExperimentCombos, tasks }) and arming it
 *   from onWidgetSubmit's returned launchInfo. The hook is decoupled and ready;
 *   it needs a `tasks`-shaped view of in-flight chips, which the hub reducer's
 *   runQueue already carries (pass `{ [hubId]: dashboardState }`).
 */

import React, { useCallback, useMemo } from 'react';
import { Box, Chip, Typography } from '@mui/material';
import { WidgetHostView } from '../../../dashboard';
import { deriveStatusMap } from '../utils/multiTaskHelpers';
import { useComboOverrides } from '../hooks/useComboOverrides';

function BatchSummaryHeader({ task }) {
  const queue = task.runQueue || [];
  // G4 — pre-submit fallback: when runQueue is empty (user hasn't clicked
  // Implement yet), preview the BNB batches from the selectionSnapshot so
  // reviewers can see what batches WILL be created before submitting.
  if (queue.length === 0) {
    const phases = task.scenarioState?.selectionSnapshot?.proposals?.phases || [];
    const allBatches = phases.flatMap(ph => (ph.batches || []).map(b => ({ ...b, phaseLabel: ph.label })));
    if (allBatches.length === 0) return null;
    const totalHyps = allBatches.reduce(
      (n, b) => n + (b.hypothesis_ids || b.proposal_ids || []).length, 0,
    );
    return (
      <Box sx={{
        p: 1.5, mb: 2,
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 1,
        backgroundColor: 'action.hover',
      }}>
        <Typography variant="body2" sx={{ fontSize: '0.82rem', mb: 0.5, color: 'text.secondary' }}>
          {totalHyps} hypotheses across {allBatches.length} batches (preview) —
          click "Implement Selected" to enqueue.
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
          {allBatches.map((b, i) => (
            <Chip
              key={i}
              label={`BNB${b.id}: ${(b.hypothesis_ids || b.proposal_ids || []).join(', ')}`}
              size="small"
              variant="outlined"
              sx={{ fontFamily: 'monospace', fontSize: '0.7rem', height: 22 }}
            />
          ))}
        </Box>
      </Box>
    );
  }
  // Post-submit: existing status-suffixed render.
  return (
    <Box sx={{
      p: 1.5, mb: 2,
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: 1,
      backgroundColor: 'rgba(255,255,255,0.02)',
    }}>
      <Typography variant="body2" sx={{ fontSize: '0.82rem', mb: 0.5 }}>
        {queue.reduce((sum, e) => sum + (e.metadata?.hypothesisIds?.length || 0), 0)} hypotheses
        selected → {queue.length} batch queue entries
      </Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
        {queue.map((entry, i) => {
          const hyps = entry.metadata?.hypothesisIds || [];
          const statusIcon = entry.status === 'completed' ? '✅'
            : entry.status === 'running' ? '▶'
              : entry.status === 'error' ? '❌' : '⏳';
          return (
            <Chip
              key={i}
              label={`B${entry.metadata?.batchId}: ${hyps.map(h => `${h}${statusIcon}`).join(', ')}`}
              size="small"
              variant="outlined"
              sx={{ fontSize: '0.7rem', height: 22 }}
            />
          );
        })}
      </Box>
    </Box>
  );
}

export default function SelectionView({
  dashboardState, config, dispatch, sessionId, hubId, apiClient,
}) {
  const task = dashboardState;

  // G3 — fetch active-combos so the widget's PreSelectedBanner has real data.
  // The hook takes (sessionId, multiTaskId) and re-fetches on the
  // combo_overrides_changed WS event automatically.
  const { activeCombos, generatedAt } = useComboOverrides(sessionId, hubId);

  // G3 — normalize activeCombos into the widget's `_activeCombos` config shape.
  // Union of every combo's `selectedItems` gives the hypothesis-ids set;
  // pendingCount = combos not yet in `ready` applyState.
  const activeCombosConfig = useMemo(() => {
    const combos = Array.isArray(activeCombos) ? activeCombos : [];
    if (combos.length === 0) return null;
    const idsUnion = new Set();
    let pending = 0;
    for (const c of combos) {
      if (c.applyState && c.applyState !== 'ready') pending += 1;
      for (const h of (c.selectedItems || [])) {
        if (h) idsUnion.add(String(h));
      }
    }
    return {
      count: combos.length,
      appliedAt: generatedAt,
      pendingCount: pending,
      hypothesisIds: Array.from(idsUnion),
    };
  }, [activeCombos, generatedAt]);

  // Incremental-submit + status enrichment (ported from RankEvolve's
  // WidgetHostView.buildWidgetConfig). Layered onto the generic host's
  // built config via enrichConfig.
  const enrichConfig = useCallback((builtConfig, spec, state) => {
    const next = { ...builtConfig };
    const runQueue = (state && state.runQueue) || [];
    if (spec && spec.incrementalSubmit) {
      const submittedIds = new Set();
      const failedIds = new Set();
      runQueue.forEach(entry => {
        const hids = entry.metadata?.hypothesisIds || [];
        if (entry.status === 'completed') hids.forEach(id => submittedIds.add(id));
        else if (entry.status === 'error') hids.forEach(id => failedIds.add(id));
      });
      next._submittedIds = submittedIds;
      next._failedHypothesisIds = failedIds;
      next._submitted = false;
      next._useDirectImplement = true;
    }
    if (spec && spec.enrichments && spec.enrichments.statusMapSource === 'runQueue') {
      next._implementationStatus = deriveStatusMap(runQueue);
    }
    next._multiTaskId = hubId || (state && state.id) || null;
    // G3 — expose active-combos to the widget's PreSelectedBanner.
    if (activeCombosConfig) {
      next._activeCombos = activeCombosConfig;
    }
    return next;
  }, [hubId, activeCombosConfig]);

  const renderHeader = useCallback((state, cfg) => {
    if (cfg?.header?.type === 'batch_summary') {
      return <BatchSummaryHeader task={state} />;
    }
    return null;
  }, []);

  const onWidgetSubmit = useCallback((spec, response) => {
    // "Confirm Selection & Start" (R2). The widget emits { selected_proposals,
    // custom_queries, total_available, is_incremental?, auto_implement? }.
    const selectedIds = (response && response.selected_proposals) || [];
    if (selectedIds.length === 0) return;

    // F3 (kept): honor an explicit `auto_implement === false` "just view" —
    // stay on Selection, no enqueue. (After R1 the chat handoff no longer sends
    // this field; the guard is retained for any host that still does.)
    if (response && response.auto_implement === false) {
      dispatch({ type: 'SET_ACTIVE_VIEW', viewIndex: 0 });
      return null;
    }

    // R2 — when the hub was opened from a LIVE conversation, the host provides
    // `confirmProposalSelection` (OpenTeam's useHubApiClient). Prefer it: ONE
    // atomic host-side command launches implementation AND resolves the still-open
    // Phase-2b pending input → the SOP advances 2b→3. The hub itself stays
    // conversation/SOP-agnostic — presence of the function is the only signal.
    if (apiClient && typeof apiClient.confirmProposalSelection === 'function') {
      const launchInfo = apiClient.confirmProposalSelection(hubId, {
        selectedIds,
        customQueries: (response && response.custom_queries) || [],
      });
      dispatch({ type: 'SET_ACTIVE_VIEW', viewIndex: 1 });
      return launchInfo;
    }

    // Fallback — standalone / no-SOP host (no conversation to advance): launch
    // implementation directly via the existing self-executing rail.
    if (apiClient && typeof apiClient.runImplementHypothesis === 'function') {
      const launchInfo = apiClient.runImplementHypothesis(hubId, {
        selectedIds,
        customQueries: (response && response.custom_queries) || [],
      });
      // Auto-jump to the Implementation tab so the user sees batches populate.
      dispatch({ type: 'SET_ACTIVE_VIEW', viewIndex: 1 });
      return launchInfo;
    }
    console.warn('[SelectionView] no confirm/runImplementHypothesis apiClient method');
    return null;
  }, [apiClient, hubId, dispatch]);

  // Build the view config the generic host consumes: prefer the Python-canonical
  // `config.widgets` (from the dashboard view_manifest), with the widget's
  // configSource pointing at scenarioState.selectionSnapshot.
  const hostConfig = useMemo(() => {
    if (config && Array.isArray(config.widgets) && config.widgets.length > 0) {
      return config;
    }
    // Front-end fallback default (matches tool.json view_manifest selection tab).
    return {
      widgets: [{
        type: 'proposal_selection',
        configSource: 'selectionSnapshot',
        readOnly: false,
        incrementalSubmit: true,
        enrichments: { statusMapSource: 'runQueue' },
      }],
      header: { type: 'batch_summary' },
    };
  }, [config]);

  return (
    <WidgetHostView
      dashboardState={task}
      config={hostConfig}
      onWidgetSubmit={onWidgetSubmit}
      enrichConfig={enrichConfig}
      renderHeader={renderHeader}
    />
  );
}
