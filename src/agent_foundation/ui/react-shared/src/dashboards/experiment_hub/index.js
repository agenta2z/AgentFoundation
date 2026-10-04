/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Experiment Hub dashboard — registration + pre-wired panel.
 *
 * Side-effect-imported by `src/index.js` so the views/dashboard register on
 * first load. The canonical tab structure (which views, order, labels,
 * per-view config) lives in Python — `resources/tools/experiment_hub/tool.json`
 * `dashboard_config.view_manifest` — and arrives at runtime in the
 * `dashboard_open` payload. The JS registry below holds only the bits that
 * cannot live in JSON (the view components + unlockWhen/autoActivateWhen
 * predicates) plus a front-end-only default manifest that mirrors the Python
 * one for dev / standalone use.
 *
 * unlockWhen rules (ported from RankEvolve's views/ViewRegistry.js):
 *   selection:      always
 *   implementation: runQueue has a non-'queued' entry (auto-activate while running)
 *   review_combo:   runQueue has a 'completed' entry
 *   experiments:    submissions > 0
 */

import React from 'react';
import {
  registerView,
  registerDashboard,
  registerManifest,
  createDashboardReducer,
  DashboardPanel,
} from '../../dashboard';
import hubDomainReducer from './reducer';
import SelectionView from './views/SelectionView';
import ImplementationView from './views/ImplementationView';
import ReviewComboView from './views/ReviewComboView';
import ExperimentsView from './views/ExperimentsView';

const DASHBOARD_ID = 'experiment_hub';

// ── View registry (JS-only bits: component + lifecycle predicates) ────────

registerView('selection', {
  component: SelectionView,
  defaultLabel: 'Selection',
  unlockWhen: () => true,
});

registerView('implementation', {
  component: ImplementationView,
  defaultLabel: 'Implementation',
  unlockWhen: (s) => (s.runQueue || []).some(r => r.status !== 'queued'),
  autoActivateWhen: (s) => (s.runQueue || []).some(r => r.status === 'running'),
});

registerView('review_combo', {
  component: ReviewComboView,
  defaultLabel: 'Review & Combo',
  unlockWhen: (s) => (s.runQueue || []).some(r => r.status === 'completed'),
});

registerView('experiments', {
  component: ExperimentsView,
  defaultLabel: 'Experiments',
  unlockWhen: (s) => (s.submissions || []).length > 0,
});

// ── Dashboard registry (reducer + front-end default manifest) ─────────────

// Mirrors resources/tools/experiment_hub/tool.json dashboard_config.view_manifest.
// A `dashboard_open` payload's manifest (Python canonical) wins at runtime.
const DEFAULT_MANIFEST = {
  id: DASHBOARD_ID,
  label: 'Experiment Hub',
  icon: '🔬',
  pipeline: ['Research', 'Selection', 'Implement', 'Experiment', 'Monitor'],
  views: [
    {
      type: 'selection',
      label: 'Selection',
      config: {
        widgets: [{
          type: 'proposal_selection',
          configSource: 'selectionSnapshot',
          readOnly: false,
          incrementalSubmit: true,
          enrichments: { statusMapSource: 'runQueue' },
        }],
        header: { type: 'batch_summary' },
      },
    },
    {
      type: 'implementation',
      label: 'Implementation',
      config: { renderLabel: 'batch', autoFollow: true, showAccordions: true },
    },
    {
      type: 'review_combo',
      label: 'Review & Combo',
      config: {
        tabField: 'batchId',
        submitLabel: 'Submit Experiment',
        submitAction: 'ADD_SUBMISSION',
        showComboHints: true,
        detailDrawer: true,
      },
    },
    {
      type: 'experiments',
      label: 'Experiments',
      config: {
        groupBy: 'comboKey',
        showLifecycleStepper: true,
        lifecycleSteps: ['Submitted', 'Running', 'Analyzing', 'Done'],
      },
    },
  ],
};

const dashboardReducer = createDashboardReducer(hubDomainReducer);

registerManifest(DASHBOARD_ID, DEFAULT_MANIFEST);
registerDashboard(DASHBOARD_ID, {
  reducer: dashboardReducer,
  manifest: DEFAULT_MANIFEST,
  initialState: {
    id: DASHBOARD_ID,
    activeView: 0,
    unlockedViewIds: [0],
    runQueue: [],
    currentRunIndex: -1,
    submissions: [],
    submissionSetup: null,
    scenarioState: {},
    liveSections: {},
    learningsVersion: 0,
    combosVersion: 0,
  },
});

// ── Pre-wired panel ───────────────────────────────────────────────────────

/**
 * ExperimentHubDashboard — a `<DashboardPanel>` pre-wired to the registered
 * experiment_hub dashboard. The host supplies the live wiring:
 *   - sessionId / hubId   — identity
 *   - apiClient           — the injected HubApiClient (mutations)
 *   - wsEvents$           — `.subscribe(cb)=>unsub` event bus (fed to reducer)
 *   - manifest            — optional override (Python canonical from dashboard_open)
 *   - initialState        — optional seed (resume payload)
 *   - onBack              — optional "back to session" handler
 */
export default function ExperimentHubDashboard(props) {
  return <DashboardPanel dashboardId={DASHBOARD_ID} {...props} />;
}

export { DASHBOARD_ID, DEFAULT_MANIFEST, dashboardReducer, hubDomainReducer };
