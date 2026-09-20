/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Experiment Hub domain reducer.
 *
 * The generic DashboardPanel runs a single `useReducer` whose state IS the hub
 * (the equivalent of one RankEvolve multi-task `task` object):
 *   {
 *     id, activeView, unlockedViewIds,
 *     runQueue, currentRunIndex, selectedRunIndex,
 *     submissions, submissionSetup,
 *     scenarioState: { selectionSnapshot, ... },
 *     liveSections: { [subTaskId]: [section] },   // accumulated from wsEvents$
 *     learningsVersion, combosVersion,
 *     activeBaselineSubmissionId, activeBaselineSource,
 *     statusInfo, metadata,
 *   }
 *
 * Two dispatch surfaces:
 *   1. Direct view actions (SET_SCENARIO_DATA, ADD_SUBMISSION, UPDATE_SUBMISSION,
 *      RUN_QUEUE_STATUS, BASELINE_OVERLAY_UPDATED, SUBMISSION_SETUP_*,
 *      SELECT_QUEUE_ENTRY, SNAPSHOT_RUN_SECTIONS) — ported from RankEvolve's
 *      useSessionManager multi-task cases, but scoped to THIS hub object.
 *   2. `DASHBOARD_EVENT {event}` — DashboardPanel feeds the per-hub WS event
 *      bus here; we translate the same payloads RankEvolve's SessionContext WS
 *      handler translated (task_status / setup_* / baseline_changed /
 *      combo_overrides_changed / token / message_end) into domain mutations.
 *
 * The shared view-state actions (SET_ACTIVE_VIEW / UNLOCK_VIEW / HYDRATE) are
 * layered by `createDashboardReducer` in index.js.
 */

import { getMultiTaskStatus } from './utils/multiTaskHelpers';

// ── live-section accumulation (decoupled streaming) ───────────────────────
//
// RankEvolve kept streaming tokens in mutable refs to avoid a render-per-token.
// Here the per-hub WS bus feeds the reducer; we accumulate per-subtask sections
// into `state.liveSections[subTaskId]`. QueueProgressView reads them. This is a
// faithful behavioral port of the ref machinery in a context-free shape.

function parseResponseTags(rawContent) {
  const responseStart = rawContent.indexOf('<Response>');
  if (responseStart === -1) {
    return { phase: 'pre_response', thinkingContent: rawContent, responseContent: '' };
  }
  const thinking = rawContent.slice(0, responseStart).trim();
  const afterTag = rawContent.slice(responseStart + '<Response>'.length);
  const responseEnd = afterTag.indexOf('</Response>');
  if (responseEnd === -1) {
    return { phase: 'in_response', thinkingContent: thinking, responseContent: afterTag };
  }
  return { phase: 'post_response', thinkingContent: thinking, responseContent: afterTag.slice(0, responseEnd) };
}

function withStatus(state) {
  // Keep the dashboard header chip in sync with run/submission progress.
  return { ...state, statusInfo: getMultiTaskStatus(state) };
}

function mapResumeQueueEntry(e, i, labelByBatchId) {
  const titleMatch = (e.title || '').match(/^B([^:]+):/);
  const batchId = e.batch_id || (titleMatch ? titleMatch[1] : `R${i + 1}`);
  const hypIds = (e.hypothesis_id || '')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean);
  const batchLabel = e.batch_label || labelByBatchId[batchId] || hypIds.join(',');
  const baseMeta = (e && typeof e.metadata === 'object' && e.metadata) || {};
  return {
    runIndex: i,
    label: e.title || `B${batchId}: ${batchLabel}`,
    description: '',
    status: e.status || 'queued',
    subTaskId: e.task_id,
    workspacePath: e.workspace || '',
    resultSummary: '',
    completedSections: [],
    metadata: {
      ...baseMeta,
      batchId,
      batchLabel,
      hypothesisIds: hypIds,
      taskId: e.task_id,
    },
  };
}

// ── RUN_QUEUE_STATUS (scoped to this hub) ─────────────────────────────────

function applyRunQueueStatus(state, action) {
  const {
    subTaskId, status, workspace, runIndex,
    request: runRequest, label: runLabel, metadata: runMetadata,
  } = action;
  let matched = false;
  let updatedQueue = (state.runQueue || []).map((entry, i) => {
    if (matched) return entry;
    const isMatch = runIndex !== undefined
      ? i === runIndex
      : (entry.subTaskId === subTaskId
        || ((status === 'starting' || status === 'queued') && entry.status === 'queued' && !entry.subTaskId
          && (state.runQueue || []).slice(0, i).every(e => e.status !== 'queued' || e.subTaskId)));
    if (!isMatch) return entry;
    matched = true;
    const normalizedStatus = status === 'starting' ? 'running' : status;
    const updated = { ...entry, status: normalizedStatus };
    if (subTaskId) updated.subTaskId = subTaskId;
    if (workspace) updated.workspacePath = workspace;
    if (runMetadata && Object.keys(runMetadata).length > 0) {
      updated.metadata = { ...(entry.metadata || {}), ...runMetadata };
    }
    if (runLabel && !entry.label) updated.label = runLabel;
    return updated;
  });

  if (!matched && (status === 'queued' || status === 'starting')) {
    updatedQueue = [...updatedQueue, {
      runIndex: updatedQueue.length,
      label: runLabel || runRequest || 'Task',
      description: '',
      status,
      subTaskId: subTaskId || null,
      workspacePath: workspace || '',
      resultSummary: '',
      completedSections: [],
      metadata: runMetadata || {},
    }];
  }

  const runningIdx = updatedQueue.findIndex(e => e.status === 'running');
  return withStatus({
    ...state,
    runQueue: updatedQueue,
    currentRunIndex: runningIdx >= 0 ? runningIdx : state.currentRunIndex,
  });
}

// ── live-section token handling ───────────────────────────────────────────

function applyToken(state, event) {
  const subTaskId = event.task_id;
  if (!subTaskId) return state;
  const liveSections = { ...(state.liveSections || {}) };
  const prior = liveSections[subTaskId] || [];
  const agentId = event.metadata?.agent_id || 'agent';
  // Single-agent accumulation: append to the matching section or create one.
  let section = prior.find(s => s.agentId === agentId && !s.isComplete);
  let next;
  if (!section) {
    section = { agentId, _raw: '', isComplete: false };
    next = [...prior, section];
  } else {
    next = prior.map(s => (s === section ? { ...s } : s));
    section = next.find(s => s.agentId === agentId && !s.isComplete);
  }
  section._raw = (section._raw || '') + (event.content || '');
  const parsed = parseResponseTags(section._raw);
  section.content = section._raw;
  section.thinkingContent = parsed.thinkingContent;
  section.responseContent = parsed.responseContent;
  section.responsePhase = parsed.phase;
  liveSections[subTaskId] = next;
  return { ...state, liveSections };
}

function finalizeLiveSections(state, subTaskId) {
  if (!subTaskId || !state.liveSections || !state.liveSections[subTaskId]) return state;
  const liveSections = { ...state.liveSections };
  liveSections[subTaskId] = (liveSections[subTaskId] || []).map(s => {
    const parsed = parseResponseTags(s._raw || s.content || '');
    const phase = parsed.phase === 'pre_response' ? 'no_tags' : parsed.phase;
    return {
      ...s,
      isComplete: true,
      responsePhase: phase,
      thinkingContent: parsed.thinkingContent,
      responseContent: parsed.responseContent,
    };
  });
  return { ...state, liveSections };
}

// ── DASHBOARD_EVENT translation (WS payloads → domain mutations) ──────────

function applyDashboardEvent(state, event) {
  if (!event || typeof event !== 'object') return state;
  // The host delivers the generic dashboard_event envelope {event_type, payload}
  // (mirrors the backend's WS `dashboard_event` shape); `payload` is the raw WS
  // event this translator switches on (event.type). Unwrap it. A raw event
  // (no envelope) is still accepted defensively.
  if (
    event.event_type !== undefined &&
    event.payload &&
    typeof event.payload === 'object'
  ) {
    event = event.payload;
  }
  switch (event.type) {
    case 'token':
      return applyToken(state, event);

    case 'message_end':
    case 'status': {
      // Finalize the live sections of the sub-task that just ended.
      if (event.task_id) return finalizeLiveSections(state, event.task_id);
      return state;
    }

    case 'task_status': {
      // Per-batch / per-run status updates for this hub's runQueue. The host
      // only forwards events scoped to this hub (multi_task_id === hub id or
      // metadata.hub_id === hub id), so route everything with a task_id into
      // RUN_QUEUE_STATUS.
      const subTaskId = event.task_id;
      if (!subTaskId) return state;
      let next = applyRunQueueStatus(state, {
        subTaskId,
        status: event.status,
        workspace: event.workspace,
        metadata: event.metadata || {},
        request: event.request || '',
        label: event.label || event.title || '',
      });
      if (event.status === 'completed' || event.status === 'error') {
        next = finalizeLiveSections(next, subTaskId);
      }
      // aggregate_only_refresh completion bumps learningsVersion (drives the
      // Accumulated Learnings drawer + archive-count refetch).
      if (event.status === 'completed' && event.tool_name === 'aggregate_only_refresh') {
        next = { ...next, learningsVersion: (next.learningsVersion || 0) + 1 };
      }
      return next;
    }

    case 'setup_task_started':
    case 'setup_completed': {
      // The OpenTeam bridge persists the canonical merged setup state; the
      // event carries it (event.setup) OR the host re-fetched and re-dispatched
      // a SUBMISSION_SETUP_* action. When `setup` is inline, apply it.
      if (event.setup) {
        return { ...state, submissionSetup: event.setup };
      }
      return state;
    }

    case 'baseline_changed': {
      if (Array.isArray(event.submissions)) {
        return applyBaselineOverlay(state, {
          submissions: event.submissions,
          baselineSubmissionId: event.baselineSubmissionId || event._baselineSubmissionId || null,
          source: event.source || event._baselineSource || 'none',
        });
      }
      return state;
    }

    case 'combo_overrides_changed':
      // Bump the version so per-hub combos hooks refetch. The hooks also
      // listen on the window event the host re-dispatches; this keeps the
      // reducer-derived counter in sync for any version-keyed consumers.
      return { ...state, combosVersion: (state.combosVersion || 0) + 1 };

    case 'submissions_updated':
      // Generic full-list replace (host may push a refetched list).
      if (Array.isArray(event.submissions)) {
        return withStatus({ ...state, submissions: event.submissions });
      }
      return state;

    default:
      return state;
  }
}

// ── baseline overlay ──────────────────────────────────────────────────────

function applyBaselineOverlay(state, action) {
  const overlayById = new Map();
  for (const s of action.submissions || []) {
    if (s && s.id) overlayById.set(s.id, s);
  }
  const newSubs = (state.submissions || []).map((s) => {
    const ov = overlayById.get(s.id);
    if (!ov) return s;
    return {
      ...s,
      verdict: ov.verdict,
      verdictLabel: ov.verdictLabel,
      deltaPct: ov.deltaPct,
      comparisonEpoch: ov.comparisonEpoch,
      stability: ov.stability,
      stabilityCov: ov.stabilityCov,
      baselineSubmissionId: ov.baselineSubmissionId,
    };
  });
  return {
    ...state,
    submissions: newSubs,
    activeBaselineSubmissionId: action.baselineSubmissionId || null,
    activeBaselineSource: action.source || 'none',
  };
}

// ── domain reducer ────────────────────────────────────────────────────────

export function hubDomainReducer(state, action) {
  switch (action && action.type) {
    case 'DASHBOARD_EVENT':
      return applyDashboardEvent(state, action.event);

    case 'SEED_SELECTION': {
      // Seed the Selection tab's snapshot + (optionally) the run queue +
      // submissions on open / resume. Accepts a `scenarioState` + `runQueue` +
      // `submissions` + `submissionSetup` (mirrors CREATE_MULTI_TASK's inputs).
      const labelByBatchId = {};
      const proposalsTree = action.scenarioState?.selectionSnapshot?.proposals
        || action.scenarioState?.proposals;
      if (proposalsTree && Array.isArray(proposalsTree.phases)) {
        for (const phase of proposalsTree.phases) {
          for (const batch of (phase.batches || [])) {
            if (batch && batch.id) labelByBatchId[batch.id] = batch.label || '';
          }
        }
      }
      const runQueue = Array.isArray(action.runQueue)
        ? action.runQueue.map((e, i) => (
          // Accept already-camelCased entries OR raw resume entries.
          e && (e.metadata || e.label) && !e.hypothesis_id
            ? { runIndex: i, completedSections: [], metadata: {}, ...e }
            : mapResumeQueueEntry(e, i, labelByBatchId)
        ))
        : (state.runQueue || []);
      return withStatus({
        ...state,
        scenarioState: { ...(state.scenarioState || {}), ...(action.scenarioState || {}) },
        runQueue,
        submissions: Array.isArray(action.submissions) ? action.submissions : (state.submissions || []),
        submissionSetup: action.submissionSetup ?? state.submissionSetup ?? null,
      });
    }

    case 'SET_SCENARIO_DATA':
      return {
        ...state,
        scenarioState: { ...(state.scenarioState || {}), ...(action.data || {}) },
      };

    case 'RUN_QUEUE_STATUS':
      return applyRunQueueStatus(state, action);

    case 'SELECT_QUEUE_ENTRY':
      return { ...state, selectedRunIndex: action.runIndex };

    case 'SNAPSHOT_RUN_SECTIONS': {
      const updatedQueue = (state.runQueue || []).map(entry =>
        entry.subTaskId === action.subTaskId
          ? { ...entry, completedSections: action.sections || [] }
          : entry
      );
      return { ...state, runQueue: updatedQueue };
    }

    case 'ADD_SUBMISSION': {
      const selectedItems = action.selectedItems || [];
      const comboKey = [...selectedItems].sort().join(',');
      const newSubmission = {
        id: action.submissionId || `sub-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
        comboKey,
        selectedItems,
        config: action.config || {},
        submittedAt: Date.now(),
        status: 'submitted',
        result: null,
      };
      return withStatus({
        ...state,
        submissions: [...(state.submissions || []), newSubmission],
      });
    }

    case 'UPDATE_SUBMISSION': {
      const updatedSubs = (state.submissions || []).map(s =>
        s.id === action.submissionId ? { ...s, ...action.updates } : s
      );
      return withStatus({ ...state, submissions: updatedSubs });
    }

    case 'BASELINE_OVERLAY_UPDATED':
      return applyBaselineOverlay(state, action);

    case 'SUBMISSION_SETUP_STARTED':
    case 'SUBMISSION_SETUP_TASK_STARTED':
    case 'SUBMISSION_SETUP_READY':
    case 'SUBMISSION_SETUP_ERROR':
    case 'SUBMISSION_SCRIPT_VERSION_SAVED':
      return { ...state, submissionSetup: action.setup };

    case 'BUMP_COMBOS_VERSION':
      return { ...state, combosVersion: (state.combosVersion || 0) + 1 };

    default:
      return state;
  }
}

export default hubDomainReducer;
