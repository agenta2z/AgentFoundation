/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * JobMonitorView — Generic grouped job/run lifecycle tracker. Groups
 * submissions by comboKey and shows lifecycle status per run.
 *
 * Ported from RankEvolve (components/views/JobMonitorView.js). The
 * `useSession()` reads become injected props: `apiClient`
 * ({ cancelSubmissionRun, updateSubmission, setBaseline, refetchHubSubmissions,
 * runExperimentCombos, switchTab }), `sessionId`, and `learningsVersion`
 * (read off the hub reducer state via `task.learningsVersion`). Hub-internal
 * view switches use the generic reducer's `SET_ACTIVE_VIEW {viewIndex}`;
 * jump-to-task navigations use the injected `apiClient.switchTab`.
 *
 * SplitActionButton is consumed from the shared AgentFoundation common barrel.
 */

import React, { useMemo, useState, useEffect, useCallback } from 'react';
import {
  Alert, Box, Button, Typography, Accordion, AccordionSummary, AccordionDetails,
  Chip, Stepper, Step, StepLabel, Table, TableHead, TableBody, TableRow, TableCell,
  CircularProgress, Tooltip, Select, Menu, MenuItem, IconButton,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import StopIcon from '@mui/icons-material/Stop';
import StarIcon from '@mui/icons-material/Star';
import StarBorderIcon from '@mui/icons-material/StarBorder';
import InsightsIcon from '@mui/icons-material/Insights';
import RefreshIcon from '@mui/icons-material/Refresh';
import HistoryIcon from '@mui/icons-material/History';
import TuneIcon from '@mui/icons-material/Tune';
import SplitActionButton from '../../../../common/SplitActionButton';
import useArchiveCount from '../../hooks/useArchiveCount';
import useAggregationSettings from '../../hooks/useAggregationSettings';
import AggregationSettingsPopover from './AggregationSettingsPopover';
import LazyAnalysisDoc from './LazyAnalysisDoc';
import AccumulatedLearningsDrawer from './AccumulatedLearningsDrawer';
import {
  isMetricsStub,
  sampleTrajectoryForTable,
} from '../../utils/metricsHelpers';

function verdictColor(verdict) {
  if (!verdict) return 'default';
  if (verdict === 'baseline') return 'primary';
  if (['win', 'strong_win'].includes(verdict)) return 'success';
  if (['loss', 'strong_loss', 'catastrophic_loss', 'diverged'].includes(verdict)) return 'error';
  if (verdict === 'possible_win') return 'info';
  if (verdict === 'possible_loss') return 'warning';
  return 'default';
}

const VERDICT_TIER = {
  strong_win: 7, win: 6, possible_win: 5, baseline: 4,
  neutral: 3, incomparable: 3, early_kill: 2,
  possible_loss: 1, loss: 0, strong_loss: -1,
  catastrophic_loss: -2, diverged: -3,
};
const MAX_INLINE_TABS = 5;

function pickDefaultRunIdx(runs) {
  if (!runs || runs.length === 0) return 0;
  const baseIdx = runs.findIndex(r => r.isBaseline);
  if (baseIdx >= 0) return baseIdx;
  let best = 0;
  let bestKey = [-Infinity, -Infinity, -Infinity];
  runs.forEach((r, i) => {
    const k = [
      VERDICT_TIER[r.verdict] ?? -Infinity,
      r.deltaPct ?? -Infinity,
      r.submittedAt ?? 0,
    ];
    if (
      k[0] > bestKey[0]
      || (k[0] === bestKey[0] && k[1] > bestKey[1])
      || (k[0] === bestKey[0] && k[1] === bestKey[1] && k[2] > bestKey[2])
    ) {
      best = i;
      bestKey = k;
    }
  });
  return best;
}

function pillLabel(run, runs, idx, configCount) {
  if (configCount <= 1) return `Run #${runs.length - idx}`;
  const siblings = runs.filter(r => r.comboKey === run.comboKey);
  if (siblings.length <= 1) {
    return run.comboKey || (run.selectedItems || []).join(',') || '?';
  }
  const sibIdx = siblings.findIndex(r => r.id === run.id);
  return `${run.comboKey} #${siblings.length - sibIdx}`;
}

function RunTabStrip({ runs, activeIdx, onChange, configCount }) {
  const [menuAnchor, setMenuAnchor] = useState(null);
  const inline = runs.slice(0, MAX_INLINE_TABS);
  const overflow = runs.slice(MAX_INLINE_TABS);
  return (
    <Box sx={{
      display: 'flex', gap: 0.5, alignItems: 'center', flexWrap: 'wrap',
      mb: 1.5, pb: 1, borderBottom: '1px solid rgba(255,255,255,0.06)',
    }}>
      {inline.map((r, i) => {
        const isActive = i === activeIdx;
        const label = pillLabel(r, runs, i, configCount);
        return (
          <Tooltip
            key={r.id}
            title={
              <>
                <div>{r.config?.name || r.comboKey || ''}</div>
                {r.verdictLabel && <div>verdict: {r.verdictLabel}</div>}
                {r.submittedAt && <div>{new Date(r.submittedAt).toLocaleString()}</div>}
                {r.isBaseline && <div>★ active baseline</div>}
              </>
            }
          >
            <Chip
              clickable
              size="small"
              color={verdictColor(r.verdict)}
              variant={isActive ? 'filled' : 'outlined'}
              icon={r.isBaseline ? <StarIcon sx={{ fontSize: 12 }} /> : undefined}
              label={r.verdictLabel ? `${label} · ${r.verdictLabel}` : label}
              onClick={() => onChange(i)}
              sx={{
                fontSize: '0.7rem', height: 22,
                fontWeight: isActive ? 700 : 500,
              }}
            />
          </Tooltip>
        );
      })}
      {overflow.length > 0 && (
        <>
          <Button
            size="small"
            endIcon={<ExpandMoreIcon />}
            sx={{ textTransform: 'none', fontSize: '0.7rem', minHeight: 22, py: 0 }}
            onClick={(e) => setMenuAnchor(e.currentTarget)}
          >
            +{overflow.length} more
          </Button>
          <Menu
            anchorEl={menuAnchor}
            open={Boolean(menuAnchor)}
            onClose={() => setMenuAnchor(null)}
          >
            {overflow.map((r, j) => {
              const realIdx = MAX_INLINE_TABS + j;
              const label = pillLabel(r, runs, realIdx, configCount);
              return (
                <MenuItem
                  key={r.id}
                  selected={realIdx === activeIdx}
                  onClick={() => { onChange(realIdx); setMenuAnchor(null); }}
                >
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                    {r.isBaseline && <StarIcon sx={{ fontSize: 14, color: 'warning.main' }} />}
                    <span>{label}</span>
                    {r.verdictLabel && (
                      <Chip
                        size="small"
                        label={r.verdictLabel}
                        color={verdictColor(r.verdict)}
                        sx={{ height: 18, fontSize: '0.6rem' }}
                      />
                    )}
                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                      {r.submittedAt ? new Date(r.submittedAt).toLocaleString() : ''}
                    </Typography>
                  </Box>
                </MenuItem>
              );
            })}
          </Menu>
        </>
      )}
    </Box>
  );
}

function shortLogPath(logPath) {
  if (!logPath) return '';
  const parts = logPath.split('/');
  return parts.slice(-2).join('/');
}

function getStepIndex(status, steps) {
  if (status === 'submitted') return 0;
  if (status === 'running') return 1;
  if (status === 'analyzing') return 2;
  if (status === 'completed') return steps.length;
  if (status === 'early_killed') return 1;
  if (status === 'config_pending') return 0;
  if (status === 'experiment_running') return 1;
  if (status === 'error') return -1;
  if (status === 'cancelled') return -1;
  return 0;
}

function ExperimentRunningCard({ run }) {
  const phase = run.experimentPhase || 'launching';
  const tail = (run._streamTail || '').split('\n').filter(Boolean).slice(-5);
  return (
    <Box sx={{
      mb: 2, p: 1.5,
      border: '1px dashed rgba(150,150,255,0.45)',
      borderRadius: 1,
      backgroundColor: 'rgba(80,80,255,0.05)',
    }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
        <CircularProgress size={14} thickness={5} />
        <Typography variant="caption" sx={{ fontWeight: 600 }}>
          Experiment in progress · phase: {phase}
        </Typography>
        <Box sx={{ flex: 1 }} />
        {run.comboKey && (
          <Chip label={run.comboKey} size="small" variant="outlined"
                sx={{ fontSize: '0.62rem', height: 18 }} />
        )}
      </Box>
      {tail.length > 0 && (
        <Box sx={{
          fontFamily: 'monospace', fontSize: '0.7rem',
          color: 'text.secondary', whiteSpace: 'pre-wrap',
          mt: 0.5, p: 0.5,
          background: 'rgba(0,0,0,0.18)', borderRadius: 0.5,
        }}>
          {tail.join('\n')}
        </Box>
      )}
    </Box>
  );
}

const FB_STATE_COLORS = {
  PENDING: 'default',
  RUNNING: 'info',
  COMPLETE: 'success',
  DEAD: 'error',
  CANCELLED: 'warning',
  FAILED: 'error',
};

function FblearnerBadge({ run }) {
  const state = run.fblearnerState;
  if (!state) return null;
  return (
    <Chip
      label={state}
      size="small"
      color={FB_STATE_COLORS[state] || 'default'}
      variant="outlined"
      sx={{ fontSize: '0.62rem', height: 18, ml: 0.5 }}
    />
  );
}

function MetricsTable({ run }) {
  const tableRows = useMemo(() => {
    const fromResult = run.result?.metrics || [];
    if (!isMetricsStub(fromResult)) return fromResult;
    return sampleTrajectoryForTable(
      run.epochTrajectory,
      run.comboKey,
      run.status,
    );
  }, [run.result?.metrics, run.epochTrajectory, run.comboKey, run.status]);

  return (
    <Table size="small" sx={{ '& .MuiTableCell-root': { fontSize: '0.78rem', py: 0.5 } }}>
      <TableHead>
        <TableRow>
          <TableCell>Hypothesis</TableCell>
          {Object.keys(tableRows[0] || {}).filter(k => k !== 'hypothesis').map(k => (
            <TableCell key={k} align="right">{k}</TableCell>
          ))}
        </TableRow>
      </TableHead>
      <TableBody>
        {tableRows.map((row, i) => (
          <TableRow key={i}>
            <TableCell>{row.hypothesis || `Row ${i}`}</TableCell>
            {Object.entries(row).filter(([k]) => k !== 'hypothesis').map(([k, v]) => (
              <TableCell key={k} align="right">{v}</TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

export default function JobMonitorView({ task, config, dispatch, sessionId, apiClient }) {
  const submissions = task.submissions || [];
  const steps = config.lifecycleSteps || ['Submitted', 'Running', 'Analyzing', 'Done'];
  const cancelSubmissionRun = apiClient && apiClient.cancelSubmissionRun;
  const updateSubmission = apiClient && apiClient.updateSubmission;
  const setBaseline = apiClient && apiClient.setBaseline;
  const refetchHubSubmissions = apiClient && apiClient.refetchHubSubmissions;
  const runExperimentCombos = apiClient && apiClient.runExperimentCombos;
  const switchTab = apiClient && apiClient.switchTab;
  // learningsVersion is bumped by the hub reducer when an aggregate refresh
  // completes; surfaced on the reducer state.
  const learningsVersion = task.learningsVersion || 0;

  const handleStepClick = useCallback((stepIndex, run) => {
    if (typeof switchTab !== 'function') return;
    if (stepIndex === 1 && run.runTaskId) {
      switchTab(run.runTaskId, 'task');
    } else if (stepIndex === 2 && run.analysisTaskId) {
      switchTab(run.analysisTaskId, 'task');
    } else if (stepIndex === steps.length && run.analysisTaskId) {
      switchTab(run.analysisTaskId, 'task');
    }
  }, [switchTab, steps.length]);
  const [aggregateRefreshing, setAggregateRefreshing] = useState(false);
  const archiveCount = useArchiveCount(sessionId, learningsVersion);
  const [aggSettings, saveAggSettings] = useAggregationSettings(sessionId);
  const [settingsAnchorEl, setSettingsAnchorEl] = useState(null);
  const hasTerminalCombos = useMemo(
    () => submissions.some((s) =>
      ['completed', 'analyzed', 'error', 'cancelled', 'failed'].includes(
        (s.status || '').toLowerCase(),
      ),
    ),
    [submissions],
  );
  const handleAggregateRefresh = useCallback(() => {
    if (!hasTerminalCombos) return;
    if (aggregateRefreshing) return;
    if (typeof runExperimentCombos !== 'function') return;
    setAggregateRefreshing(true);
    runExperimentCombos(task.id, {
      combos: [],
      aggregateOnly: true,
      minEpochs: aggSettings.minEpochs,
      excludeIncomparable: !aggSettings.includeIncomparable,
      excludeErrored: !aggSettings.includeErrored,
      forceRefresh: aggSettings.forceRefresh,
    });
  }, [
    hasTerminalCombos, aggregateRefreshing, runExperimentCombos, task.id,
    aggSettings.minEpochs, aggSettings.includeIncomparable,
    aggSettings.includeErrored, aggSettings.forceRefresh,
  ]);

  useEffect(() => {
    setAggregateRefreshing(false);
  }, [learningsVersion]);

  const activeBaselineId = task.activeBaselineSubmissionId || null;
  const baselineSource = task.activeBaselineSource || 'unknown';

  useEffect(() => {
    if (
      task?.id
      && typeof refetchHubSubmissions === 'function'
      && !task.activeBaselineSource
    ) {
      refetchHubSubmissions(task.id).catch(() => {});
    }
  }, [task?.id, task.activeBaselineSource, refetchHubSubmissions]);

  const baselineCandidates = useMemo(
    () => submissions.filter((s) => s.isBaseline),
    [submissions]
  );

  const activeBaselineRow = useMemo(
    () => submissions.find((s) => s.id === activeBaselineId) || null,
    [submissions, activeBaselineId]
  );

  const sortStorageKey = `rankevolve.experiments.sortBy.${task?.id || 'default'}`;
  const [sortBy, setSortByState] = useState(() => {
    try { return localStorage.getItem(sortStorageKey) || 'recency'; }
    catch (e) { return 'recency'; }
  });
  const setSortBy = useCallback((v) => {
    setSortByState(v);
    try { localStorage.setItem(sortStorageKey, v); } catch (e) { /* ignore */ }
  }, [sortStorageKey]);

  const handleToggleBaseline = useCallback(async (sub) => {
    if (!task?.id || !sub?.id || typeof updateSubmission !== 'function') return;
    const wasBaseline = !!sub.isBaseline;
    try {
      await updateSubmission(task.id, sub.id, { isBaseline: !wasBaseline });
      if (!wasBaseline && typeof setBaseline === 'function') {
        await setBaseline(task.id, sub.id);
      } else if (typeof refetchHubSubmissions === 'function') {
        await refetchHubSubmissions(task.id);
      }
    } catch (e) {
      console.warn('[JobMonitorView] toggle baseline failed', e);
    }
  }, [task?.id, updateSubmission, setBaseline, refetchHubSubmissions]);

  const handlePickBaseline = useCallback(async (newId) => {
    if (!task?.id || typeof setBaseline !== 'function') return;
    try {
      await setBaseline(task.id, newId || null);
    } catch (e) {
      console.warn('[JobMonitorView] pick baseline failed', e);
    }
  }, [task?.id, setBaseline]);

  const [learningsOpen, setLearningsOpen] = useState(false);

  const activeRunStorageKey = `rankevolve_hub_combo_active_run_${task?.id || 'default'}`;
  const [activeRunByCombo, setActiveRunByCombo] = useState(() => {
    try {
      const raw = localStorage.getItem(activeRunStorageKey);
      const parsed = raw ? JSON.parse(raw) : {};
      return parsed && typeof parsed === 'object' ? parsed : {};
    } catch (e) {
      return {};
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(activeRunStorageKey, JSON.stringify(activeRunByCombo));
    } catch (e) { /* quota / disabled — silently ignore */ }
  }, [activeRunStorageKey, activeRunByCombo]);
  const setActiveRunForCombo = useCallback((groupKey, idx) => {
    setActiveRunByCombo((prev) => ({ ...prev, [groupKey]: idx }));
  }, []);

  const handleOpenInMLHub = useCallback((flowUri) => {
    if (!flowUri) return;
    try {
      window.open(flowUri, '_blank', 'noopener,noreferrer');
    } catch (e) {
      console.warn('[JobMonitorView] open failed', e);
    }
  }, []);

  const handleCancelRun = useCallback(async (submissionId) => {
    if (!task?.id || !submissionId) return;
    if (typeof cancelSubmissionRun !== 'function') return;
    await cancelSubmissionRun(task.id, submissionId);
  }, [task?.id, cancelSubmissionRun]);

  const handleViewSubprocessOutput = useCallback((runTaskId) => {
    if (!runTaskId) return;
    if (typeof switchTab === 'function') switchTab(runTaskId, 'task');
  }, [switchTab]);

  const comboGroups = useMemo(() => {
    const groups = {};
    submissions.forEach(sub => {
      const items = sub.selectedItems || [];
      const key = [...items].sort().join(',') || 'unknown';
      if (!groups[key]) {
        groups[key] = {
          comboKey: key,
          items,
          runs: [],
        };
      }
      groups[key].runs.push(sub);
    });
    Object.values(groups).forEach(g => {
      g.runs.sort((a, b) => {
        const ab = (b.isBaseline ? 1 : 0) - (a.isBaseline ? 1 : 0);
        if (ab !== 0) return ab;
        const ck = (a.comboKey || '').localeCompare(b.comboKey || '');
        if (ck !== 0) return ck;
        return (b.submittedAt || 0) - (a.submittedAt || 0);
      });
      g.configCount = new Set(g.runs.map(r => r.comboKey)).size;
    });
    return Object.values(groups);
  }, [submissions]);

  const sortedGroups = useMemo(() => {
    const arr = [...comboGroups];
    arr.sort((a, b) => {
      const aBase = a.runs.some(r => r.isBaseline);
      const bBase = b.runs.some(r => r.isBaseline);
      if (aBase !== bBase) return bBase - aBase;
      switch (sortBy) {
        case 'recency': {
          const aT = Math.max(...a.runs.map(r => r.submittedAt || 0));
          const bT = Math.max(...b.runs.map(r => r.submittedAt || 0));
          return bT - aT;
        }
        case 'winMagnitude': {
          const aD = Math.max(...a.runs.map(r => r.deltaPct ?? -Infinity));
          const bD = Math.max(...b.runs.map(r => r.deltaPct ?? -Infinity));
          return bD - aD;
        }
        case 'lossMagnitude': {
          const aD = Math.min(...a.runs.map(r => r.deltaPct ?? Infinity));
          const bD = Math.min(...b.runs.map(r => r.deltaPct ?? Infinity));
          return aD - bD;
        }
        case 'hypothesis':
          return (a.items[0] || '').localeCompare(b.items[0] || '');
        case 'stability': {
          const rank = { stable: 0, unstable: 1, insufficient: 2, diverged: 3 };
          const aR = Math.min(...a.runs.map(r => rank[r.stability] ?? 9));
          const bR = Math.min(...b.runs.map(r => rank[r.stability] ?? 9));
          return aR - bR;
        }
        case 'status': {
          const rank = { completed: 0, early_killed: 1, error: 2 };
          const aR = Math.min(...a.runs.map(r => rank[r.status] ?? 9));
          const bR = Math.min(...b.runs.map(r => rank[r.status] ?? 9));
          return aR - bR;
        }
        case 'epochs': {
          const aE = Math.max(...a.runs.map(r => r.epochsCompleted || 0));
          const bE = Math.max(...b.runs.map(r => r.epochsCompleted || 0));
          return bE - aE;
        }
        default:
          return 0;
      }
    });
    return arr;
  }, [comboGroups, sortBy]);

  const handleGoToCombo = useCallback(() => {
    dispatch({ type: 'SET_ACTIVE_VIEW', viewIndex: 2 });
  }, [dispatch]);

  if (submissions.length === 0) {
    return (
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 2, p: 4 }}>
        <Typography color="text.secondary">No experiments submitted yet.</Typography>
        <Button variant="outlined" onClick={handleGoToCombo} sx={{ textTransform: 'none' }}>
          ← Submit a combo from the Review tab
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
        <Typography variant="caption" color="text.secondary">
          Sort by:
        </Typography>
        <Select
          size="small"
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
          sx={{ fontSize: '0.78rem', minWidth: 160, '& .MuiSelect-select': { py: 0.5 } }}
        >
          <MenuItem value="recency">Recency (newest first)</MenuItem>
          <MenuItem value="winMagnitude">Win magnitude</MenuItem>
          <MenuItem value="lossMagnitude">Loss magnitude</MenuItem>
          <MenuItem value="hypothesis">Hypothesis</MenuItem>
          <MenuItem value="stability">Stability</MenuItem>
          <MenuItem value="status">Status</MenuItem>
          <MenuItem value="epochs">Epochs</MenuItem>
        </Select>
        <SplitActionButton
          variant="outlined"
          size="small"
          sx={{ ml: 1 }}
          primary={{
            label: 'Accumulated Learnings',
            icon: <InsightsIcon />,
            tooltip: 'View accumulated learnings: cross-experiment synthesis + proposed re-rankings + future combo recommendations',
            onClick: () => setLearningsOpen(true),
          }}
          secondary={[
            {
              key: 'refresh',
              icon: <RefreshIcon fontSize="small" />,
              label: 'Refresh Learnings',
              tooltip: 'Re-run LLM aggregator (~30s; staged + archived; preserves edits)',
              disabledTooltip: 'No completed combos yet — run /experiment-hypothesis-combos first',
              disabled: !hasTerminalCombos || aggregateRefreshing,
              loading: aggregateRefreshing,
              ariaLabel: 'Refresh accumulated learnings',
              onClick: handleAggregateRefresh,
            },
            {
              key: 'archives',
              icon: <HistoryIcon fontSize="small" />,
              label: 'View archives…',
              tooltip: archiveCount > 0
                ? `${archiveCount} archived version${archiveCount === 1 ? '' : 's'}`
                : undefined,
              disabledTooltip: 'No archives yet',
              disabled: archiveCount === 0,
              ariaLabel: 'View accumulated learnings archives',
              onClick: () => setLearningsOpen(true),
            },
            {
              key: 'settings',
              icon: <TuneIcon fontSize="small" />,
              label: 'Aggregation settings…',
              tooltip: 'Configure which experiments feed the LLM aggregator',
              ariaLabel: 'Aggregation settings',
              onClick: (e) => setSettingsAnchorEl(
                e?.currentTarget || e?.target || null,
              ),
            },
          ]}
        />
        <AggregationSettingsPopover
          open={!!settingsAnchorEl}
          anchorEl={settingsAnchorEl}
          onClose={() => setSettingsAnchorEl(null)}
          settings={aggSettings}
          onSave={saveAggSettings}
        />
        <Box sx={{ flex: 1 }} />
        <Typography variant="caption" color="text.secondary">
          {sortedGroups.length} hypothesis group{sortedGroups.length === 1 ? '' : 's'} · baselines pinned to top
        </Typography>
      </Box>

      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5, flexWrap: 'wrap' }}>
        <Typography variant="caption" color="text.secondary">
          Compare vs:
        </Typography>
        <Select
          size="small"
          value={
            baselineSource === 'choice_file' && activeBaselineId
              ? activeBaselineId
              : ''
          }
          onChange={(e) => handlePickBaseline(e.target.value)}
          displayEmpty
          sx={{ fontSize: '0.78rem', minWidth: 220, '& .MuiSelect-select': { py: 0.5 } }}
        >
          <MenuItem value="">
            <em>
              Auto
              {activeBaselineRow
                ? ` · ${activeBaselineRow.comboKey || activeBaselineRow.id}`
                : ''}
            </em>
          </MenuItem>
          {baselineCandidates.map((b) => (
            <MenuItem key={b.id} value={b.id}>
              {b.comboKey || b.id}
              {(b.finalMetrics && typeof b.finalMetrics.ndcg_10 === 'number')
                ? ` (NDCG@10=${b.finalMetrics.ndcg_10.toFixed(4)})`
                : ''}
            </MenuItem>
          ))}
        </Select>
        <Tooltip
          title={
            baselineSource === 'query'
              ? 'Active baseline: explicit query-string override'
              : baselineSource === 'choice_file'
              ? 'Active baseline: persisted user choice (this session)'
              : baselineSource === 'is_baseline_resolver'
              ? `Active baseline: auto-picked from ${baselineCandidates.length} flagged baselines (highest NDCG)`
              : baselineSource === 'is_baseline_first'
              ? 'Active baseline: the only flagged baseline'
              : baselineSource === 'none'
              ? 'No baseline flagged — chips show persisted values'
              : 'Loading baseline state…'
          }
        >
          <Chip
            size="small"
            variant="outlined"
            label={
              baselineSource === 'choice_file'
                ? 'user choice'
                : baselineSource === 'is_baseline_resolver'
                ? `auto · best of ${baselineCandidates.length}`
                : baselineSource === 'is_baseline_first'
                ? 'auto · single'
                : baselineSource === 'query'
                ? 'query override'
                : baselineSource === 'none'
                ? 'no baseline'
                : 'loading…'
            }
            sx={{ fontSize: '0.62rem', height: 18 }}
          />
        </Tooltip>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ fontStyle: 'italic' }}
        >
          ⓘ Chip Δ recomputes against the active baseline. Per-row analysis
          text and the Accumulated Learnings doc reflect the original
          baseline at the time they were written.
        </Typography>
      </Box>

      {sortedGroups.map((group) => {
        const isBaselineGroup = group.runs.some(r => r.isBaseline);
        return (
        <Accordion key={group.comboKey} defaultExpanded={sortedGroups.length <= 3 || isBaselineGroup} sx={{
          mb: 2,
          backgroundColor: isBaselineGroup ? 'rgba(255,215,0,0.06)' : 'rgba(255,255,255,0.02)',
          border: isBaselineGroup ? '1px solid rgba(255,215,0,0.45)' : '1px solid rgba(255,255,255,0.08)',
          '&:before': { display: 'none' },
        }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: 1 }}>
              <Typography sx={{ fontWeight: 600, fontSize: '0.9rem' }}>
                Hypothesis: {group.items.join(', ')}
              </Typography>
              <Chip label={`${group.runs.length} run${group.runs.length > 1 ? 's' : ''}`}
                size="small" variant="outlined" sx={{ fontSize: '0.65rem', height: 20 }} />
              {group.configCount > 1 && (
                <Chip label={`${group.configCount} configs`}
                  size="small" variant="outlined"
                  sx={{ fontSize: '0.65rem', height: 20 }} />
              )}
              <Box sx={{ flex: 1 }} />
              <Button
                size="small"
                sx={{ textTransform: 'none', fontSize: '0.72rem' }}
                onClick={(e) => {
                  e.stopPropagation();
                  handleGoToCombo();
                }}
              >
                Re-select →
              </Button>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            {(() => {
              const stored = activeRunByCombo[group.comboKey];
              const fallback = pickDefaultRunIdx(group.runs);
              const activeIdx = (
                Number.isInteger(stored)
                && stored >= 0
                && stored < group.runs.length
              ) ? stored : fallback;
              const visibleRuns = group.runs.length > 1
                ? [{ run: group.runs[activeIdx], runIdx: activeIdx }]
                : group.runs.map((r, i) => ({ run: r, runIdx: i }));
              return (
                <>
                  {group.runs.length > 1 && (
                    <RunTabStrip
                      runs={group.runs}
                      activeIdx={activeIdx}
                      configCount={group.configCount}
                      onChange={(i) => setActiveRunForCombo(group.comboKey, i)}
                    />
                  )}
                  {visibleRuns.map(({ run, runIdx }) => (
              run.status === 'experiment_running'
                ? <ExperimentRunningCard key={run.id} run={run} />
                : (
              <Box key={run.id} sx={{
                mb: 2, p: 2,
                border: '1px solid rgba(255,255,255,0.06)',
                borderRadius: 1,
                backgroundColor: 'rgba(255,255,255,0.01)',
              }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1, flexWrap: 'wrap' }}>
                  <Tooltip
                    title={
                      run.isBaseline
                        ? 'Unmark as baseline (verdict overlays recompute against the resolver pick)'
                        : 'Mark as baseline (pins to top; chip Δ + verdict matrix re-derive against this run; analysis text below remains vs the original baseline)'
                    }
                  >
                    <IconButton
                      size="small"
                      onClick={(e) => { e.stopPropagation(); handleToggleBaseline(run); }}
                      sx={{ p: 0.25, color: run.isBaseline ? 'warning.main' : 'text.disabled' }}
                    >
                      {run.isBaseline ? <StarIcon fontSize="small" /> : <StarBorderIcon fontSize="small" />}
                    </IconButton>
                  </Tooltip>
                  <Typography sx={{ fontWeight: 600, fontSize: '0.85rem' }}>
                    Run #{group.runs.length - runIdx}
                  </Typography>
                  {group.configCount > 1 && run.comboKey && (
                    <Tooltip
                      title={
                        run.config?.name
                        || run.config?.gin_config
                        || run.comboKey
                      }
                    >
                      <Chip
                        label={run.comboKey}
                        size="small"
                        color={verdictColor(run.verdict)}
                        variant="outlined"
                        sx={{ fontSize: '0.62rem', height: 18 }}
                      />
                    </Tooltip>
                  )}
                  <Typography variant="caption" color="text.secondary">
                    {run.submittedAt ? new Date(run.submittedAt).toLocaleString() : ''}
                  </Typography>
                  <Box sx={{ flex: 1 }} />
                  {run.verdictLabel && (
                    <Tooltip
                      title={
                        run.verdict === 'baseline'
                          ? 'This run is the active baseline'
                          : `Δ ${(run.deltaPct ?? 0) >= 0 ? '+' : ''}${run.deltaPct ?? '?'}% vs baseline (${run.baselineSubmissionId || '?'}) at fair-epoch ${run.comparisonEpoch ?? '?'} · stability=${run.stability || '?'} (CoV=${run.stabilityCov ?? '?'})`
                      }
                    >
                      <Chip
                        label={run.verdictLabel}
                        size="small"
                        color={verdictColor(run.verdict)}
                        variant={run.verdict === 'baseline' ? 'filled' : 'outlined'}
                        sx={{ fontSize: '0.65rem', height: 20, fontWeight: 600 }}
                      />
                    </Tooltip>
                  )}
                  <Tooltip title={run.runTaskId ? 'Open run terminal output' : ''}>
                    <Chip
                      label={
                        run.status === 'early_killed' ? 'early killed'
                        : run.status === 'config_pending' ? 'config pending'
                        : run.status
                      }
                      size="small"
                      color={
                        run.status === 'completed' ? 'success'
                        : run.status === 'error' ? 'error'
                        : run.status === 'cancelled' || run.status === 'early_killed' ? 'warning'
                        : run.status === 'config_pending' ? 'default'
                        : 'info'
                      }
                      variant="outlined"
                      clickable={Boolean(run.runTaskId)}
                      onClick={
                        run.runTaskId
                          ? () => switchTab && switchTab(run.runTaskId, 'task')
                          : undefined
                      }
                      sx={{
                        fontSize: '0.65rem',
                        height: 20,
                        cursor: run.runTaskId ? 'pointer' : 'default',
                      }}
                    />
                  </Tooltip>
                  {run.runMode !== 'local' && <FblearnerBadge run={run} />}
                </Box>

                {run.runMode === 'local' && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', mb: 1 }}>
                    <Chip
                      label="local run"
                      size="small"
                      variant="outlined"
                      sx={{ fontSize: '0.62rem', height: 18 }}
                    />
                    {run.runHost && (
                      <Typography variant="caption" color="text.secondary">
                        host: <code>{run.runHost}</code>
                      </Typography>
                    )}
                    {run.runLogPath && (
                      <Tooltip title={run.runLogPath}>
                        <Typography variant="caption" color="text.secondary">
                          · log: <code>{shortLogPath(run.runLogPath)}</code>
                        </Typography>
                      </Tooltip>
                    )}
                    {run.epochsCompleted != null && (
                      <Typography variant="caption" color="text.secondary">
                        · epochs: <code>{run.epochsCompleted}</code>
                      </Typography>
                    )}
                  </Box>
                )}

                {run.runMode !== 'local' && (run.flowUri || run.experimentId) && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', mb: 1 }}>
                    <Typography variant="caption" color="text.secondary">
                      flow: <code>{run.experimentId || (run.flowUri || '').split('/').pop()}</code>
                    </Typography>
                    {run.mastJob && (
                      <Typography variant="caption" color="text.secondary">
                        · mast: <code>{run.mastJob}</code>
                      </Typography>
                    )}
                    {run.fblearnerLastPolledAt && (
                      <Tooltip title={new Date(run.fblearnerLastPolledAt * 1000).toLocaleString()}>
                        <Typography variant="caption" color="text.secondary">
                          · polled {Math.max(1, Math.round((Date.now() / 1000 - run.fblearnerLastPolledAt) / 1))}s ago
                        </Typography>
                      </Tooltip>
                    )}
                    {run.flowUri && (
                      <Button
                        size="small"
                        startIcon={<OpenInNewIcon fontSize="inherit" />}
                        onClick={() => handleOpenInMLHub(run.flowUri)}
                        sx={{ textTransform: 'none', fontSize: '0.7rem', py: 0 }}
                      >
                        Open in MLHub
                      </Button>
                    )}
                  </Box>
                )}

                {(run.runTaskId || (run.status === 'running' && run.runTaskId)) && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', mb: 1 }}>
                    {run.runTaskId && (
                      <Button
                        size="small"
                        onClick={() => handleViewSubprocessOutput(run.runTaskId)}
                        sx={{ textTransform: 'none', fontSize: '0.7rem', py: 0 }}
                      >
                        View subprocess output
                      </Button>
                    )}
                    {run.status === 'running' && run.runTaskId && (
                      <Button
                        size="small"
                        color="warning"
                        startIcon={<StopIcon fontSize="inherit" />}
                        onClick={() => handleCancelRun(run.id)}
                        sx={{ textTransform: 'none', fontSize: '0.7rem', py: 0 }}
                      >
                        Cancel run
                      </Button>
                    )}
                  </Box>
                )}

                {run.runMode !== 'local' && run.fblearnerMetrics && Object.keys(run.fblearnerMetrics).length > 0 && (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 1 }}>
                    {run.fblearnerMetrics.gpuUtil != null && (
                      <Chip size="small" variant="outlined" label={`GPU: ${run.fblearnerMetrics.gpuUtil}`} sx={{ fontSize: '0.65rem', height: 18 }} />
                    )}
                    {run.fblearnerMetrics.hostsHealthy != null && (
                      <Chip size="small" variant="outlined" label={`hosts: ${run.fblearnerMetrics.hostsHealthy}`} sx={{ fontSize: '0.65rem', height: 18 }} />
                    )}
                    {run.fblearnerMetrics.errors != null && (
                      <Chip size="small" variant="outlined"
                        color={run.fblearnerMetrics.errors > 0 ? 'warning' : 'default'}
                        label={`errors: ${run.fblearnerMetrics.errors}`}
                        sx={{ fontSize: '0.65rem', height: 18 }} />
                    )}
                    {run.fblearnerMetrics.costUsd != null && (
                      <Chip size="small" variant="outlined" label={`cost: $${run.fblearnerMetrics.costUsd}`} sx={{ fontSize: '0.65rem', height: 18 }} />
                    )}
                  </Box>
                )}

                {(run.killReason || run.fblearnerError) && (
                  <Typography variant="body2" color="error" sx={{ fontSize: '0.78rem', mb: 1 }}>
                    {run.runMode === 'local' ? 'Kill reason' : 'FBLearner error'}: {run.killReason || run.fblearnerError}
                  </Typography>
                )}

                {config.showLifecycleStepper && (
                  <Stepper
                    activeStep={getStepIndex(run.status, steps)}
                    alternativeLabel
                    sx={{ mb: 1, '& .MuiStepLabel-label': { fontSize: '0.7rem' } }}
                  >
                    {steps.map((label, stepIndex) => {
                      const isClickable = (
                        (stepIndex === 1 && Boolean(run.runTaskId))
                        || (stepIndex === 2 && Boolean(run.analysisTaskId))
                        || (stepIndex === steps.length && Boolean(run.analysisTaskId))
                      );
                      return (
                        <Step key={label}>
                          <StepLabel
                            onClick={
                              isClickable
                                ? () => handleStepClick(stepIndex, run)
                                : undefined
                            }
                            sx={{
                              cursor: isClickable ? 'pointer' : 'default',
                              ...(isClickable && {
                                '&:hover .MuiStepLabel-label': {
                                  textDecoration: 'underline',
                                },
                              }),
                            }}
                          >
                            {label}
                          </StepLabel>
                        </Step>
                      );
                    })}
                  </Stepper>
                )}

                {run.status === 'early_killed' && (
                  <Alert severity="warning" sx={{ mb: 1, fontSize: '0.78rem', py: 0.25 }}>
                    Run was killed at epoch {run.epochsCompleted ?? '?'} before reaching the
                    convergence threshold (&gt; 30 epochs). Numbers below reflect partial training only.
                    {run.fblearnerError ? ` Reason: ${run.fblearnerError}.` : ''}
                  </Alert>
                )}

                {run.config && (run.config.name || run.config.baseConfig) && (
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                    Config: {run.config.name || ''} {run.config.baseConfig ? `(${run.config.baseConfig})` : ''}
                  </Typography>
                )}

                {run.status === 'running' && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                    <CircularProgress size={16} />
                    <Typography variant="body2" sx={{ fontSize: '0.82rem' }}>
                      Experiment running...
                    </Typography>
                  </Box>
                )}

                {(run.status === 'completed' || run.status === 'early_killed') && run.result && (
                  <Box sx={{ mt: 1 }}>
                    <MetricsTable run={run} />
                    {(run.analysisSummary || run.result.analysis) && (
                      <LazyAnalysisDoc
                        summary={run.analysisSummary}
                        fallbackInline={run.result.analysis}
                        analysisFile={run.analysisFile}
                      />
                    )}
                  </Box>
                )}

                {run.status === 'error' && (
                  <Typography variant="body2" color="error" sx={{ mt: 1, fontSize: '0.82rem' }}>
                    Experiment failed. Check logs for details.
                  </Typography>
                )}
                {run.status === 'error' && run.analysisFile && (
                  <LazyAnalysisDoc
                    summary={run.analysisSummary}
                    fallbackInline={run.result?.analysis}
                    analysisFile={run.analysisFile}
                  />
                )}
              </Box>
                )
                  ))}
                </>
              );
            })()}
          </AccordionDetails>
        </Accordion>
        );
      })}

      <Box sx={{ textAlign: 'center', mt: 2 }}>
        <Button variant="outlined" onClick={handleGoToCombo} sx={{ textTransform: 'none' }}>
          ← Submit Another Combo
        </Button>
      </Box>

      <AccumulatedLearningsDrawer
        open={learningsOpen}
        onClose={() => setLearningsOpen(false)}
        multiTaskId={task?.id}
        taskSubmissions={submissions}
        sessionId={sessionId}
        learningsVersion={learningsVersion}
        addSubmission={apiClient && apiClient.addSubmission}
        onJumpToSelection={() => {
          const targetTaskId = task?.id;
          if (!targetTaskId) {
            console.warn('[AccumulatedLearningsDrawer] onJumpToSelection: task.id missing; skipping auto-jump');
            return;
          }
          setLearningsOpen(false);
          const fire = () => {
            dispatch({ type: 'SET_ACTIVE_VIEW', viewIndex: 0 });
          };
          if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
            window.requestAnimationFrame(fire);
          } else {
            setTimeout(fire, 0);
          }
        }}
      />
    </Box>
  );
}
