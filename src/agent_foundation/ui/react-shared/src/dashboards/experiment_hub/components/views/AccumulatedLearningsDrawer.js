/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * AccumulatedLearningsDrawer — right-anchored drawer showing the per-session
 * learnings doc + ranking adjustments + future combo recommendations + Apply
 * tab.
 *
 * Ported from RankEvolve (components/views/AccumulatedLearningsDrawer.js). The
 * `useSession()` reads become injected props: `sessionId`, `learningsVersion`,
 * and `addSubmission` (apiClient.addSubmission). The learnings doc / archives
 * / proposal-overrides reads + writes are session-scoped REST (not part of the
 * hub mutation surface), so they stay direct-fetch (via applyHelpers + inline
 * fetch). The `onJumpToSelection` callback is threaded into ApplyChangesView
 * (the RankEvolve original referenced it out of scope — fixed here so Apply
 * All's success path works).
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Drawer, AppBar, Toolbar, Box, Tabs, Tab, Typography, IconButton, Button,
  CircularProgress, Alert, Chip, Tooltip, Card, CardContent, Checkbox,
  Table, TableHead, TableBody, TableRow, TableCell, Collapse, Divider,
  Select, MenuItem,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import RefreshIcon from '@mui/icons-material/Refresh';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import ArrowRightAltIcon from '@mui/icons-material/ArrowRightAlt';
import UndoIcon from '@mui/icons-material/Undo';
import RestoreIcon from '@mui/icons-material/Restore';
import { MarkdownRenderer } from '../../../../common/MarkdownRenderer';
import { useHypothesisImplementations } from '../../hooks/useHypothesisImplementations';
import {
  isComboApplyEligible,
  partitionCombos,
} from '../../utils/comboPendingPredicate';
import { postRerankApply, postCombosApply } from '../../utils/applyHelpers';

// ─────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────

function verdictChipColor(confidence) {
  if (!confidence) return 'default';
  if (confidence === 'high') return 'success';
  if (confidence === 'medium-high') return 'info';
  if (confidence === 'medium') return 'default';
  return 'warning';
}

function deltaArrow(deltaRank) {
  if (deltaRank === 0 || deltaRank == null) return <ArrowRightAltIcon fontSize="inherit" />;
  if (deltaRank < 0) return <ArrowUpwardIcon fontSize="inherit" sx={{ color: 'success.main' }} />;
  return <ArrowDownwardIcon fontSize="inherit" sx={{ color: 'warning.main' }} />;
}

// ─────────────────────────────────────────────────────────────────────────
// Sub-views
// ─────────────────────────────────────────────────────────────────────────

function FullDocView({ markdownBody }) {
  if (!markdownBody) return <Typography color="text.secondary">No doc body.</Typography>;
  return (
    <Box sx={{ '& h1': { mt: 0 }, '& table': { fontSize: '0.78rem' }, fontSize: '0.86rem' }}>
      <MarkdownRenderer content={markdownBody} />
    </Box>
  );
}

function HypothesisRerankView({ entries, selectedIds, onToggle, onSelectAll, onClearAll }) {
  const [expanded, setExpanded] = useState(null);
  if (!entries?.length) {
    return <Typography color="text.secondary">No re-ranking adjustments.</Typography>;
  }
  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
        <Typography variant="caption" color="text.secondary">
          {selectedIds.size} of {entries.length} selected
        </Typography>
        <Box sx={{ flex: 1 }} />
        <Button size="small" onClick={onSelectAll} sx={{ textTransform: 'none', fontSize: '0.72rem' }}>
          Select all
        </Button>
        <Button size="small" onClick={onClearAll} sx={{ textTransform: 'none', fontSize: '0.72rem' }}>
          Clear
        </Button>
      </Box>
      <Table size="small" sx={{ '& .MuiTableCell-root': { fontSize: '0.78rem', py: 0.5 } }}>
        <TableHead>
          <TableRow>
            <TableCell padding="checkbox" />
            <TableCell>H</TableCell>
            <TableCell align="center">Old → New</TableCell>
            <TableCell align="center">Δ</TableCell>
            <TableCell>Confidence</TableCell>
            <TableCell>Rationale</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {entries.map((r) => (
            <React.Fragment key={r.id}>
              <TableRow hover sx={{ cursor: 'pointer' }}
                onClick={() => setExpanded(expanded === r.id ? null : r.id)}>
                <TableCell padding="checkbox" onClick={(e) => e.stopPropagation()}>
                  <Checkbox
                    size="small"
                    checked={selectedIds.has(r.id)}
                    onChange={() => onToggle(r.id)}
                  />
                </TableCell>
                <TableCell><strong>{r.id}</strong></TableCell>
                <TableCell align="center">
                  <Box sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.5 }}>
                    <span>{r.oldRank}</span>
                    {deltaArrow(r.deltaRank)}
                    <strong>{r.newRank}</strong>
                  </Box>
                </TableCell>
                <TableCell align="center">
                  <Chip
                    label={r.deltaRank > 0 ? `+${r.deltaRank}` : `${r.deltaRank}`}
                    size="small"
                    color={r.deltaRank < 0 ? 'success' : (r.deltaRank > 0 ? 'warning' : 'default')}
                    variant="outlined"
                    sx={{ fontSize: '0.62rem', height: 18 }}
                  />
                </TableCell>
                <TableCell>
                  <Chip label={r.confidence || '?'} size="small" color={verdictChipColor(r.confidence)} variant="outlined" sx={{ fontSize: '0.62rem', height: 18 }} />
                </TableCell>
                <TableCell sx={{ maxWidth: 320, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  <Tooltip title={r.rationale || ''} placement="top">
                    <span>{r.rationale || '—'}</span>
                  </Tooltip>
                </TableCell>
              </TableRow>
              {expanded === r.id && (
                <TableRow>
                  <TableCell colSpan={6} sx={{ backgroundColor: 'rgba(255,255,255,0.02)' }}>
                    <Typography variant="body2" sx={{ fontSize: '0.78rem', whiteSpace: 'pre-wrap' }}>
                      {r.rationale || '(no rationale)'}
                    </Typography>
                    {r.evidenceSubmissions?.length > 0 && (
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                        Evidence: {r.evidenceSubmissions.join(', ')}
                      </Typography>
                    )}
                  </TableCell>
                </TableRow>
              )}
            </React.Fragment>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
}

function ComboRecommendationsView({ combos, taskSubmissions, selectedIds, onToggle, onSelectAll, onClearAll, implementations }) {
  const [expandedRisk, setExpandedRisk] = useState({});

  if (!combos?.length) {
    return <Typography color="text.secondary">No combo recommendations.</Typography>;
  }

  function alreadySubmitted(comboId) {
    return (taskSubmissions || []).some(
      s => s?.config?.future_combo_id === comboId
    );
  }

  const submittableCount = combos.filter(c => !alreadySubmitted(c.comboId)).length;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
        <Typography variant="caption" color="text.secondary">
          {selectedIds.size} of {submittableCount} selected
          {combos.length - submittableCount > 0
            ? ` (${combos.length - submittableCount} already submitted)` : ''}
        </Typography>
        <Box sx={{ flex: 1 }} />
        {onSelectAll && (
          <Button size="small" onClick={onSelectAll} sx={{ textTransform: 'none', fontSize: '0.72rem' }}>
            Select all
          </Button>
        )}
        {onClearAll && (
          <Button size="small" onClick={onClearAll} sx={{ textTransform: 'none', fontSize: '0.72rem' }}>
            Clear
          </Button>
        )}
      </Box>
      {combos.map((c) => {
        const submitted = alreadySubmitted(c.comboId);
        const { gatingHypotheses, configReady } =
          isComboApplyEligible(c, implementations);
        const pendingImpl = gatingHypotheses.length > 0;
        const pendingCfg = !configReady;
        const checkboxDisabled = submitted;
        const pendingTooltip = pendingImpl || pendingCfg
          ? (`You can still select this combo — it will be queued and `
            + `auto-enable in Review & Combos when ${pendingImpl ? 'implementation' : 'config'} completes.`)
          : '';
        return (
          <Card key={c.comboId} variant="outlined" sx={{ opacity: submitted ? 0.7 : 1 }}>
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1, flexWrap: 'wrap' }}>
                <Tooltip title={pendingTooltip} placement="top" arrow disableHoverListener={!pendingTooltip}>
                  <span>
                    <Checkbox
                      size="small"
                      checked={selectedIds.has(c.comboId)}
                      disabled={checkboxDisabled}
                      onChange={() => onToggle(c.comboId)}
                      sx={{ p: 0 }}
                    />
                  </span>
                </Tooltip>
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                  {c.comboId}
                </Typography>
                <Chip label={c.comboKey} size="small" sx={{ fontSize: '0.62rem', height: 20, fontFamily: 'monospace' }} />
                {(c.selectedItems || []).map(h => (
                  <Chip key={h} label={h} size="small" variant="outlined" sx={{ fontSize: '0.62rem', height: 18 }} />
                ))}
                <Box sx={{ flex: 1 }} />
                {pendingImpl && (
                  <Tooltip title={`Pending implementation of: ${gatingHypotheses.join(', ')}. Will auto-enable in Review & Combos when these land.`} placement="top" arrow>
                    <Chip
                      label={`Queued — implement ${gatingHypotheses.slice(0, 3).join(', ')}${gatingHypotheses.length > 3 ? `, +${gatingHypotheses.length - 3}` : ''}`}
                      size="small"
                      color="info"
                      variant="outlined"
                      sx={{ fontSize: '0.62rem', height: 20 }}
                    />
                  </Tooltip>
                )}
                {pendingCfg && (
                  <Tooltip title={`Pending config generation: ${c.configPathProposed || c.configStatus || ''}. Will auto-enable when generation completes.`} placement="top" arrow>
                    <Chip
                      label={`Queued — config ${c.configStatus || 'needs_generation'}`}
                      size="small"
                      color="info"
                      variant="outlined"
                      sx={{ fontSize: '0.62rem', height: 20 }}
                    />
                  </Tooltip>
                )}
                {submitted && (
                  <Chip label="✓ Already submitted" size="small" color="success" sx={{ fontSize: '0.62rem', height: 20 }} />
                )}
              </Box>
              {c.title && (
                <Typography variant="body2" sx={{ fontWeight: 500, mb: 0.5 }}>
                  {c.title}
                </Typography>
              )}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1, flexWrap: 'wrap' }}>
                <Tooltip title={`Predicted absolute NDCG@10 ≈ ${c.expectedNdcg10Absolute ?? '?'}`}>
                  <Chip
                    label={`Δ +${c.expectedNdcg10Lift ?? '?'}%`}
                    size="small"
                    color={c.expectedNdcg10Lift > 1 ? 'success' : 'default'}
                    sx={{ fontWeight: 600 }}
                  />
                </Tooltip>
                <Chip label={c.confidence || '?'} size="small" color={verdictChipColor(c.confidence)} variant="outlined" />
                <Chip
                  label={c.configStatus === 'ready' ? '✓ config ready' : '⏳ ' + (c.configStatus || 'needs_generation')}
                  size="small"
                  color={c.configStatus === 'ready' ? 'success' : 'warning'}
                  variant="outlined"
                />
                <Chip
                  label={c.constraintCheck === 'PASSED' ? '✓ slot check' : '✗ slot ' + c.constraintCheck}
                  size="small"
                  color={c.constraintCheck === 'PASSED' ? 'success' : 'error'}
                  variant="outlined"
                />
                {c.estimatedComputeHours && (
                  <Typography variant="caption" color="text.secondary">
                    ~{c.estimatedComputeHours}h
                  </Typography>
                )}
              </Box>
              {c.rationale && (
                <Typography variant="body2" sx={{ fontSize: '0.78rem', mb: 0.5 }}>
                  {c.rationale}
                </Typography>
              )}
              {c.risk && (
                <Box sx={{ mt: 0.5 }}>
                  <Button
                    size="small"
                    sx={{ textTransform: 'none', fontSize: '0.72rem', p: 0 }}
                    onClick={() => setExpandedRisk(prev => ({ ...prev, [c.comboId]: !prev[c.comboId] }))}
                  >
                    {expandedRisk[c.comboId] ? 'Hide risk' : 'Show risk'}
                  </Button>
                  <Collapse in={!!expandedRisk[c.comboId]}>
                    <Typography variant="caption" sx={{ display: 'block', color: 'warning.light', mt: 0.5, fontSize: '0.74rem' }}>
                      <strong>Risk: </strong>{c.risk}
                    </Typography>
                  </Collapse>
                </Box>
              )}
              {c.configPathProposed && (
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5, fontFamily: 'monospace', fontSize: '0.7rem' }}>
                  config: {c.configPathProposed}
                </Typography>
              )}
            </CardContent>
          </Card>
        );
      })}
    </Box>
  );
}

function ApplyChangesView({
  actions, selectedRerank, selectedCombos, sessionId, multiTaskId, taskSubmissions,
  onApplied, addSubmission, implementations, viewedArchiveId, onJumpToSelection,
}) {
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  const [confirmRerank, setConfirmRerank] = useState(false);
  const [confirmCombos, setConfirmCombos] = useState(false);

  const log = actions?.applied_changes_log || [];
  const lastApply = [...log].reverse().find(e => e.action === 'apply_rerank');

  const rerankCount = selectedRerank.size;
  const selectedComboObjects = useMemo(() => {
    return (actions?.newCombos || [])
      .filter(c => selectedCombos.has(c.comboId));
  }, [actions, selectedCombos]);
  const { eligible: eligibleSelectedCombos, pending: pendingSelectedCombos } = useMemo(
    () => partitionCombos(selectedComboObjects, implementations),
    [selectedComboObjects, implementations],
  );
  const eligibleComboCount = eligibleSelectedCombos.length;
  const pendingComboCount = pendingSelectedCombos.length;

  async function handleApplyRerank({ skipConfirm = false } = {}) {
    if (!skipConfirm && !confirmRerank) {
      setConfirmRerank(true);
      setTimeout(() => setConfirmRerank(false), 5000);
      return;
    }
    setBusy('rerank');
    setError(null);
    try {
      const rankings = (actions.hypothesisRerank || [])
        .filter(r => selectedRerank.has(r.id))
        .map(r => ({
          id: r.id,
          oldRank: r.oldRank,
          newRank: r.newRank,
          rationale: r.rationale,
          confidence: r.confidence,
        }));
      await postRerankApply(sessionId, rankings, actions.deprioritize || []);
      setConfirmRerank(false);
      onApplied();
    } catch (e) {
      setError(`Apply rerank failed: ${e.message}`);
      throw e;
    } finally {
      setBusy(null);
    }
  }

  async function handleApplyCombos({ skipConfirm = false } = {}) {
    if (!skipConfirm && !confirmCombos) {
      setConfirmCombos(true);
      setTimeout(() => setConfirmCombos(false), 5000);
      return;
    }
    if (selectedComboObjects.length === 0) {
      setError('No combos selected.');
      setConfirmCombos(false);
      return;
    }
    setBusy('combos');
    setError(null);
    try {
      await postCombosApply(
        sessionId, multiTaskId, selectedComboObjects,
        viewedArchiveId || null, 'ui',
      );
      setConfirmCombos(false);
      onApplied();
    } catch (e) {
      setError(`Apply combos failed: ${e.message}`);
      throw e;
    } finally {
      setBusy(null);
    }
  }

  async function handleApplyAll() {
    setBusy('all');
    setError(null);
    let stepLabel = 'reranks';
    let fullSuccess = true;
    try {
      if (rerankCount > 0) {
        stepLabel = 'reranks (1/2)';
        setBusy('all-rerank');
        await handleApplyRerank({ skipConfirm: true });
      }
      if (selectedComboObjects.length > 0) {
        stepLabel = 'combos (2/2)';
        setBusy('all-combos');
        await handleApplyCombos({ skipConfirm: true });
      }
    } catch (e) {
      fullSuccess = false;
      setError(`Apply All failed at ${stepLabel}: ${e.message}. Retry the failed step from the section below.`);
    } finally {
      setBusy(null);
    }
    if (fullSuccess && typeof onJumpToSelection === 'function') {
      onJumpToSelection();
    }
  }

  async function handleRevertCombos() {
    setBusy('revertCombos');
    setError(null);
    try {
      const url = `/api/sessions/${encodeURIComponent(sessionId)}`
        + `/combo_overrides/${encodeURIComponent(multiTaskId)}/revert_last`;
      const res = await fetch(url, { method: 'POST' });
      if (!res.ok) {
        if (res.status === 404) {
          setError('Nothing to revert (no apply history for this hub).');
          return;
        }
        throw new Error(`HTTP ${res.status}`);
      }
      window.dispatchEvent(new CustomEvent('combo_overrides_changed', {
        detail: { multi_task_id: multiTaskId, action: 'revert' },
      }));
      onApplied();
    } catch (e) {
      setError(`Revert combos failed: ${e.message}`);
    } finally {
      setBusy(null);
    }
  }

  async function handleRevertLast() {
    setBusy('revert');
    setError(null);
    try {
      const res = await fetch(
        `/api/sessions/${sessionId}/proposal_overrides/revert_last`,
        { method: 'POST' }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (!data.ok) {
        setError(data.reason || 'No-op revert');
      } else {
        window.dispatchEvent(new CustomEvent('proposal_overrides_applied'));
        onApplied();
      }
    } catch (e) {
      setError(`Revert failed: ${e.message}`);
    } finally {
      setBusy(null);
    }
  }

  const applyAllBusy = busy === 'all' || busy === 'all-rerank' || busy === 'all-combos';
  const applyAllProgress = busy === 'all-rerank' ? 'reranks (1/2)'
    : busy === 'all-combos' ? 'combos (2/2)'
    : '';

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {error && <Alert severity="error" onClose={() => setError(null)}>{error}</Alert>}

      <Box sx={{ p: 2, bgcolor: 'rgba(25,118,210,0.08)', borderRadius: 1 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>
          🚀 Apply All
        </Typography>
        <Typography variant="body2" sx={{ fontSize: '0.78rem', mb: 1.25, opacity: 0.85 }}>
          Apply all {rerankCount} rerank{rerankCount === 1 ? '' : 's'} AND all{' '}
          {selectedComboObjects.length} selected combo{selectedComboObjects.length === 1 ? '' : 's'}{' '}
          in one click. Selection tab gets new ranks + auto-selects hypotheses
          from all applied combos (including pending). Review & Combos shows
          combos with un-implemented hypotheses greyed-out with a "Pending
          implementation" / "Pending config" label until prerequisites land.
          {pendingComboCount > 0 && (
            <> {pendingComboCount} of the selected combos are pending — they'll be persisted but
              not auto-create runnable rows until prerequisites are met.</>
          )}
        </Typography>
        <Button
          variant="contained"
          color="primary"
          disabled={busy !== null || (rerankCount === 0 && selectedComboObjects.length === 0)}
          onClick={handleApplyAll}
          sx={{ textTransform: 'none', minWidth: 220 }}
        >
          {applyAllBusy
            ? `Applying… ${applyAllProgress}`
            : `🚀 Apply All (${rerankCount} rerank${rerankCount === 1 ? '' : 's'} + ${selectedComboObjects.length} combo${selectedComboObjects.length === 1 ? '' : 's'})`}
        </Button>
      </Box>

      <Divider />

      <Box>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>1. Apply hypothesis re-ranking</Typography>
        <Typography variant="body2" sx={{ fontSize: '0.78rem', mb: 1, opacity: 0.85 }}>
          Will write {rerankCount} rank override{rerankCount === 1 ? '' : 's'} to{' '}
          <code>{`<session>/proposal_overrides.json`}</code>. Selection-tab cards
          re-render with new ranks. NEVER mutates session_state.json. Reversible.
        </Typography>
        <Button
          variant="contained"
          color={confirmRerank ? 'warning' : 'primary'}
          disabled={busy !== null || rerankCount === 0}
          onClick={handleApplyRerank}
          sx={{ textTransform: 'none' }}
        >
          {busy === 'rerank'
            ? 'Applying…'
            : confirmRerank
              ? `Confirm: apply ${rerankCount} rerank${rerankCount === 1 ? '' : 's'}`
              : `Apply ${rerankCount} rerank${rerankCount === 1 ? '' : 's'}`}
        </Button>
      </Box>

      <Divider />

      <Box>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>2. Apply Combos</Typography>
        <Typography variant="body2" sx={{ fontSize: '0.78rem', mb: 1, opacity: 0.85 }}>
          Replaces the active combos for this hub with {selectedComboObjects.length} selected
          combo{selectedComboObjects.length === 1 ? '' : 's'}. The Selection tab pre-narrows
          to the union of their hypotheses (including pending combos' hypotheses); Review &amp; Combos
          shows them as the active set (prior combos move to Historical, runs preserved). Reversible.
          Writes to <code>{`<session>/hub/${multiTaskId}/combos/current.json`}</code>.
          {' '}{eligibleComboCount} ready combo{eligibleComboCount === 1 ? '' : 's'} will auto-create
          submission rows; {pendingComboCount > 0
            ? `${pendingComboCount} pending combo${pendingComboCount === 1 ? '' : 's'} will be persisted with their applyState (greyed-out in Review & Combos until prerequisites land).`
            : 'no pending combos in selection.'}
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <Button
            variant="contained"
            color={confirmCombos ? 'warning' : 'primary'}
            disabled={busy !== null || selectedComboObjects.length === 0}
            onClick={handleApplyCombos}
            sx={{ textTransform: 'none' }}
          >
            {busy === 'combos'
              ? 'Applying…'
              : confirmCombos
                ? `Confirm: apply ${selectedComboObjects.length} combo${selectedComboObjects.length === 1 ? '' : 's'}`
                : `Apply ${selectedComboObjects.length} combo${selectedComboObjects.length === 1 ? '' : 's'}`}
          </Button>
          <Button
            size="small"
            startIcon={<UndoIcon />}
            onClick={handleRevertCombos}
            disabled={busy !== null}
            sx={{ textTransform: 'none' }}
          >
            {busy === 'revertCombos' ? 'Reverting…' : 'Revert combos'}
          </Button>
        </Box>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
          Note: This persists submissions only. Use the per-row Run button on the
          Experiments tab to actually launch a training run.
        </Typography>
      </Box>

      <Divider />

      <Box>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>3. Applied-changes log</Typography>
        {log.length === 0 ? (
          <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.78rem' }}>
            No applies yet for this session.
          </Typography>
        ) : (
          <Box>
            {log.slice().reverse().map((entry, i) => (
              <Box key={i} sx={{ mb: 0.5, fontSize: '0.74rem', fontFamily: 'monospace' }}>
                <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                  {entry.appliedAt} · <strong>{entry.action}</strong>
                  {entry.affected_h_ids?.length > 0 && ` · H: ${entry.affected_h_ids.join(',')}`}
                  {entry.reverted_apply_at && ` · undid: ${entry.reverted_apply_at}`}
                </Typography>
              </Box>
            ))}
            {lastApply && (
              <Button
                size="small"
                startIcon={<UndoIcon />}
                onClick={handleRevertLast}
                disabled={busy !== null}
                sx={{ textTransform: 'none', mt: 1 }}
              >
                {busy === 'revert' ? 'Reverting…' : 'Revert last apply'}
              </Button>
            )}
          </Box>
        )}
      </Box>
    </Box>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────

export default function AccumulatedLearningsDrawer({
  open, onClose, multiTaskId, taskSubmissions, onJumpToSelection,
  // Injected context (was useSession()):
  sessionId, learningsVersion = 0, addSubmission,
}) {
  const { implementations } = useHypothesisImplementations(sessionId, multiTaskId);

  const [data, setData] = useState({ markdownBody: '', actions: null, exists: false, lastModified: null });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [selectedRerank, setSelectedRerank] = useState(new Set());
  const [selectedCombos, setSelectedCombos] = useState(new Set());
  const [archives, setArchives] = useState([]);
  const [viewedArchiveId, setViewedArchiveId] = useState(null);
  const [archiveBody, setArchiveBody] = useState(null);
  const [archiveLoading, setArchiveLoading] = useState(false);

  const loadDoc = useCallback(async () => {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/sessions/${sessionId}/learnings`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
      const r = json?.actions?.hypothesisRerank || [];
      setSelectedRerank(new Set(r.map(x => x.id)));
      const c = json?.actions?.newCombos || [];
      const alreadyIds = new Set((taskSubmissions || [])
        .map(s => s?.config?.future_combo_id).filter(Boolean));
      setSelectedCombos(new Set(
        c.filter(x => !alreadyIds.has(x.comboId)).map(x => x.comboId)
      ));
    } catch (e) {
      setError(`Load failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }, [sessionId, taskSubmissions]);

  useEffect(() => {
    if (open) loadDoc();
  }, [open, loadDoc, learningsVersion]);

  useEffect(() => {
    if (!open || !sessionId) return undefined;
    let cancelled = false;
    fetch(`/api/sessions/${encodeURIComponent(sessionId)}/learnings/archives`)
      .then((res) => (res.ok ? res.json() : { archives: [] }))
      .then((body) => {
        if (cancelled) return;
        setArchives(Array.isArray(body?.archives) ? body.archives : []);
      })
      .catch(() => { if (!cancelled) setArchives([]); });
    return () => { cancelled = true; };
  }, [open, sessionId, data.lastModified, learningsVersion]);

  useEffect(() => {
    if (!viewedArchiveId || !sessionId) {
      setArchiveBody(null);
      return undefined;
    }
    let cancelled = false;
    setArchiveLoading(true);
    fetch(`/api/sessions/${encodeURIComponent(sessionId)}/learnings/archives/${encodeURIComponent(viewedArchiveId)}`)
      .then((res) => (res.ok ? res.text() : null))
      .then((body) => {
        if (cancelled) return;
        setArchiveBody(body);
      })
      .catch(() => { if (!cancelled) setArchiveBody(null); })
      .finally(() => { if (!cancelled) setArchiveLoading(false); });
    return () => { cancelled = true; };
  }, [viewedArchiveId, sessionId]);

  // G9 — real restore-from-archive wire (backend at
  // OT-R/learnings_routes.py:restore_learnings_archive). Env-gated at the AF
  // service (`RANKEVOLVE_LEARNINGS_RESTORE_ENABLED=1`) — a 501 response
  // surfaces the gate to the user; a 200 refreshes the current doc + archive
  // list; anything else shows the raw error. Not a stub anymore.
  const handleRestoreFromArchive = useCallback(async (archiveId) => {
    if (!sessionId || !archiveId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `/api/sessions/${encodeURIComponent(sessionId)}`
          + `/learnings/restore/${encodeURIComponent(archiveId)}`,
        { method: 'POST' },
      );
      if (res.status === 501) {
        setError(
          'Restore is feature-gated — set '
          + 'RANKEVOLVE_LEARNINGS_RESTORE_ENABLED=1 on the server to enable.',
        );
        return;
      }
      if (!res.ok) {
        const detail = await res.text().catch(() => '');
        throw new Error(`HTTP ${res.status}${detail ? `: ${detail}` : ''}`);
      }
      // Refresh both the current doc and the archive index. The AF handler
      // snapshots current→archive before overwriting, so the archive list
      // grows by one entry too.
      await loadDoc();
      try {
        const archRes = await fetch(
          `/api/sessions/${encodeURIComponent(sessionId)}/learnings/archives`,
        );
        if (archRes.ok) {
          const body = await archRes.json();
          setArchives(Array.isArray(body?.archives) ? body.archives : []);
        }
      } catch {
        /* archive-list refresh is best-effort; useEffect on lastModified will
           also re-fetch on the next tick */
      }
      setViewedArchiveId(null);
    } catch (e) {
      setError(`Restore failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }, [sessionId, loadDoc]);
  // The existing JSX (Restore IconButton at ~:837 + Return-to-current at
  // ~:891) is wired to the old `handleRestoreStub` param name. We wire our
  // real handler under the same identifier so the JSX doesn't need changes
  // beyond the caller passing the archive_id.
  const handleRestoreStub = useCallback(() => {
    if (viewedArchiveId) {
      void handleRestoreFromArchive(viewedArchiveId);
    }
  }, [viewedArchiveId, handleRestoreFromArchive]);

  const regenerate = useCallback(async () => {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `/api/sessions/${sessionId}/learnings/regenerate`,
        { method: 'POST' }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
    } catch (e) {
      setError(`Regenerate failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  const a = data.actions;
  const rerankCount = a?.hypothesisRerank?.length ?? 0;
  const recCount = a?.newCombos?.length ?? 0;

  const toggleRerank = (id) => setSelectedRerank(prev => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });
  const toggleCombo = (id) => setSelectedCombos(prev => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });
  const selectAllRerank = () => setSelectedRerank(new Set((a?.hypothesisRerank || []).map(r => r.id)));
  const clearAllRerank = () => setSelectedRerank(new Set());
  const selectAllCombos = () => {
    const alreadyIds = new Set((taskSubmissions || [])
      .map(s => s?.config?.future_combo_id).filter(Boolean));
    setSelectedCombos(new Set(
      (a?.newCombos || [])
        .filter(x => !alreadyIds.has(x.comboId))
        .map(x => x.comboId)
    ));
  };
  const clearAllCombos = () => setSelectedCombos(new Set());

  return (
    <Drawer
      anchor="right"
      open={open}
      onClose={onClose}
      PaperProps={{ sx: { width: { xs: '100%', md: 760, lg: 920 } } }}
    >
      <AppBar position="static" color="default" elevation={0}>
        <Toolbar sx={{ minHeight: 48 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            📊 Accumulated Learnings
          </Typography>
          {a?.summary && (
            <Typography variant="caption" sx={{ ml: 2, opacity: 0.7 }}>
              {a.summary.totalExperiments} exp · {a.summary.convergedCount} converged ·{' '}
              {a.summary.earlyKilledCount} early-killed · {a.summary.failedCount} failed
            </Typography>
          )}
          <Box sx={{ flex: 1 }} />
          {data.lastModified && (
            <Tooltip title={new Date(data.lastModified * 1000).toLocaleString()}>
              <Typography variant="caption" sx={{ mr: 1, opacity: 0.6 }}>
                {`updated ${Math.max(1, Math.round((Date.now()/1000 - data.lastModified)/60))}m ago`}
              </Typography>
            </Tooltip>
          )}
          {archives.length > 0 && (
            <Tooltip title={`${archives.length} archived version${archives.length === 1 ? '' : 's'}`}>
              <Select
                size="small"
                value={viewedArchiveId || 'LIVE'}
                onChange={(e) => setViewedArchiveId(e.target.value === 'LIVE' ? null : e.target.value)}
                sx={{ mr: 1, minWidth: 180, fontSize: '0.78rem',
                      '& .MuiSelect-select': { py: 0.5 } }}
              >
                <MenuItem value="LIVE">Current (live)</MenuItem>
                {archives.map((ar) => (
                  <MenuItem key={ar.archive_id} value={ar.archive_id}>
                    {`v${ar.version} · ${ar.archived_at?.slice(0, 16) || ar.archive_id} · ${ar.source || ar.reason || ''}`}
                  </MenuItem>
                ))}
              </Select>
            </Tooltip>
          )}
          {viewedArchiveId && (
            <Tooltip title="Restore this archive — coming soon">
              <IconButton size="small" onClick={handleRestoreStub} sx={{ mr: 0.5 }}>
                <RestoreIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
          <Tooltip title="Recompute (algorithmic)">
            <span>
              <IconButton onClick={regenerate} disabled={loading || !sessionId}>
                <RefreshIcon />
              </IconButton>
            </span>
          </Tooltip>
          <IconButton onClick={onClose}><CloseIcon /></IconButton>
        </Toolbar>
        <Tabs
          value={activeTab}
          onChange={(_, v) => setActiveTab(v)}
          sx={{ minHeight: 36, '& .MuiTab-root': { minHeight: 36, textTransform: 'none', fontSize: '0.82rem' } }}
        >
          <Tab label="Full doc" />
          <Tab label={`Re-ranking (${rerankCount})`} />
          <Tab label={`Combos (${recCount})`} />
          <Tab label="Apply" />
        </Tabs>
      </AppBar>
      <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
        {error && <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>{error}</Alert>}
        {loading && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <CircularProgress size={16} />
            <Typography variant="caption">Loading…</Typography>
          </Box>
        )}
        {!loading && !data.exists && (
          <Box sx={{ textAlign: 'center', py: 6 }}>
            <Typography color="text.secondary" sx={{ mb: 2 }}>
              No accumulated learnings doc yet for this session.
            </Typography>
            <Button variant="contained" onClick={regenerate} disabled={!sessionId}>
              Generate now
            </Button>
          </Box>
        )}
        {viewedArchiveId && activeTab === 0 && (
          <Box sx={{ mb: 2 }}>
            <Alert
              severity="info"
              action={
                <Button size="small" onClick={() => setViewedArchiveId(null)}>
                  ↩ Return to current
                </Button>
              }
            >
              {(() => {
                const ar = archives.find((x) => x.archive_id === viewedArchiveId);
                return ar
                  ? `Viewing archived v${ar.version} from ${ar.archived_at} · source: ${ar.source || ar.reason || 'n/a'}`
                  : `Viewing archived ${viewedArchiveId}`;
              })()}
            </Alert>
            {archiveLoading ? (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 2 }}>
                <CircularProgress size={16} />
                <Typography variant="caption">Loading archive…</Typography>
              </Box>
            ) : archiveBody ? (
              <FullDocView markdownBody={archiveBody} />
            ) : (
              <Typography color="text.secondary">Archive body unavailable.</Typography>
            )}
          </Box>
        )}
        {!viewedArchiveId && data.exists && (
          <>
            {activeTab === 0 && <FullDocView markdownBody={data.markdownBody} />}
            {activeTab === 1 && (
              <HypothesisRerankView
                entries={a?.hypothesisRerank || []}
                selectedIds={selectedRerank}
                onToggle={toggleRerank}
                onSelectAll={selectAllRerank}
                onClearAll={clearAllRerank}
              />
            )}
            {activeTab === 2 && (
              <ComboRecommendationsView
                combos={a?.newCombos || []}
                taskSubmissions={taskSubmissions}
                selectedIds={selectedCombos}
                onToggle={toggleCombo}
                onSelectAll={selectAllCombos}
                onClearAll={clearAllCombos}
                implementations={implementations}
              />
            )}
            {activeTab === 3 && (
              <ApplyChangesView
                actions={a}
                selectedRerank={selectedRerank}
                selectedCombos={selectedCombos}
                sessionId={sessionId}
                multiTaskId={multiTaskId}
                taskSubmissions={taskSubmissions}
                addSubmission={addSubmission}
                onApplied={loadDoc}
                implementations={implementations}
                viewedArchiveId={viewedArchiveId}
                onJumpToSelection={onJumpToSelection}
              />
            )}
          </>
        )}
      </Box>
    </Drawer>
  );
}
