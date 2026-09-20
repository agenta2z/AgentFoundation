/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * MultiChoiceComboView — Tabbed multi-choice selection with combo submission.
 * Groups items by batchId, allows cross-tab selection, shows combo hints, and
 * supports a detail drawer.
 *
 * Ported from RankEvolve (components/views/MultiChoiceComboView.js). The
 * `useSession()` reads become injected props: `sessionId`, `dispatch`,
 * `apiClient` ({ runExperimentCombos }). View switches use the generic
 * reducer's `SET_ACTIVE_VIEW {viewIndex}` action. The combo helpers come from
 * the local `multiTaskHelpers` util; sidecar reads stay direct-fetch.
 */

import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import {
  Alert, Box, Button, Card, CardActionArea, Checkbox, Chip,
  Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle,
  Tab, Tabs, TextField, Tooltip, Typography,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import {
  findMatchingSubmissions, deriveStatusMap,
  computeBlockedIds, checkComboConstraints,
} from '../../utils/multiTaskHelpers';
import HypothesisDetailDrawer from './HypothesisDetailDrawer';
import SubmissionFooterBar from '../hub/SubmissionFooterBar';
import ComboReviewSection from './ComboReviewSection';
import HowCombosWork from '../hub/HowCombosWork';
import { useProposalOverrides } from '../../hooks/useProposalOverrides';
import { useComboOverrides } from '../../hooks/useComboOverrides';
import { mergeOverrides } from '../../utils/mergeOverrides';
import { postCombosApply } from '../../utils/applyHelpers';

export default function MultiChoiceComboView({
  task, config, dispatch, sessionId, apiClient,
}) {
  const runExperimentCombos = apiClient && apiClient.runExperimentCombos;
  const { overrides } = useProposalOverrides(sessionId);
  const { activeCombos, generatedAt: combosAppliedAt, flagMapWarnings } = useComboOverrides(
    sessionId, task.id,
  );
  const [bannerDismissed, setBannerDismissed] = useState(false);
  const openSetupWizard = useCallback(() => {
    try {
      window.dispatchEvent(new CustomEvent('open_hub_setup_wizard', {
        detail: { multi_task_id: task.id },
      }));
    } catch (e) {
      console.warn('[MultiChoiceComboView] openSetupWizard dispatch failed:', e);
    }
  }, [task.id]);
  const activeHypothesisIds = useMemo(() => {
    const s = new Set();
    for (const c of activeCombos || []) {
      for (const h of c.selectedItems || []) {
        if (typeof h === 'string' && h) s.add(h);
      }
    }
    return s;
  }, [activeCombos]);
  const [activeTab, setActiveTab] = useState(0);
  const [selectedIds, setSelectedIds] = useState(() => new Set(activeHypothesisIds));
  const [userTouched, setUserTouched] = useState(false);
  useEffect(() => {
    if (userTouched) return;
    setSelectedIds(new Set(activeHypothesisIds));
  }, [activeHypothesisIds, userTouched]);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerHypothesisId, setDrawerHypothesisId] = useState(null);
  const [maxConcurrency, setMaxConcurrency] = useState(() => {
    try {
      const v = parseInt(window.localStorage.getItem('rankevolve_combo_max_concurrency') || '', 10);
      return Number.isFinite(v) && v > 0 ? v : 2;
    } catch (e) { return 2; }
  });

  const [selectedComboIds, setSelectedComboIds] = useState(() => new Set());
  const [expandedComboIds, setExpandedComboIds] = useState(() => new Set());
  const [removeConfirmComboId, setRemoveConfirmComboId] = useState(null);
  const [addComboBusy, setAddComboBusy] = useState(false);
  const [submitBusy, setSubmitBusy] = useState(false);
  const [comboError, setComboError] = useState(null);
  const [freeTextHs, setFreeTextHs] = useState('');

  const prevComboIdsRef = useRef(new Set());
  useEffect(() => {
    const currentIds = new Set((activeCombos || []).map(c => c.comboId).filter(Boolean));
    const prev = prevComboIdsRef.current;
    const newOnes = [];
    for (const id of currentIds) if (!prev.has(id)) newOnes.push(id);
    setSelectedComboIds(prevSel => {
      let changed = false;
      const next = new Set();
      for (const id of prevSel) {
        if (currentIds.has(id)) next.add(id);  // prune removed
        else changed = true;
      }
      for (const id of newOnes) {
        if (!next.has(id)) { next.add(id); changed = true; }  // auto-check new
      }
      return changed ? next : prevSel;
    });
    setExpandedComboIds(prevExp => {
      let changed = false;
      const next = new Set();
      for (const id of prevExp) {
        if (currentIds.has(id)) next.add(id);
        else changed = true;
      }
      return changed ? next : prevExp;
    });
    prevComboIdsRef.current = currentIds;
  }, [activeCombos]);

  const handleToggleCombo = useCallback((comboId) => {
    setSelectedComboIds(prev => {
      const next = new Set(prev);
      if (next.has(comboId)) next.delete(comboId);
      else next.add(comboId);
      return next;
    });
  }, []);
  const handleToggleExpandCombo = useCallback((comboId) => {
    setExpandedComboIds(prev => {
      const next = new Set(prev);
      if (next.has(comboId)) next.delete(comboId);
      else next.add(comboId);
      return next;
    });
  }, []);
  const handleSelectAllCombos = useCallback(() => {
    setSelectedComboIds(new Set((activeCombos || []).map(c => c.comboId).filter(Boolean)));
  }, [activeCombos]);
  const handleClearSelectedCombos = useCallback(() => {
    setSelectedComboIds(new Set());
  }, []);
  const handleExpandAllCombos = useCallback(() => {
    setExpandedComboIds(new Set((activeCombos || []).map(c => c.comboId).filter(Boolean)));
  }, [activeCombos]);
  const handleCollapseAllCombos = useCallback(() => {
    setExpandedComboIds(new Set());
  }, []);

  const handleRequestRemoveCombo = useCallback((comboId) => {
    setRemoveConfirmComboId(comboId);
  }, []);
  const handleConfirmRemoveCombo = useCallback(async () => {
    const comboId = removeConfirmComboId;
    if (!comboId) return;
    setComboError(null);
    try {
      const remaining = (activeCombos || []).filter(c => c.comboId !== comboId);
      await postCombosApply(
        sessionId, task.id, remaining, null, 'ui:remove',
      );
    } catch (e) {
      setComboError(`Remove combo ${comboId} failed: ${e.message}`);
    } finally {
      setRemoveConfirmComboId(null);
    }
  }, [removeConfirmComboId, activeCombos, sessionId, task.id]);
  const handleCancelRemoveCombo = useCallback(() => {
    setRemoveConfirmComboId(null);
  }, []);

  const parseFreeTextHs = useCallback((raw) => {
    if (!raw) return [];
    const seen = new Set();
    const out = [];
    for (const tok of String(raw).split(/[,\s;]+/)) {
      const t = tok.trim().toUpperCase();
      if (!/^H\d+$/.test(t)) continue;
      if (seen.has(t)) continue;
      seen.add(t);
      out.push(t);
    }
    return out;
  }, []);

  const freeTextParsedHs = useMemo(
    () => parseFreeTextHs(freeTextHs),
    [freeTextHs, parseFreeTextHs],
  );

  const composedHs = useMemo(() => {
    const set = new Set();
    for (const h of selectedIds) {
      if (typeof h === 'string' && h) set.add(h.toUpperCase());
    }
    for (const h of freeTextParsedHs) set.add(h);
    return Array.from(set).sort((a, b) => {
      const an = parseInt(String(a).replace(/^H/i, ''), 10);
      const bn = parseInt(String(b).replace(/^H/i, ''), 10);
      if (Number.isFinite(an) && Number.isFinite(bn) && an !== bn) return an - bn;
      return String(a).localeCompare(String(b));
    });
  }, [selectedIds, freeTextParsedHs]);

  const nextManId = useCallback((combos) => {
    const max = (combos || [])
      .map(c => c?.comboId || '')
      .filter(id => /^MAN-\d+$/.test(id))
      .reduce((m, id) => Math.max(m, parseInt(id.slice(4), 10)), 0);
    return `MAN-${max + 1}`;
  }, []);
  const handleAddCombo = useCallback(async () => {
    setComboError(null);
    const items = composedHs;
    if (items.length < 2) return;
    const newKey = items.join('_');
    const dup = (activeCombos || []).find(c => c?.comboKey === newKey);
    if (dup) {
      setComboError(`This combo already exists as ${dup.comboId}.`);
      return;
    }
    setAddComboBusy(true);
    try {
      const newCombo = {
        comboId: nextManId(activeCombos),
        comboKey: newKey,
        selectedItems: items,
        title: `Manual combo (${items.join(' + ')})`,
        rationale: '(user-composed)',
        configStatus: 'needs_generation',
        configPathProposed: `hstu-${items.map(h => h.toLowerCase()).join('-')}.gin (NEW)`,
      };
      await postCombosApply(
        sessionId, task.id,
        [...(activeCombos || []), newCombo],
        null, 'ui:add',
      );
      setSelectedIds(new Set());
      setFreeTextHs('');
    } catch (e) {
      setComboError(`Add combo failed: ${e.message}`);
    } finally {
      setAddComboBusy(false);
    }
  }, [composedHs, activeCombos, sessionId, task.id, nextManId]);

  const selectedCombos = useMemo(
    () => (activeCombos || []).filter(c => selectedComboIds.has(c?.comboId)),
    [activeCombos, selectedComboIds],
  );
  const handleDirectComboSubmit = useCallback(() => {
    try {
      window.localStorage.setItem('rankevolve_combo_max_concurrency', String(maxConcurrency));
    } catch (e) { /* ignore */ }
    if (typeof runExperimentCombos !== 'function') {
      console.warn('[MultiChoiceComboView] runExperimentCombos unavailable');
      return;
    }
    setSubmitBusy(true);
    try {
      if (selectedCombos.length > 0) {
        const combosArg = selectedCombos.map(c => c?.selectedItems || []).filter(a => a.length > 0);
        if (combosArg.length === 0) return;
        runExperimentCombos(task.id, {
          combos: combosArg,
          maxConcurrency,
        });
      } else if (selectedIds.size > 0) {
        runExperimentCombos(task.id, {
          combos: [[...selectedIds]],
          maxConcurrency,
        });
      }
    } finally {
      setSubmitBusy(false);
    }
  }, [selectedCombos, selectedIds, maxConcurrency, runExperimentCombos, task.id]);

  const statusMap = useMemo(() => deriveStatusMap(task.runQueue), [task.runQueue]);

  const batchGroups = useMemo(() => {
    const groups = {};
    (task.runQueue || []).forEach(entry => {
      const batchId = entry.metadata?.batchId || 'misc';
      if (!groups[batchId]) {
        groups[batchId] = {
          batchId,
          batchLabel: entry.metadata?.batchLabel || batchId,
          hypotheses: [],
          status: entry.status,
        };
      }
      (entry.metadata?.hypothesisIds || []).forEach(hid => {
        groups[batchId].hypotheses.push({
          id: hid,
          status: statusMap[hid] || 'queued',
          batchId,
        });
      });
    });
    return Object.values(groups);
  }, [task.runQueue, statusMap]);

  const proposalDetails = useMemo(() => {
    const snapshot = task.scenarioState?.selectionSnapshot || {};
    const baseProposals = snapshot.widgetConfig?.proposals || snapshot.proposals || {};
    const merged = mergeOverrides(baseProposals, overrides);
    const detailMap = {};
    (merged.phases || []).forEach(phase => {
      (phase.proposals || []).forEach(p => {
        detailMap[p.id] = p;
      });
    });
    return detailMap;
  }, [task.scenarioState, overrides]);

  const toggleSelect = useCallback((id) => {
    setUserTouched(true);
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const selectAllCompleted = useCallback(() => {
    setUserTouched(true);
    const completed = Object.entries(statusMap)
      .filter(([, s]) => s === 'completed')
      .map(([id]) => id);
    setSelectedIds(new Set(completed));
  }, [statusMap]);

  const clearSelection = useCallback(() => {
    setUserTouched(true);
    setSelectedIds(new Set());
  }, []);

  const resetToActiveCombos = useCallback(() => {
    setUserTouched(false);
    setSelectedIds(new Set(activeHypothesisIds));
  }, [activeHypothesisIds]);

  const handleViewDetails = useCallback((hypothesisId) => {
    if (config.detailDrawer) {
      setDrawerHypothesisId(hypothesisId);
      setDrawerOpen(true);
    }
  }, [config.detailDrawer]);

  const comboHint = useMemo(() => {
    if (!config.showComboHints || selectedIds.size === 0) return null;
    return findMatchingSubmissions([...selectedIds], task.submissions);
  }, [selectedIds, task.submissions, config.showComboHints]);

  const allProposals = useMemo(
    () => Object.values(proposalDetails),
    [proposalDetails],
  );
  const comboConstraints = useMemo(() => {
    const snap = task.scenarioState?.selectionSnapshot || {};
    const proposals = snap.widgetConfig?.proposals || snap.proposals || {};
    return proposals.combo_constraints || [];
  }, [task.scenarioState]);
  const { blocked, includedBy } = useMemo(
    () => computeBlockedIds(selectedIds, allProposals),
    [selectedIds, allProposals],
  );
  const constraintViolations = useMemo(
    () => checkComboConstraints(selectedIds, comboConstraints),
    [selectedIds, comboConstraints],
  );
  const submitDisabled = useMemo(() => {
    if (selectedIds.size === 0) return true;
    if (constraintViolations.errors.length > 0) return true;
    for (const id of selectedIds) {
      if (blocked.has(id)) return true;
    }
    return false;
  }, [selectedIds, constraintViolations, blocked]);

  const activeGroup = batchGroups[activeTab];

  return (
    <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <Dialog
        open={removeConfirmComboId !== null}
        onClose={handleCancelRemoveCombo}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>Remove combo from active set?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Remove <strong>{removeConfirmComboId}</strong> from the active list?
            Any submission rows for this combo will be marked inactive but
            not deleted. Running runs are not cancelled — use the per-row
            Cancel button if needed.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCancelRemoveCombo} sx={{ textTransform: 'none' }}>Cancel</Button>
          <Button onClick={handleConfirmRemoveCombo} color="error" sx={{ textTransform: 'none' }}>
            Remove
          </Button>
        </DialogActions>
      </Dialog>

      <Tabs
        value={activeTab}
        onChange={(_, v) => setActiveTab(v)}
        variant="scrollable"
        scrollButtons="auto"
        sx={{
          borderBottom: '1px solid rgba(255,255,255,0.08)',
          '& .MuiTab-root': { textTransform: 'none', fontSize: '0.82rem', minHeight: 36 },
        }}
      >
        {batchGroups.map((group) => {
          const completedCount = group.hypotheses.filter(h => h.status === 'completed').length;
          return (
            <Tab key={group.batchId} label={`B${group.batchId} (${completedCount}/${group.hypotheses.length})`} />
          );
        })}
      </Tabs>

      <Box sx={{ mt: 1 }}>
        <HowCombosWork />
      </Box>

      {activeHypothesisIds.size > 0 && (() => {
        const readyCombos = (activeCombos || []).filter(
          c => (c?.applyState || 'ready') === 'ready'
        );
        const pendingCombos = (activeCombos || []).filter(
          c => (c?.applyState || 'ready') !== 'ready'
        );
        const compositionParts = [];
        if (readyCombos.length > 0) {
          compositionParts.push(`${readyCombos.length} active`);
        }
        if (pendingCombos.length > 0) {
          compositionParts.push(`${pendingCombos.length} pending`);
        }
        const composition = compositionParts.join(' + ');
        return (
          <Box sx={{ px: 2, pt: 1.5 }}>
            <Alert
              severity="info"
              sx={{ mb: 0.5, fontSize: '0.78rem' }}
              action={
                userTouched ? (
                  <Button size="small" onClick={resetToActiveCombos} sx={{ textTransform: 'none' }}>
                    Reset to active combos
                  </Button>
                ) : null
              }
            >
              Pre-selected to {activeHypothesisIds.size} hypothes
              {activeHypothesisIds.size === 1 ? 'is' : 'es'} from {composition}{' '}
              combo{activeCombos.length === 1 ? '' : 's'}
              {combosAppliedAt ? ` · applied ${combosAppliedAt}` : ''}.
              {pendingCombos.length > 0 && ' Pending combos auto-promote when their hypotheses are implemented (see Review & Combos).'}
              {userTouched && ' (selection manually edited)'}
            </Alert>
          </Box>
        );
      })()}

      {(constraintViolations.errors.length > 0 || constraintViolations.warnings.length > 0) && (
        <Box sx={{ px: 2, pt: 1.5 }}>
          {constraintViolations.errors.map((v, i) => (
            <Alert key={`err-${i}`} severity="error" sx={{ mb: 0.5, fontSize: '0.78rem' }}>
              <strong>{v.constraint.label || 'Combo violates constraint'}</strong>: {v.constraint.reason}
              {v.missing && v.missing.length > 0 && (
                <> — missing: {v.missing.join(', ')}</>
              )}
              {v.conflicting && v.conflicting.length > 0 && (
                <> — conflicting: {v.conflicting.join(', ')}</>
              )}
            </Alert>
          ))}
          {constraintViolations.warnings.map((v, i) => (
            <Alert key={`warn-${i}`} severity="warning" sx={{ mb: 0.5, fontSize: '0.78rem' }}>
              <strong>{v.constraint.label}</strong>: {v.constraint.reason}
              {v.missing && v.missing.length > 0 && (
                <> — missing any of: {v.missing.join(', ')}</>
              )}
            </Alert>
          ))}
        </Box>
      )}

      <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
        {activeGroup && activeGroup.hypotheses.map(hyp => {
          const detail = proposalDetails[hyp.id] || {};
          const isCompleted = hyp.status === 'completed';
          const isSelected = selectedIds.has(hyp.id);
          const hypothesisResult = task.scenarioState?.hypothesisResults?.[hyp.id];
          const includedByH = includedBy.get(hyp.id) || null;
          const blockReason = blocked.get(hyp.id) || null;
          const isBlocked = blockReason !== null;
          const checkboxDisabled = !isCompleted || isBlocked || includedByH !== null;
          const tooltipText = includedByH
            ? `Included by ${includedByH} (auto-selected)`
            : (blockReason || (!isCompleted ? 'Hypothesis not yet implemented' : ''));

          return (
            <Card
              key={hyp.id}
              sx={{
                mb: 1.5,
                border: '1px solid',
                borderColor: isSelected
                  ? 'primary.main'
                  : includedByH
                    ? 'info.main'
                    : isBlocked
                      ? 'warning.main'
                      : 'rgba(255,255,255,0.12)',
                backgroundColor: isSelected
                  ? 'rgba(74, 144, 217, 0.06)'
                  : includedByH
                    ? 'rgba(33, 150, 243, 0.05)'
                    : isBlocked
                      ? 'rgba(255, 152, 0, 0.04)'
                      : 'rgba(255,255,255,0.02)',
                opacity: isBlocked ? 0.65 : 1,
              }}
            >
              <CardActionArea
                onClick={() => handleViewDetails(hyp.id)}
                disabled={!isCompleted}
                sx={{ display: 'flex', alignItems: 'center', px: 2, py: 1.5, justifyContent: 'flex-start' }}
              >
                <Tooltip title={tooltipText} placement="top" arrow>
                  <span onClick={(e) => e.stopPropagation()}>
                    <Checkbox
                      checked={isSelected || (includedByH !== null)}
                      onChange={(e) => { e.stopPropagation(); toggleSelect(hyp.id); }}
                      disabled={checkboxDisabled}
                      size="small"
                      sx={{ p: 0.5 }}
                    />
                  </span>
                </Tooltip>
                <Box sx={{ flex: 1, ml: 1 }}>
                  <Typography sx={{ fontSize: '0.9rem', fontWeight: 700 }}>
                    <Typography component="span" sx={{ color: 'primary.main', fontWeight: 700, mr: 0.5, fontSize: '0.85rem' }}>
                      {hyp.id}
                    </Typography>
                    {detail.title || hyp.id}
                  </Typography>
                  {hypothesisResult && (
                    <Typography sx={{ fontSize: '0.75rem', color: 'text.secondary', mt: 0.3 }}>
                      Consensus: {hypothesisResult.consensusInfo?.total_iterations || '?'} iterations
                    </Typography>
                  )}
                </Box>
                <Chip
                  label={hyp.status === 'completed' ? '✅ Completed' : hyp.status === 'error' ? '❌ Error' : '⏳ Pending'}
                  size="small"
                  color={hyp.status === 'completed' ? 'success' : hyp.status === 'error' ? 'error' : 'default'}
                  variant="outlined"
                  sx={{ fontSize: '0.65rem', height: 20 }}
                />
                {detail.impact && (
                  <Chip label={detail.impact} size="small" variant="outlined"
                    sx={{ fontSize: '0.65rem', height: 20, ml: 0.5 }} />
                )}
                {includedByH && (
                  <Chip label={`included by ${includedByH}`} size="small" color="info" variant="outlined"
                    sx={{ fontSize: '0.65rem', height: 20, ml: 0.5 }} />
                )}
                {isBlocked && (
                  <Chip label="🎰 slot conflict" size="small" color="warning" variant="outlined"
                    sx={{ fontSize: '0.65rem', height: 20, ml: 0.5 }} />
                )}
                {detail._isComboHypothesis && (
                  <Tooltip
                    title={
                      detail._comboReasonAnnotated
                        ? `${detail._deprioritizeReason} — comboKey="${detail._comboKey}"`
                        : `Pre-bundles: ${(detail._componentIds || []).join(', ')}. Use the corresponding combo (comboKey="${detail._comboKey}") on the Review & Combos tab.`
                    }
                  >
                    <Chip
                      label={`🔗 Combo alias of ${(detail._componentIds || []).join(', ')}`}
                      size="small"
                      color="warning"
                      variant="outlined"
                      sx={{ fontSize: '0.65rem', height: 20, ml: 0.5 }}
                    />
                  </Tooltip>
                )}
              </CardActionArea>
            </Card>
          );
        })}
      </Box>

      <Box sx={{
        borderTop: '1px solid rgba(255,255,255,0.08)',
        px: 2, py: 1.5,
        backgroundColor: 'rgba(0, 0, 0, 0.1)',
        display: 'flex',
        flexDirection: 'column',
        gap: 1,
      }}>
        <Typography variant="body2" sx={{ fontSize: '0.82rem' }}>
          {composedHs.length} hypothes{composedHs.length === 1 ? 'is' : 'es'} composed
          {selectedIds.size > 0 && freeTextParsedHs.length > 0
            ? ` (${selectedIds.size} BNB-checked + ${freeTextParsedHs.length} typed)`
            : selectedIds.size > 0
              ? ' from the BNB grid above'
              : freeTextParsedHs.length > 0
                ? ' from the typed input below'
                : ' — check hypotheses above and/or type IDs below to assemble a new combo'}
        </Typography>

        {composedHs.length > 0 && (
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
            {composedHs.map(h => {
              const fromTyped = freeTextParsedHs.includes(h);
              const fromBnb = selectedIds.has(h);
              return (
                <Chip
                  key={h}
                  size="small"
                  label={h}
                  variant="outlined"
                  color={fromTyped && !fromBnb ? 'secondary' : 'default'}
                  sx={{ fontSize: '0.62rem', height: 18 }}
                  title={fromTyped && fromBnb ? 'BNB-checked + typed (deduped)' : (fromTyped ? 'Typed' : 'BNB-checked')}
                />
              );
            })}
          </Box>
        )}

        {comboHint && comboHint.exact.length > 0 && (
          <Box sx={{
            p: 1, borderRadius: 1,
            backgroundColor: 'rgba(74, 144, 217, 0.08)',
            border: '1px solid rgba(74, 144, 217, 0.2)',
          }}>
            <Typography variant="body2" sx={{ fontSize: '0.78rem' }}>
              This combo was submitted before: {comboHint.exact.length} run(s).
              Latest: {new Date(comboHint.exact[comboHint.exact.length - 1].submittedAt).toLocaleString()}
              {' '}
              <Button
                size="small" sx={{ textTransform: 'none', fontSize: '0.75rem', p: 0 }}
                onClick={() => dispatch({ type: 'SET_ACTIVE_VIEW', viewIndex: 3 })}
              >
                → View in Experiments
              </Button>
            </Typography>
          </Box>
        )}

        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
          <Button size="small" onClick={selectAllCompleted} sx={{ textTransform: 'none', fontSize: '0.78rem' }}>
            Select All Completed
          </Button>
          <Button size="small" onClick={clearSelection} sx={{ textTransform: 'none', fontSize: '0.78rem' }}>
            Clear
          </Button>
          <Tooltip title="Add Hs by ID (e.g. H99, H100) — for not-yet-implemented hypotheses">
            <TextField
              size="small"
              placeholder="H99, H100…"
              value={freeTextHs}
              onChange={(e) => setFreeTextHs(e.target.value)}
              sx={{ width: 180, '& .MuiInputBase-input': { fontFamily: 'monospace', fontSize: '0.78rem' } }}
            />
          </Tooltip>
          {freeTextHs && (
            <Button
              size="small"
              onClick={() => setFreeTextHs('')}
              sx={{ textTransform: 'none', fontSize: '0.72rem' }}
            >
              Clear typed
            </Button>
          )}
          <Box sx={{ flex: 1 }} />
          <Tooltip title={
            composedHs.length < 2
              ? 'Compose at least 2 Hs (check above + type below) to assemble a combo'
              : 'Add a new combo from the composed hypothesis set'
          }>
            <span>
              <Button
                size="small" variant="outlined"
                startIcon={<AddIcon fontSize="small" />}
                onClick={handleAddCombo}
                disabled={composedHs.length < 2 || addComboBusy}
                sx={{ textTransform: 'none', fontSize: '0.78rem' }}
              >
                {addComboBusy ? 'Adding…' : `Add Combo from ${composedHs.length} hypothes${composedHs.length === 1 ? 'is' : 'es'}`}
              </Button>
            </span>
          </Tooltip>
        </Box>

      </Box>

      {flagMapWarnings && flagMapWarnings.length > 0 && !bannerDismissed && (
        <Box sx={{ px: 2, mb: 1 }}>
          <Alert
            severity="warning"
            onClose={() => setBannerDismissed(true)}
            action={
              <Button
                size="small"
                color="inherit"
                onClick={openSetupWizard}
                sx={{ textTransform: 'none' }}
              >
                Fix flag map →
              </Button>
            }
          >
            This hub&apos;s <code>hypothesisFlagMap</code> contains{' '}
            {flagMapWarnings.length} identity placeholder{flagMapWarnings.length === 1 ? '' : 's'}:{' '}
            <code>{flagMapWarnings.slice(0, 6).join(', ')}{flagMapWarnings.length > 6 ? ', …' : ''}</code>.
            {' '}Flag composition will reject these at runtime. Migrate to scoped
            bindings like <code>hstu_encoder.enable_h1</code> via the SetupWizard.
          </Alert>
        </Box>
      )}

      <Box sx={{ maxHeight: '45vh', overflow: 'auto', flexShrink: 0 }}>
        <ComboReviewSection
          activeCombos={activeCombos}
          selectedComboIds={selectedComboIds}
          expandedComboIds={expandedComboIds}
          onToggleCombo={handleToggleCombo}
          onToggleExpand={handleToggleExpandCombo}
          onRemoveCombo={handleRequestRemoveCombo}
          onSelectAll={handleSelectAllCombos}
          onClearSelected={handleClearSelectedCombos}
          onExpandAll={handleExpandAllCombos}
          onCollapseAll={handleCollapseAllCombos}
          onOpenSetupWizard={openSetupWizard}
        />
        {comboError && (
          <Box sx={{ px: 2, pb: 1, pt: 1 }}>
            <Alert severity="error" onClose={() => setComboError(null)} sx={{ fontSize: '0.78rem' }}>
              {comboError}
            </Alert>
          </Box>
        )}
      </Box>

      <Box sx={{
        borderTop: '1px solid rgba(255,255,255,0.08)',
        px: 2, py: 1, backgroundColor: 'rgba(0, 0, 0, 0.15)',
        display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap',
      }}>
        <Tooltip title="Max parallel combos when /experiment-hypothesis-combos runs">
          <TextField
            label="Max concurrency" type="number" size="small"
            value={maxConcurrency}
            onChange={(e) => setMaxConcurrency(Math.max(1, Math.min(8, parseInt(e.target.value, 10) || 1)))}
            inputProps={{ min: 1, max: 8, style: { width: 48, fontSize: '0.8rem' } }}
            sx={{ width: 120 }}
          />
        </Tooltip>
        <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
          {selectedCombos.length > 0
            ? `${selectedCombos.length} combo${selectedCombos.length === 1 ? '' : 's'} checked`
            : selectedIds.size > 0
              ? `${selectedIds.size} H${selectedIds.size === 1 ? '' : 's'} (legacy single-experiment)`
              : 'No combos checked'}
        </Typography>
        <Box sx={{ flex: 1 }} />
        <Tooltip title={
          selectedCombos.length > 0
            ? `Submit ${selectedCombos.length} checked combo${selectedCombos.length === 1 ? '' : 's'} via /experiment-hypothesis-combos`
            : selectedIds.size > 0
              ? 'No combos checked — falls back to one experiment of the currently-checked hypotheses'
              : 'Check at least one combo above (or hypotheses below) to submit'
        }>
          <span>
            <Button
              size="small" variant="contained"
              onClick={handleDirectComboSubmit}
              disabled={
                submitBusy
                || (selectedCombos.length === 0 && selectedIds.size === 0)
                || (selectedCombos.length === 0 && submitDisabled)
              }
              sx={{ textTransform: 'none', fontSize: '0.78rem' }}
            >
              {submitBusy
                ? 'Submitting…'
                : selectedCombos.length > 0
                  ? `⚙ Submit ${selectedCombos.length} combo${selectedCombos.length === 1 ? '' : 's'}`
                  : `⚙ Submit Experiment (${selectedIds.size} H${selectedIds.size === 1 ? '' : 's'})`}
            </Button>
          </span>
        </Tooltip>
      </Box>

      <SubmissionFooterBar
        task={task}
        selectedIds={selectedIds}
        submitLabel={config.submitLabel || 'Submit Experiment'}
        proposalDetails={proposalDetails}
        selectedCombos={selectedCombos}
        submitDisabled={submitDisabled}
        submitDisabledReason={
          constraintViolations.errors.length > 0
            ? constraintViolations.errors[0].constraint.label || constraintViolations.errors[0].constraint.reason
            : selectedCombos.length === 0 && selectedIds.size === 0
              ? 'Check at least one combo above (or hypotheses below)'
              : ''
        }
        sessionId={sessionId}
        dispatch={dispatch}
        apiClient={apiClient}
        config={config.hubConfig || config}
      />

      {config.detailDrawer && (
        <HypothesisDetailDrawer
          open={drawerOpen}
          onClose={() => setDrawerOpen(false)}
          hypothesisId={drawerHypothesisId}
          task={task}
          proposalDetails={proposalDetails}
        />
      )}
    </Box>
  );
}
