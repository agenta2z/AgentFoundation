/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * SubmissionFooterBar — wrapper around <SubmissionActionButton> that owns the
 * submission-flow modals (SetupWizardModal, ScriptEditorDrawer,
 * SubmitExperimentConfirm, UploadExistingRunnerModal) and the shared
 * bookkeeping.
 *
 * Ported from RankEvolve (components/hub/SubmissionFooterBar.js). The
 * `useSession()` reads become injected props:
 *   - `sessionId`            (was activeSessionId)
 *   - `dispatch`             (hub dashboard reducer dispatch)
 *   - `apiClient`            ({ addSubmission, runSubmission, setupSubmission,
 *                              saveScriptVersion, switchTab })
 *   - `config`              (hub/session config; threaded to SetupWizardModal)
 * The lazy per-hub setup-state GET is a read → direct REST. Hub-internal view
 * switches use the generic reducer's `SET_ACTIVE_VIEW {viewIndex}` action;
 * jump-to-setup-task uses the injected `apiClient.switchTab`.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Snackbar,
  Typography,
} from '@mui/material';
import SetupWizardModal from './SetupWizardModal';
import ScriptEditorDrawer from './ScriptEditorDrawer';
import SubmitExperimentConfirm from './SubmitExperimentConfirm';
import SubmissionActionButton from './SubmissionActionButton';
import UploadExistingRunnerModal from './UploadExistingRunnerModal';

const EXPERIMENTS_VIEW_INDEX = 3;

function StatusBadge({ status }) {
  if (status === 'ready') {
    return <Chip size="small" color="success" variant="outlined" label="✅ Ready"
      sx={{ fontSize: '0.7rem', height: 22 }} />;
  }
  if (status === 'in_progress') {
    return <Chip size="small" color="info" variant="outlined" label="⏳ Generating"
      sx={{ fontSize: '0.7rem', height: 22 }} />;
  }
  if (status === 'error') {
    return <Chip size="small" color="error" variant="outlined" label="⚠ Error"
      sx={{ fontSize: '0.7rem', height: 22 }} />;
  }
  return <Chip size="small" variant="outlined" label="⚪ Not set up"
    sx={{ fontSize: '0.7rem', height: 22 }} />;
}

function relativeTime(epochMs) {
  if (!epochMs) return '';
  const sec = Math.floor((Date.now() - epochMs) / 1000);
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}min ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  return `${Math.floor(hr / 24)}d ago`;
}

export default function SubmissionFooterBar({
  task,
  selectedIds,
  submitLabel,  // unused after v2 refactor — kept for back-compat callers
  onSubmit,
  proposalDetails,
  submitDisabled = false,
  submitDisabledReason = '',
  selectedCombos = null,
  // Injected context (was useSession()):
  sessionId,
  dispatch,
  apiClient,
  config,
}) {
  const addSubmission = apiClient && apiClient.addSubmission;
  const runSubmission = apiClient && apiClient.runSubmission;
  const setupSubmission = apiClient && apiClient.setupSubmission;
  const saveScriptVersion = apiClient && apiClient.saveScriptVersion;
  const switchTab = apiClient && apiClient.switchTab;

  const [wizardOpen, setWizardOpen] = useState(false);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [errorDialogOpen, setErrorDialogOpen] = useState(false);
  const [snackOpen, setSnackOpen] = useState(false);
  const [snackMessage, setSnackMessage] = useState('');
  const setup = task.submissionSetup || null;

  const priorStatusRef = useRef(setup?.status);
  useEffect(() => {
    const prior = priorStatusRef.current;
    const current = setup?.status;
    if (prior === 'in_progress' && current === 'error') {
      setSnackOpen(true);
      setSnackMessage('Experiment runner setup failed for this hub.');
    }
    priorStatusRef.current = current;
  }, [setup?.status]);

  // Lazy-fetch the per-hub setup state on mount (read → direct REST).
  useEffect(() => {
    if (setup) return undefined;
    if (!sessionId || !task.id) return undefined;
    let cancelled = false;
    fetch(
      `/api/hubs/${encodeURIComponent(task.id)}/submission-setup?session_id=${encodeURIComponent(sessionId)}`,
    )
      .then((res) => (res.ok ? res.json() : null))
      .then((body) => {
        if (cancelled) return;
        const fetched = body?.setup;
        dispatch({
          type: 'SUBMISSION_SETUP_STARTED',
          taskId: task.id,
          setup: fetched && Object.keys(fetched).length > 0 ? fetched : null,
        });
      })
      .catch(() => {
        // Best-effort; errors leave setup=null and the button shows "Not set up".
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId, task.id, setup, dispatch]);

  const status = setup?.status || 'not_started';
  const scriptName = useMemo(() => {
    if (!setup?.scriptPath) return null;
    const parts = String(setup.scriptPath).split('/');
    return parts[parts.length - 1] || null;
  }, [setup?.scriptPath]);
  const lastEditedAgo = setup?.generatedAt ? relativeTime(setup.generatedAt) : '';

  const selectedItems = useMemo(() => [...(selectedIds || [])], [selectedIds]);
  const submitCombos = useMemo(
    () => (Array.isArray(selectedCombos) ? selectedCombos.filter(Boolean) : []),
    [selectedCombos],
  );
  const isComboBatch = submitCombos.length > 0;

  const defaultExperimentName = useMemo(() => {
    if (selectedItems.length === 0) return '';
    const base = setup?.setupName
      ? `${setup.setupName}_${selectedItems.join('_')}`
      : `combo_${selectedItems.join('_')}`;
    return base.replace(/[^A-Za-z0-9_\-]/g, '_');
  }, [setup?.setupName, selectedItems]);

  const defaultEnableFlags = useMemo(() => {
    const map = (setup?.inputs && setup.inputs.hypothesisFlagMap) || {};
    return selectedItems
      .map((hid) => map[hid] || `enable_${String(hid).toLowerCase()}`)
      .filter(Boolean);
  }, [selectedItems, setup?.inputs]);

  const defaultAppLayerVersion = useMemo(() => {
    const fromInputs =
      setup?.inputs && typeof setup.inputs.lastAppLayerVersion === 'string'
        ? setup.inputs.lastAppLayerVersion
        : '';
    if (fromInputs) return fromInputs;
    try {
      return window.localStorage.getItem('rankevolve_last_app_layer_version') || '';
    } catch (e) {
      return '';
    }
  }, [setup?.inputs]);

  const hasBuildCommand = useMemo(() => {
    const ref = setup?.inputs?.referenceCommand;
    return typeof ref === 'string' && ref.trim().length > 0;
  }, [setup?.inputs]);

  const handleSetupClick = useCallback(() => setWizardOpen(true), []);
  const handleRegenerateClick = useCallback(() => setWizardOpen(true), []);

  useEffect(() => {
    const onOpenWizard = (ev) => {
      const detail = ev?.detail || {};
      if (!task?.id || detail.multi_task_id === task.id) {
        setWizardOpen(true);
      }
    };
    window.addEventListener('open_hub_setup_wizard', onOpenWizard);
    return () => window.removeEventListener('open_hub_setup_wizard', onOpenWizard);
  }, [task?.id]);
  const handleViewScriptClick = useCallback(() => setDrawerOpen(true), []);
  const handleViewInputsClick = useCallback(() => {
    setWizardOpen(true);
  }, []);
  const handleViewErrorClick = useCallback(() => {
    setErrorDialogOpen(true);
  }, []);

  const handleCopyError = useCallback(async () => {
    try {
      await navigator.clipboard?.writeText(setup?.error || '');
      setSnackMessage('Copied to clipboard');
      setSnackOpen(true);
    } catch (e) {
      setSnackMessage('Copy failed — select the text manually');
      setSnackOpen(true);
    }
  }, [setup?.error]);

  const handleUploadInstead = useCallback(() => {
    setErrorDialogOpen(false);
    setUploadModalOpen(true);
  }, []);

  const handleRetryFromDialog = useCallback(() => {
    setErrorDialogOpen(false);
    setWizardOpen(true);
  }, []);

  const handleUploadFromWizard = useCallback(() => {
    setWizardOpen(false);
    setUploadModalOpen(true);
  }, []);

  // State B click — navigate to the setup task subtab (host-owned navigation).
  const handleJumpToTaskClick = useCallback(
    (taskId) => {
      if (!taskId) return;
      if (typeof switchTab === 'function') switchTab(taskId, 'task');
    },
    [switchTab],
  );

  const handleSubmitClick = useCallback(() => {
    if (submitDisabled && !isComboBatch) {
      console.warn('[SubmissionFooterBar] Submit blocked:', submitDisabledReason);
      return;
    }
    if (!isComboBatch && selectedItems.length === 0) return;
    if (isComboBatch && submitCombos.length === 0) return;
    if (!setup || setup.status !== 'ready' || !setup.scriptPath || !setup.launchPath) {
      console.warn('[SubmissionFooterBar] Submit clicked without ready setup');
      return;
    }
    setConfirmOpen(true);
  }, [selectedItems, setup, submitDisabled, submitDisabledReason, isComboBatch, submitCombos]);

  const handleConfirmSubmit = useCallback(
    async ({ enableFlags, experimentName, appLayerVersion }) => {
      setConfirmOpen(false);
      try {
        if (appLayerVersion) {
          window.localStorage.setItem(
            'rankevolve_last_app_layer_version',
            appLayerVersion,
          );
        }
      } catch (e) { /* ignore */ }

      if (typeof addSubmission !== 'function' || typeof runSubmission !== 'function') {
        console.warn('[SubmissionFooterBar] addSubmission/runSubmission unavailable');
        return;
      }

      if (isComboBatch) {
        const map = (setup?.inputs && setup.inputs.hypothesisFlagMap) || {};
        const failures = [];
        for (const combo of submitCombos) {
          const items = combo?.selectedItems || [];
          if (items.length === 0) continue;
          const comboId = combo?.comboId || `combo_${[...items].sort().join('_')}`;
          const perComboName = `${experimentName || 'experiment'}_${comboId}`;
          const perComboFlags = items
            .map(hid => map[hid] || `enable_${String(hid).toLowerCase()}`)
            .filter(Boolean);
          try {
            const created = await addSubmission(task.id, {
              selectedItems: items,
              comboKey: combo?.comboKey || [...items].sort().join(','),
              config: {
                name: setup?.setupName
                  ? `${setup.setupName} — ${comboId}`
                  : comboId,
                future_combo_id: comboId,
                gin_config: combo?.configPathProposed || '',
              },
            });
            const submissionId = created?.id || created?.submission_id;
            if (!submissionId) {
              failures.push({ comboId, error: 'addSubmission returned no id' });
              continue;
            }
            await runSubmission(task.id, submissionId, {
              scriptPath: setup.scriptPath,
              launchPath: setup.launchPath,
              enableFlags: perComboFlags,
              experimentName: perComboName,
              appLayerVersion,
              submissionLabel: items.join(','),
            });
          } catch (e) {
            failures.push({ comboId, error: e?.message || String(e) });
          }
        }
        if (failures.length > 0) {
          const summary = failures
            .map(f => `${f.comboId} (${f.error})`)
            .join('; ');
          setSnackMessage(`Submitted with ${failures.length} failure(s): ${summary}`);
          setSnackOpen(true);
        }
        dispatch({ type: 'SET_ACTIVE_VIEW', viewIndex: EXPERIMENTS_VIEW_INDEX });
        if (typeof onSubmit === 'function') onSubmit();
        return;
      }

      const items = selectedItems;
      if (items.length === 0) return;
      const created = await addSubmission(task.id, {
        selectedItems: items,
        comboKey: [...items].sort().join(','),
        config: {
          name: setup.setupName ? `${setup.setupName} — ${items.join(',')}` : items.join(','),
        },
      });
      const submissionId = created?.id || created?.submission_id;
      if (!submissionId) {
        console.warn('[SubmissionFooterBar] addSubmission returned no id', created);
        return;
      }
      await runSubmission(task.id, submissionId, {
        scriptPath: setup.scriptPath,
        launchPath: setup.launchPath,
        enableFlags,
        experimentName,
        appLayerVersion,
        submissionLabel: items.join(','),
      });
      dispatch({ type: 'SET_ACTIVE_VIEW', viewIndex: EXPERIMENTS_VIEW_INDEX });
      if (typeof onSubmit === 'function') onSubmit();
    },
    [
      onSubmit,
      addSubmission,
      runSubmission,
      dispatch,
      task.id,
      selectedItems,
      setup,
      isComboBatch,
      submitCombos,
    ],
  );

  const captionText = useMemo(() => {
    if (status === 'ready' && scriptName) {
      return `${scriptName}${lastEditedAgo ? ` · created ${lastEditedAgo}` : ''}`;
    }
    if (status === 'error') {
      return '';
    }
    if (status === 'in_progress') {
      return '';
    }
    return 'Create an experiment runner for this hub before submitting';
  }, [status, scriptName, lastEditedAgo]);

  const errorPreview = useMemo(() => {
    if (!setup?.error) return '';
    const firstLine = setup.error.split('\n')[0];
    return firstLine.length > 100 ? `${firstLine.slice(0, 100)}…` : firstLine;
  }, [setup?.error]);

  const versionCount = setup?.scriptVersions?.length || 0;

  return (
    <Box
      sx={{
        borderTop: '1px solid rgba(255,255,255,0.08)',
        px: 2,
        py: 1.5,
        backgroundColor: 'rgba(0,0,0,0.10)',
        display: 'flex',
        flexDirection: 'column',
        gap: 1,
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Typography variant="body2" sx={{ fontSize: '0.82rem', fontWeight: 600 }}>
          Experiment Runner
        </Typography>
        <Box sx={{ flex: 1 }} />
        {versionCount > 1 && (
          <Chip
            size="small"
            label={`${versionCount} versions`}
            onClick={() => setDrawerOpen(true)}
            sx={{ fontSize: '0.7rem', height: 22 }}
          />
        )}
        <StatusBadge status={status} />
      </Box>

      {captionText && (
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ fontSize: '0.72rem' }}
        >
          {captionText}
        </Typography>
      )}

      {status === 'error' && (
        <Alert
          severity="error"
          variant="outlined"
          sx={{ fontSize: '0.78rem', py: 0.5 }}
          action={
            <Button onClick={handleViewErrorClick} size="small" color="inherit">
              View details
            </Button>
          }
        >
          Setup failed: {errorPreview || 'unknown'}
        </Alert>
      )}

      <SubmissionActionButton
        status={status}
        selectedIds={selectedIds}
        setup={setup}
        onSetupClick={handleSetupClick}
        onRegenerateClick={handleRegenerateClick}
        onSubmitClick={handleSubmitClick}
        onViewScriptClick={handleViewScriptClick}
        onJumpToTaskClick={handleJumpToTaskClick}
        onViewInputsClick={handleViewInputsClick}
        onViewErrorClick={handleViewErrorClick}
        externalDisabled={submitDisabled && !isComboBatch}
        externalDisabledReason={submitDisabledReason}
        selectedCombos={submitCombos}
      />

      <SetupWizardModal
        open={wizardOpen}
        onClose={() => setWizardOpen(false)}
        task={task}
        selectedIds={selectedIds}
        proposalDetails={proposalDetails}
        onUploadInsteadClick={handleUploadFromWizard}
        setupSubmission={setupSubmission}
        config={config}
      />

      <UploadExistingRunnerModal
        open={uploadModalOpen}
        onClose={() => setUploadModalOpen(false)}
        task={task}
        setupSubmission={setupSubmission}
      />

      <ScriptEditorDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        task={task}
        sessionId={sessionId}
        saveScriptVersion={saveScriptVersion}
      />

      <SubmitExperimentConfirm
        open={confirmOpen}
        onCancel={() => setConfirmOpen(false)}
        onConfirm={handleConfirmSubmit}
        combo={selectedItems}
        defaultFlags={defaultEnableFlags}
        defaultExperimentName={defaultExperimentName}
        defaultAppLayerVersion={defaultAppLayerVersion}
        hasBuildCommand={hasBuildCommand}
        scriptPath={setup?.scriptPath}
        selectedCombos={submitCombos}
        hypothesisFlagMap={(setup?.inputs && setup.inputs.hypothesisFlagMap) || {}}
      />

      <Dialog
        open={errorDialogOpen}
        onClose={() => setErrorDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Experiment runner setup failed
          {setup?.setupName ? ` — ${setup.setupName}` : ''}
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            The setup task did not complete successfully. Full error message:
          </Typography>
          <Box
            component="pre"
            sx={{
              p: 2,
              bgcolor: 'rgba(0,0,0,0.30)',
              fontFamily: 'monospace',
              fontSize: 13,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              maxHeight: 400,
              overflow: 'auto',
              borderRadius: 1,
              m: 0,
            }}
          >
            {setup?.error
              || '(no error message captured — check the agent server logs for the setup task)'}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCopyError}>Copy error</Button>
          <Button onClick={handleUploadInstead} color="secondary">
            Upload script directly instead
          </Button>
          <Button onClick={() => setErrorDialogOpen(false)}>Close</Button>
          <Button variant="contained" onClick={handleRetryFromDialog}>
            Retry Setup
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snackOpen}
        autoHideDuration={8000}
        onClose={() => setSnackOpen(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          severity={snackMessage.startsWith('Copied') ? 'success' : 'error'}
          onClose={() => setSnackOpen(false)}
          action={
            status === 'error' && !snackMessage.startsWith('Copied') ? (
              <Button
                color="inherit"
                size="small"
                onClick={() => {
                  setSnackOpen(false);
                  handleViewErrorClick();
                }}
              >
                View details
              </Button>
            ) : null
          }
        >
          {snackMessage}
        </Alert>
      </Snackbar>
    </Box>
  );
}
