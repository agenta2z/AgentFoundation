/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * UploadExistingRunnerModal — direct-upload escape hatch for users who
 * already have a working `submit.py` + `launch.json`. Bypasses PTI.
 *
 * Ported from RankEvolve (components/hub/UploadExistingRunnerModal.js); the
 * `useSession().setupSubmission` read is replaced by an injected
 * `setupSubmission` prop (apiClient.setupSubmission).
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  AlertTitle,
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  TextField,
  Typography,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

const RECENT_PATHS_KEY = 'rankevolve_setup_recent_paths';
const RECENT_PATHS_CAP = 10;

function loadRecentPaths() {
  try {
    const raw = localStorage.getItem(RECENT_PATHS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((p) => typeof p === 'string') : [];
  } catch {
    return [];
  }
}

function saveRecentPaths(paths) {
  try {
    localStorage.setItem(
      RECENT_PATHS_KEY,
      JSON.stringify(paths.slice(0, RECENT_PATHS_CAP)),
    );
  } catch {
    // Best-effort; localStorage failures must not block the upload.
  }
}

function ValidationReport({ report }) {
  if (!report || !Array.isArray(report.findings) || report.findings.length === 0) {
    if (report?.severity === 'ok') {
      return (
        <Alert severity="success" sx={{ fontSize: '0.78rem' }}>
          Validation passed.
        </Alert>
      );
    }
    return null;
  }
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
      {report.findings.map((f, idx) => (
        <Alert
          key={idx}
          severity={f.severity || 'info'}
          variant="outlined"
          sx={{ fontSize: '0.76rem', py: 0.25 }}
        >
          <strong>{f.category || 'check'}:</strong> {f.message}
        </Alert>
      ))}
    </Box>
  );
}

export default function UploadExistingRunnerModal({ open, onClose, task, setupSubmission }) {
  const priorInputs = task?.submissionSetup?.inputs || {};
  const priorOriginIsUpload = useMemo(() => {
    const versions = task?.submissionSetup?.scriptVersions || [];
    if (versions.length === 0) return false;
    const last = versions[versions.length - 1];
    return last?.source === 'upload';
  }, [task?.submissionSetup?.scriptVersions]);

  const [setupName, setSetupName] = useState('');
  const [scriptPath, setScriptPath] = useState('');
  const [launchPath, setLaunchPath] = useState('');
  const [recentPaths, setRecentPaths] = useState(() => loadRecentPaths());
  const [validating, setValidating] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!open) return;
    setError('');
    setReport(null);
    setRecentPaths(loadRecentPaths());
    setSetupName(priorInputs.name || 'Imported runner');
    if (priorOriginIsUpload) {
      setScriptPath(priorInputs.scriptPath || '');
      setLaunchPath(priorInputs.launchPath || '');
    } else {
      setScriptPath('');
      setLaunchPath('');
    }
  }, [open, priorOriginIsUpload, priorInputs.name, priorInputs.scriptPath, priorInputs.launchPath]);

  const submitDisabled = !setupName.trim() || !scriptPath.trim() || !launchPath.trim();
  const uploadDisabled = submitDisabled || report?.severity === 'error';

  const buildPayload = useCallback(() => ({
    setupName: setupName.trim(),
    mode: 'import',
    scriptPath: scriptPath.trim(),
    launchPath: launchPath.trim(),
  }), [setupName, scriptPath, launchPath]);

  const handleValidate = useCallback(async () => {
    if (!task?.id || submitDisabled || typeof setupSubmission !== 'function') return;
    setValidating(true);
    setError('');
    setReport(null);
    try {
      const res = await setupSubmission(task.id, buildPayload(), { dryRun: true });
      if (res?.error === 'validation') {
        setReport(res.detail?.validation_report || null);
      } else if (res?.error) {
        setError(res.detail?.detail || res.detail || res.error);
      } else if (res?.validation_report) {
        setReport(res.validation_report);
      } else {
        setReport({ severity: 'ok', findings: [] });
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setValidating(false);
    }
  }, [task?.id, submitDisabled, setupSubmission, buildPayload]);

  const handleUpload = useCallback(async () => {
    if (!task?.id || uploadDisabled || typeof setupSubmission !== 'function') return;
    setUploading(true);
    setError('');
    try {
      const res = await setupSubmission(task.id, buildPayload());
      if (res?.error === 'validation') {
        setReport(res.detail?.validation_report || null);
        setError('Validation failed; see findings below.');
      } else if (res?.error) {
        setError(res.detail?.detail || res.detail || res.error);
      } else {
        const updated = [scriptPath.trim(), launchPath.trim(), ...recentPaths]
          .filter((p, i, a) => p && a.indexOf(p) === i);
        saveRecentPaths(updated);
        setRecentPaths(updated);
        if (res?.validation_report) {
          setReport(res.validation_report);
        }
        onClose();
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setUploading(false);
    }
  }, [task?.id, uploadDisabled, setupSubmission, buildPayload, scriptPath, launchPath, recentPaths, onClose]);

  const renderRecentChips = (setter) => {
    if (!recentPaths.length) return null;
    return (
      <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 0.5 }}>
        {recentPaths.slice(0, 5).map((p) => (
          <Chip
            key={p}
            label={p.length > 60 ? `…${p.slice(-58)}` : p}
            size="small"
            onClick={() => setter(p)}
            sx={{ fontSize: '0.66rem', height: 20, maxWidth: 320 }}
          />
        ))}
      </Box>
    );
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ pb: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography sx={{ fontSize: '1rem', fontWeight: 600 }}>
            Upload existing experiment runner
          </Typography>
          <Box sx={{ flex: 1 }} />
          <IconButton size="small" onClick={onClose}>
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>
      </DialogTitle>
      <DialogContent dividers sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <Alert severity="warning" sx={{ fontSize: '0.78rem' }}>
          <AlertTitle sx={{ fontSize: '0.82rem' }}>Advanced — bypasses PTI</AlertTitle>
          <Typography variant="caption" sx={{ display: 'block' }}>
            Your script must meet the runner contract:
            (1) print <code>FLOW_URI: &lt;url&gt;</code> and <code>MAST_JOB: &lt;job&gt;</code> per
            launch attempt via bare <code>print()</code> (NOT <code>logger.info()</code>);
            (2) accept <code>--enable-flags --experiment-name --app-layer-version
            --max-retry --experiment-workspace</code>;
            (3) install <code>SIGTERM</code> / <code>SIGINT</code> handlers;
            (4) exit 0 only on success. Click <strong>Validate</strong> to check before uploading.
          </Typography>
          <Typography variant="caption" sx={{ display: 'block', mt: 0.5 }}>
            Server snapshots the files at upload time — later edits to your source path do not propagate.
            Re-import to refresh.
          </Typography>
        </Alert>

        <TextField
          label="Setup name"
          value={setupName}
          onChange={(e) => setSetupName(e.target.value)}
          size="small"
          fullWidth
          required
          helperText="Displayed in the footer + Submit Confirm modal."
        />

        <Box>
          <TextField
            label="submit.py absolute path (server-side)"
            placeholder="/data/users/.../my_submit.py"
            value={scriptPath}
            onChange={(e) => setScriptPath(e.target.value)}
            size="small"
            fullWidth
            required
          />
          {renderRecentChips(setScriptPath)}
        </Box>

        <Box>
          <TextField
            label="launch.json absolute path (server-side)"
            placeholder="/data/users/.../my_launch.json"
            value={launchPath}
            onChange={(e) => setLaunchPath(e.target.value)}
            size="small"
            fullWidth
            required
          />
          {renderRecentChips(setLaunchPath)}
        </Box>

        {report && <ValidationReport report={report} />}

        {error && (
          <Alert severity="error" sx={{ fontSize: '0.78rem' }}>
            {error}
          </Alert>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={validating || uploading} sx={{ textTransform: 'none' }}>
          Cancel
        </Button>
        <Button
          onClick={handleValidate}
          disabled={validating || submitDisabled}
          sx={{ textTransform: 'none' }}
        >
          {validating ? 'Validating…' : 'Validate'}
        </Button>
        <Button
          variant="contained"
          onClick={handleUpload}
          disabled={uploading || uploadDisabled}
          sx={{ textTransform: 'none' }}
        >
          {uploading ? 'Uploading…' : '📤 Upload as Experiment Runner'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
