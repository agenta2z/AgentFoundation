/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * SetupWizardModal — collects the user's reference materials and submits to
 * the hub's setup_submission action which proxies PTI to create the
 * experiment runner.
 *
 * Ported from RankEvolve (components/hub/SetupWizardModal.js). The two
 * SessionContext reads are replaced by injected props: `setupSubmission`
 * (apiClient.setupSubmission) and `config` (the hub/session config object
 * carrying `target_path`).
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
  FormControl,
  IconButton,
  InputLabel,
  Link,
  MenuItem,
  Select,
  Tab,
  Tabs,
  TextField,
  Typography,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import CloseIcon from '@mui/icons-material/Close';
import useSubmissionTemplates from '../../hooks/useSubmissionTemplates';
import HowCombosWork from './HowCombosWork';

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
    localStorage.setItem(RECENT_PATHS_KEY, JSON.stringify(paths.slice(0, RECENT_PATHS_CAP)));
  } catch {
    // Best-effort; localStorage failures must not break the wizard.
  }
}

function deriveDefaultName(selectedIds) {
  const ids = Array.from(selectedIds || []);
  if (ids.length === 0) return 'Submission setup';
  if (ids.length === 1) return `Setup for ${ids[0]}`;
  return `Setup for ${ids.length} hypotheses`;
}

export default function SetupWizardModal({
  open,
  onClose,
  task,
  selectedIds,
  proposalDetails,
  onUploadInsteadClick,
  setupSubmission,
  config,
}) {
  const { templates: libraryTemplates, loading: librariesLoading } =
    useSubmissionTemplates();
  const [refTab, setRefTab] = useState(0); // 0=Library, 1=Custom paths
  const [setupName, setSetupName] = useState('');
  const [referenceScripts, setReferenceScripts] = useState([]);
  const [libraryTemplate, setLibraryTemplate] = useState('');
  const [pendingPath, setPendingPath] = useState('');
  const [referenceCommand, setReferenceCommand] = useState('');
  const [additionalInstructions, setAdditionalInstructions] = useState('');
  const [recentPaths, setRecentPaths] = useState(() => loadRecentPaths());
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!open) return;
    const prior = task?.submissionSetup?.inputs || {};
    setSetupName(prior.name || deriveDefaultName(selectedIds));
    setReferenceScripts(Array.isArray(prior.referenceScripts) ? prior.referenceScripts : []);
    setLibraryTemplate(prior.libraryTemplate || '');
    setRefTab(prior.libraryTemplate ? 0 : 1);
    setReferenceCommand(prior.referenceCommand || '');
    setAdditionalInstructions(prior.additionalInstructions || '');
    setPendingPath('');
    setError('');
    setRecentPaths(loadRecentPaths());
  }, [open, task?.submissionSetup, selectedIds]);

  const selectedTemplate = useMemo(
    () => libraryTemplates.find((t) => t.id === libraryTemplate) || null,
    [libraryTemplates, libraryTemplate],
  );

  const selectedHypothesisIds = useMemo(
    () => Array.from(selectedIds || []),
    [selectedIds],
  );
  const hypothesisFlagMap = useMemo(() => {
    const map = {};
    for (const hid of selectedHypothesisIds) {
      const detail = proposalDetails?.[hid] || {};
      const flag =
        detail.flagName ||
        detail.enable_flag ||
        `enable_${String(hid).toLowerCase()}`;
      map[hid] = flag;
    }
    return map;
  }, [selectedHypothesisIds, proposalDetails]);

  const flagWarnings = useMemo(() => {
    const warns = [];
    for (const hid of selectedHypothesisIds) {
      const flag = hypothesisFlagMap[hid] || '';
      if (!flag.includes('.')) {
        warns.push(
          `${hid} → "${flag}" is BARE (no scope). Edit to add the @gin.configurable class, e.g., "hstu_encoder.${flag}". See implementation template § Gin Scope Convention.`
        );
      }
    }
    return warns;
  }, [selectedHypothesisIds, hypothesisFlagMap]);

  const addPath = useCallback(() => {
    const candidate = pendingPath.trim();
    if (!candidate) return;
    if (referenceScripts.includes(candidate)) {
      setPendingPath('');
      return;
    }
    setReferenceScripts((prev) => [...prev, candidate]);
    setPendingPath('');
  }, [pendingPath, referenceScripts]);

  const removePath = useCallback((path) => {
    setReferenceScripts((prev) => prev.filter((p) => p !== path));
  }, []);

  const useRecentPath = useCallback((path) => {
    if (referenceScripts.includes(path)) return;
    setReferenceScripts((prev) => [...prev, path]);
  }, [referenceScripts]);

  const handleSubmit = useCallback(async () => {
    if (!task?.id) return;
    setError('');
    if (!setupName.trim()) {
      setError('Setup name is required.');
      return;
    }
    if (
      referenceScripts.length === 0 &&
      !libraryTemplate &&
      !referenceCommand.trim()
    ) {
      setError(
        'Provide at least one reference: a library template, a custom script '
        + 'path, OR a reference command — PTI needs reference material to '
        + 'create the experiment runner.',
      );
      return;
    }
    if (typeof setupSubmission !== 'function') {
      setError('Setup action unavailable (apiClient.setupSubmission not provided).');
      return;
    }
    setSubmitting(true);
    try {
      const result = await setupSubmission(task.id, {
        setupName: setupName.trim(),
        referenceScripts,
        libraryTemplate: libraryTemplate || null,
        referenceCommand: referenceCommand.trim(),
        additionalInstructions: additionalInstructions.trim(),
        selectedHypothesisIds,
        hypothesisFlagMap,
      });
      if (result && result.error) {
        setError(
          result.error === 'in_progress'
            ? 'A setup is already in progress for this hub. Wait for it to finish.'
            : `Failed to start setup: ${result.detail || result.error}`,
        );
        return;
      }
      if (referenceScripts.length > 0) {
        const merged = [...referenceScripts, ...recentPaths.filter((p) => !referenceScripts.includes(p))];
        saveRecentPaths(merged);
        setRecentPaths(merged);
      }
      onClose();
    } catch (e) {
      setError(`Failed to start setup: ${e}`);
    } finally {
      setSubmitting(false);
    }
  }, [
    task?.id,
    setupName,
    referenceScripts,
    libraryTemplate,
    referenceCommand,
    additionalInstructions,
    selectedHypothesisIds,
    hypothesisFlagMap,
    setupSubmission,
    onClose,
    recentPaths,
  ]);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ pb: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography sx={{ fontSize: '1rem', fontWeight: 600 }}>
            Create Experiment Runner
          </Typography>
          <Box sx={{ flex: 1 }} />
          <IconButton size="small" onClick={onClose}>
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>
      </DialogTitle>
      <DialogContent dividers sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {task?.submissionSetup?.status === 'error' && Boolean(task?.submissionSetup?.error) && (
          <Alert severity="error" sx={{ mb: 0 }}>
            <AlertTitle>Previous attempt failed</AlertTitle>
            <Typography variant="body2" sx={{ mb: 1 }}>
              The previous attempt to create this experiment runner failed:
            </Typography>
            <Box
              component="pre"
              sx={{
                p: 1.5,
                bgcolor: 'rgba(0,0,0,0.3)',
                fontFamily: 'monospace',
                fontSize: 12,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                maxHeight: 200,
                overflow: 'auto',
                borderRadius: 1,
                m: 0,
              }}
            >
              {task.submissionSetup.error}
            </Box>
            <Typography
              variant="caption"
              sx={{ mt: 1, display: 'block' }}
              color="text.secondary"
            >
              Adjust your inputs below if this error suggests a fix, then click Create Experiment Runner.
            </Typography>
          </Alert>
        )}

        <TextField
          label="Setup name"
          value={setupName}
          onChange={(e) => setSetupName(e.target.value)}
          size="small"
          fullWidth
          helperText="Displayed in the footer + Submit Confirm modal."
        />

        <Box>
          <Typography variant="body2" sx={{ fontSize: '0.82rem', fontWeight: 600, mb: 0.5 }}>
            Reference material
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
            What the runner should be based on. Pick a library
            template, paste your own paths, or both — they're combined
            into the prompt.
          </Typography>
          <Tabs
            value={refTab}
            onChange={(_e, v) => setRefTab(v)}
            sx={{ minHeight: 32, '& .MuiTab-root': { minHeight: 32, fontSize: '0.78rem', textTransform: 'none' } }}
          >
            <Tab label="From Library" />
            <Tab label="Custom paths" />
          </Tabs>

          {refTab === 0 && (
            <Box sx={{ mt: 1.5, display: 'flex', flexDirection: 'column', gap: 1 }}>
              <FormControl size="small" fullWidth disabled={librariesLoading}>
                <InputLabel id="library-template-label">Pick template</InputLabel>
                <Select
                  labelId="library-template-label"
                  label="Pick template"
                  value={libraryTemplate}
                  onChange={(e) => setLibraryTemplate(e.target.value)}
                >
                  <MenuItem value="">
                    <em>(none — use Custom paths)</em>
                  </MenuItem>
                  {libraryTemplates.map((t) => (
                    <MenuItem key={t.id} value={t.id}>
                      {t.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              {librariesLoading && (
                <Typography variant="caption" color="text.secondary">
                  Loading library…
                </Typography>
              )}
              {selectedTemplate && (
                <Box
                  sx={{
                    border: '1px solid rgba(255,255,255,0.12)',
                    borderRadius: 1,
                    p: 1.25,
                    backgroundColor: 'rgba(255,255,255,0.02)',
                  }}
                >
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                    {selectedTemplate.description}
                  </Typography>
                  <Typography variant="caption" sx={{ display: 'block' }}>
                    Includes:
                  </Typography>
                  {(selectedTemplate.files || []).map((f) => (
                    <Typography
                      key={f.name}
                      variant="caption"
                      sx={{ display: 'block', pl: 2, fontFamily: 'monospace', fontSize: '0.7rem' }}
                    >
                      • {f.name}
                    </Typography>
                  ))}
                </Box>
              )}
            </Box>
          )}

          {refTab === 1 && (
            <Box sx={{ mt: 1.5 }}>
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-start' }}>
                <TextField
                  size="small"
                  fullWidth
                  placeholder="/data/users/.../fbsource/fbcode/<…>/submit_job.py"
                  value={pendingPath}
                  onChange={(e) => setPendingPath(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      addPath();
                    }
                  }}
                />
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={<AddIcon />}
                  onClick={addPath}
                  sx={{ textTransform: 'none' }}
                >
                  Add
                </Button>
              </Box>
              {referenceScripts.length > 0 && (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, mt: 1 }}>
                  {referenceScripts.map((p) => (
                    <Box key={p} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="caption" sx={{ flex: 1, fontFamily: 'monospace', fontSize: '0.72rem' }}>
                        {p}
                      </Typography>
                      <IconButton size="small" onClick={() => removePath(p)}>
                        <CloseIcon fontSize="inherit" />
                      </IconButton>
                    </Box>
                  ))}
                </Box>
              )}
              {recentPaths.length > 0 && (
                <Box sx={{ mt: 1 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                    Recent paths:
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {recentPaths
                      .filter((p) => !referenceScripts.includes(p))
                      .slice(0, 5)
                      .map((p) => (
                        <Chip
                          key={p}
                          label={p.split('/').slice(-2).join('/')}
                          size="small"
                          variant="outlined"
                          onClick={() => useRecentPath(p)}
                          sx={{ fontSize: '0.65rem', maxWidth: 280 }}
                        />
                      ))}
                  </Box>
                </Box>
              )}
            </Box>
          )}
        </Box>

        <TextField
          label="Reference command (build / launch)"
          value={referenceCommand}
          onChange={(e) => setReferenceCommand(e.target.value)}
          multiline
          minRows={2}
          fullWidth
          size="small"
          placeholder="cd /data/users/.../fbsource/fbcode && app-layer main fire-app -d ..."
          helperText="Authoritative source for launch.json — buck target, cwd, extra args."
        />

        <TextField
          label="Additional instructions"
          value={additionalInstructions}
          onChange={(e) => setAdditionalInstructions(e.target.value)}
          multiline
          minRows={3}
          fullWidth
          size="small"
          placeholder="Hardware (GRANDTETON 8GPU), entitlement, MaaS config provider, …"
        />

        <Box
          sx={{
            border: '1px solid rgba(255,255,255,0.12)',
            borderRadius: 1,
            p: 1.5,
            backgroundColor: 'rgba(255,255,255,0.02)',
          }}
        >
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
            Context being shipped to PTI:
          </Typography>
          <Typography variant="caption" sx={{ display: 'block' }}>
            • Target codebase: <code>{config?.target_path || '(not set)'}</code>
          </Typography>
          <Typography variant="caption" sx={{ display: 'block' }}>
            • Selected hypotheses: {selectedHypothesisIds.length > 0 ? selectedHypothesisIds.join(', ') : '(none)'}
          </Typography>
          <HowCombosWork defaultExpanded={selectedHypothesisIds.length > 0 && flagWarnings.length > 0} />
          <Typography variant="caption" sx={{ display: 'block' }}>
            • Flag mapping (auto-derived; refine in Additional Instructions):
          </Typography>
          {selectedHypothesisIds.length > 0 ? (
            selectedHypothesisIds.map((hid) => {
              const flag = hypothesisFlagMap[hid] || '';
              const isBare = !flag.includes('.');
              return (
                <Typography
                  key={hid}
                  variant="caption"
                  sx={{
                    display: 'block',
                    pl: 2,
                    fontFamily: 'monospace',
                    fontSize: '0.7rem',
                    color: isBare ? 'warning.main' : 'inherit',
                  }}
                >
                  {hid} → {flag}{isBare ? '  ⚠ BARE (will be rejected at submit)' : ''}
                </Typography>
              );
            })
          ) : null}
          {flagWarnings.length > 0 && (
            <Alert severity="warning" sx={{ mt: 1, fontSize: '0.72rem' }}>
              <strong>Bare flag name(s) detected.</strong> Per the implementation
              contract, gin flag bindings must be SCOPED to a @gin.configurable
              class — bare names are top-level macros that do not bind to model
              fields, producing silent-no-op overlays at submit time.
              <br/>
              {flagWarnings.map((w, i) => (
                <span key={i} style={{ display: 'block', marginTop: 4 }}>• {w}</span>
              ))}
              <span style={{ display: 'block', marginTop: 6 }}>
                Edit the implementation's <code>proposalDetails</code> entry
                (or the Additional Instructions field below) to provide a scoped
                flag name like <code>hstu_encoder.enable_h17</code> before submitting.
              </span>
            </Alert>
          )}
        </Box>

        {error && (
          <Alert severity="error" sx={{ fontSize: '0.78rem' }}>
            {error}
          </Alert>
        )}
      </DialogContent>
      <DialogActions>
        {onUploadInsteadClick && (
          <Box sx={{ flex: 1, pl: 1 }}>
            <Link
              component="button"
              type="button"
              variant="caption"
              underline="hover"
              color="text.secondary"
              onClick={onUploadInsteadClick}
              disabled={submitting}
              sx={{ fontSize: '0.74rem' }}
            >
              Have an existing experiment runner script? Upload it instead →
            </Link>
          </Box>
        )}
        <Button onClick={onClose} disabled={submitting} sx={{ textTransform: 'none' }}>
          Cancel
        </Button>
        <Button
          variant="contained"
          onClick={handleSubmit}
          disabled={submitting}
          sx={{ textTransform: 'none' }}
        >
          {submitting ? 'Starting…' : '🚀 Create Experiment Runner'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
