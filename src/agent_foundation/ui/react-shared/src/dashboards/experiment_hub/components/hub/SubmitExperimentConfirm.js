/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * SubmitExperimentConfirm — confirmation modal opened by SubmissionFooterBar
 * before runSubmission is called. Lets the user audit/edit the comma-separated
 * config field names passed to `submit_v<n>.py --enable-flags`.
 *
 * Ported verbatim from RankEvolve (components/hub/SubmitExperimentConfirm.js).
 */

import React, { useEffect, useMemo, useState } from 'react';
import {
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  TextField,
  Typography,
} from '@mui/material';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';

function basename(path) {
  if (!path) return '';
  const parts = String(path).split('/');
  return parts[parts.length - 1] || '';
}

function parseFlagsInput(text) {
  return String(text || '')
    .split(',')
    .map((tok) => tok.trim())
    .filter(Boolean);
}

export default function SubmitExperimentConfirm({
  open,
  onCancel,
  onConfirm,
  combo,
  defaultFlags,
  defaultExperimentName,
  defaultAppLayerVersion,
  hasBuildCommand = false,
  scriptPath,
  selectedCombos = null,
  hypothesisFlagMap = {},
}) {
  const [flagsText, setFlagsText] = useState('');
  const [experimentName, setExperimentName] = useState('');
  const [appLayerVersion, setAppLayerVersion] = useState('');

  useEffect(() => {
    if (!open) return;
    setFlagsText(Array.isArray(defaultFlags) ? defaultFlags.join(',') : '');
    setExperimentName(defaultExperimentName || '');
    setAppLayerVersion(defaultAppLayerVersion || '');
  }, [open, defaultFlags, defaultExperimentName, defaultAppLayerVersion]);

  const parsedFlags = useMemo(() => parseFlagsInput(flagsText), [flagsText]);

  const willAutoBuild = hasBuildCommand && appLayerVersion.trim().length === 0;

  const previewCommand = useMemo(() => {
    const script = basename(scriptPath) || 'submit_v1.py';
    const flagsArg = parsedFlags.join(',');
    const expArg = experimentName || '<empty>';
    if (willAutoBuild) {
      return (
        `Phase 1 (auto-build): app-layer ... (captures fire-app:<sha>)\n`
        + `Phase 2 (submit): ${script} --enable-flags ${flagsArg || '<empty>'} `
        + `--experiment-name ${expArg} --app-layer-version <auto-built>`
      );
    }
    return (
      `${script} --enable-flags ${flagsArg || '<empty>'} `
      + `--experiment-name ${expArg} --app-layer-version `
      + `${appLayerVersion || '<empty>'}`
    );
  }, [scriptPath, parsedFlags, experimentName, appLayerVersion, willAutoBuild]);

  const canConfirm =
    parsedFlags.length > 0
    && experimentName.trim().length > 0
    && (hasBuildCommand || appLayerVersion.trim().length > 0);

  const handleConfirm = () => {
    const isBatchNow = Array.isArray(selectedCombos) && selectedCombos.filter(Boolean).length > 0;
    const ok = isBatchNow
      ? (experimentName.trim().length > 0
         && (hasBuildCommand || appLayerVersion.trim().length > 0))
      : canConfirm;
    if (!ok) return;
    onConfirm({
      enableFlags: parsedFlags,  // ignored on the batch path
      experimentName: experimentName.trim(),
      appLayerVersion: appLayerVersion.trim(),
    });
  };

  const comboItems = Array.isArray(combo) ? combo : [];
  const comboBatch = Array.isArray(selectedCombos)
    ? selectedCombos.filter(Boolean)
    : [];
  const isBatch = comboBatch.length > 0;

  const canConfirmBatch =
    isBatch
    && experimentName.trim().length > 0
    && (hasBuildCommand || appLayerVersion.trim().length > 0);
  const confirmEnabled = isBatch ? canConfirmBatch : canConfirm;

  return (
    <Dialog open={!!open} onClose={onCancel} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ pb: 1 }}>
        {isBatch ? `Submit ${comboBatch.length} Experiments` : 'Submit Experiment'}
      </DialogTitle>
      <DialogContent sx={{ pt: 1 }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {isBatch ? (
            <Box>
              <Typography variant="caption" color="text.secondary">
                {comboBatch.length} combo{comboBatch.length === 1 ? '' : 's'} will be submitted
                (one experiment row per combo, run with max-concurrency
                from the Review &amp; Combos tab)
              </Typography>
              <Box sx={{
                mt: 0.5, maxHeight: 220, overflow: 'auto',
                border: '1px solid rgba(255,255,255,0.08)', borderRadius: 1,
              }}>
                {comboBatch.map((c, idx) => {
                  const items = c?.selectedItems || [];
                  const resolvedFlags = items.map(
                    hid => hypothesisFlagMap[hid] || `enable_${String(hid).toLowerCase()}`
                  );
                  const hasBareFlag = resolvedFlags.some(f => !f.includes('.'));
                  return (
                    <Box
                      key={c?.comboId || idx}
                      sx={{
                        px: 1, py: 0.75,
                        borderBottom: idx < comboBatch.length - 1
                          ? '1px solid rgba(255,255,255,0.04)' : 'none',
                      }}
                    >
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, alignItems: 'center' }}>
                        <Typography variant="body2" sx={{ fontWeight: 600, fontSize: '0.78rem', mr: 0.5 }}>
                          {c?.comboId || '?'}
                        </Typography>
                        {items.map(h => (
                          <Chip
                            key={h}
                            size="small"
                            label={h}
                            variant="outlined"
                            sx={{ fontSize: '0.66rem', height: 18 }}
                          />
                        ))}
                        {(c?.applyState && c.applyState !== 'ready') && (
                          <Chip
                            size="small"
                            color="warning"
                            variant="outlined"
                            label={c.applyState}
                            sx={{ fontSize: '0.62rem', height: 18, ml: 0.5 }}
                          />
                        )}
                      </Box>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, alignItems: 'center', mt: 0.5, pl: 1.5 }}>
                        <Typography
                          variant="caption"
                          sx={{ fontSize: '0.66rem', color: 'text.secondary', mr: 0.5 }}
                        >
                          flags:
                        </Typography>
                        {resolvedFlags.length === 0 ? (
                          <Typography variant="caption" sx={{ fontSize: '0.66rem', color: 'text.secondary' }}>
                            (baseline — no flags)
                          </Typography>
                        ) : (
                          resolvedFlags.map((f, fi) => (
                            <Chip
                              key={`${f}_${fi}`}
                              size="small"
                              label={f}
                              variant={f.includes('.') ? 'filled' : 'outlined'}
                              color={f.includes('.') ? 'default' : 'warning'}
                              sx={{ fontSize: '0.62rem', height: 18, fontFamily: 'monospace' }}
                              title={!f.includes('.')
                                ? 'BARE flag (no scope) — runner will reject this at submit time. Edit the per-hub hypothesisFlagMap to provide a scoped name like "hstu_encoder.enable_x".'
                                : undefined}
                            />
                          ))
                        )}
                      </Box>
                      {hasBareFlag && (
                        <Typography
                          variant="caption"
                          sx={{ fontSize: '0.62rem', color: 'warning.main', display: 'block', pl: 1.5, mt: 0.25 }}
                        >
                          ⚠ Contains BARE flag(s); runner will reject at submit time. Fix the per-hub hypothesisFlagMap.
                        </Typography>
                      )}
                    </Box>
                  );
                })}
              </Box>
            </Box>
          ) : (
            <Box>
              <Typography variant="caption" color="text.secondary">
                Hypotheses in this combo
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                {comboItems.length === 0 ? (
                  <Typography variant="body2" color="text.secondary">
                    (none)
                  </Typography>
                ) : (
                  comboItems.map((id) => (
                    <Chip
                      key={id}
                      size="small"
                      label={id}
                      variant="outlined"
                      sx={{ fontSize: '0.72rem', height: 22 }}
                    />
                  ))
                )}
              </Box>
            </Box>
          )}

          {!isBatch && (
            <TextField
              label="Flags to enable"
              value={flagsText}
              onChange={(e) => setFlagsText(e.target.value)}
              multiline
              minRows={2}
              fullWidth
              helperText="Comma-separated config field names — the script will set each to True. Verify against the model's config dataclass; the wizard's default is a best-effort synthesis."
              sx={{ '& .MuiInputBase-input': { fontFamily: 'monospace', fontSize: '0.85rem' } }}
            />
          )}

          <TextField
            label={isBatch ? 'Experiment name prefix' : 'Experiment name'}
            value={experimentName}
            onChange={(e) => setExperimentName(e.target.value)}
            fullWidth
            helperText={
              isBatch
                ? 'Each combo runs as a separate FBLearner job named "<prefix>_<comboId>" — combo IDs disambiguate.'
                : 'Used for FBLearner job naming; concurrent runs must differ.'
            }
          />

          <TextField
            label={hasBuildCommand ? 'App-layer version (optional)' : 'App-layer version'}
            value={appLayerVersion}
            onChange={(e) => setAppLayerVersion(e.target.value)}
            placeholder="fire-app:2941a32"
            fullWidth
            required={!hasBuildCommand}
            helperText={
              hasBuildCommand
                ? 'Leave empty to auto-build a fresh fbpkg from the setup\'s reference command (~49s on warm cache). Provide a value to skip the build (fast iteration on a known fbpkg).'
                : 'The fbpkg version from your `app-layer main fire-app` build. Defaults to the last value you used.'
            }
            sx={{ '& .MuiInputBase-input': { fontFamily: 'monospace', fontSize: '0.85rem' } }}
          />

          {!isBatch && (
            <Box
              sx={{
                borderRadius: 1,
                border: '1px solid rgba(255,255,255,0.08)',
                p: 1,
                backgroundColor: 'rgba(0,0,0,0.18)',
              }}
            >
              <Typography variant="caption" color="text.secondary">
                Will spawn
              </Typography>
              <Typography
                variant="body2"
                sx={{
                  fontFamily: 'monospace',
                  fontSize: '0.78rem',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-all',
                  mt: 0.5,
                }}
              >
                {previewCommand}
              </Typography>
            </Box>
          )}
        </Box>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button onClick={onCancel} sx={{ textTransform: 'none' }}>
          Cancel
        </Button>
        <Button
          variant="contained"
          startIcon={<RocketLaunchIcon />}
          onClick={handleConfirm}
          disabled={!confirmEnabled}
          sx={{ textTransform: 'none', fontWeight: 600 }}
        >
          {isBatch ? `Confirm Submit (${comboBatch.length})` : 'Confirm Submit'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
