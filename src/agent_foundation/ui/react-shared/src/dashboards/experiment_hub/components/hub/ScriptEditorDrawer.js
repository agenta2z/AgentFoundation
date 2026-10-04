/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * ScriptEditorDrawer — right-side drawer for viewing / editing the generated
 * experiment launcher. Save creates a NEW version (never overwrites).
 *
 * Ported from RankEvolve (components/hub/ScriptEditorDrawer.js). The
 * `useSession()` reads are replaced by injected props: `sessionId` and
 * `saveScriptVersion` (apiClient.saveScriptVersion). The per-version content
 * GET is a session/hub-scoped read (not part of the mutating apiClient
 * surface), so it talks to the REST route directly.
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Drawer,
  IconButton,
  MenuItem,
  Select,
  Tab,
  Tabs,
  Typography,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

const DRAWER_WIDTH = 720;

function pickLatestVersion(setup) {
  const versions = (setup && setup.scriptVersions) || [];
  if (versions.length === 0) return null;
  return versions.reduce((acc, v) => (
    !acc || (v.version || 0) > (acc.version || 0) ? v : acc
  ), null);
}

async function fetchVersion(multiTaskId, sessionId, version) {
  const res = await fetch(
    `/api/hubs/${encodeURIComponent(multiTaskId)}/submission-setup/script-version/${version}?session_id=${encodeURIComponent(sessionId)}`,
  );
  if (!res.ok) {
    throw new Error(`Failed to load version ${version}: ${res.status}`);
  }
  return res.json();
}

export default function ScriptEditorDrawer({ open, onClose, task, sessionId, saveScriptVersion }) {
  const setup = task?.submissionSetup || null;
  const versions = useMemo(
    () => (setup?.scriptVersions || []).slice().sort((a, b) => (a.version || 0) - (b.version || 0)),
    [setup?.scriptVersions],
  );
  const latest = useMemo(() => pickLatestVersion(setup), [setup]);

  const [selectedVersion, setSelectedVersion] = useState(latest?.version || null);
  const [scriptText, setScriptText] = useState('');
  const [originalScriptText, setOriginalScriptText] = useState('');
  const [launchText, setLaunchText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [info, setInfo] = useState('');
  const [saving, setSaving] = useState(false);
  const [mode, setMode] = useState(0); // 0=Edit, 1=View, 2=Diff vs PTI v1

  useEffect(() => {
    if (!open) return;
    setSelectedVersion(latest?.version || null);
    setError('');
    setInfo('');
    setMode(0);
  }, [open, latest?.version]);

  useEffect(() => {
    if (!open || !sessionId || !task?.id || !selectedVersion) return undefined;
    let cancelled = false;
    setLoading(true);
    fetchVersion(task.id, sessionId, selectedVersion)
      .then((v) => {
        if (cancelled) return;
        setScriptText(v.scriptContent || '');
        setOriginalScriptText(v.scriptContent || '');
        setLaunchText(v.launchContent || '');
        setError('');
      })
      .catch((e) => {
        if (cancelled) return;
        setError(String(e.message || e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, sessionId, task?.id, selectedVersion]);

  const dirty = scriptText !== originalScriptText;
  const lineCount = useMemo(() => scriptText.split('\n').length, [scriptText]);

  const handleSelectVersion = useCallback((event) => {
    setSelectedVersion(Number(event.target.value));
  }, []);

  const handleRevert = useCallback(() => {
    setScriptText(originalScriptText);
    setInfo('Reverted to last saved content.');
  }, [originalScriptText]);

  const handleSave = useCallback(async () => {
    if (!task?.id) return;
    if (!dirty) {
      setInfo('No changes to save.');
      return;
    }
    if (typeof saveScriptVersion !== 'function') {
      setError('Save unavailable (apiClient.saveScriptVersion not provided).');
      return;
    }
    setSaving(true);
    setError('');
    setInfo('');
    try {
      const res = await saveScriptVersion(task.id, scriptText, launchText);
      if (res?.error) {
        setError(`Save failed: ${res.detail || res.error}`);
      } else {
        const latestAfterSave = pickLatestVersion(res.setup);
        if (latestAfterSave) {
          setSelectedVersion(latestAfterSave.version);
          setOriginalScriptText(scriptText);
          setInfo(`Saved as v${latestAfterSave.version}.`);
        } else {
          setInfo('Saved.');
        }
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  }, [task?.id, dirty, saveScriptVersion, scriptText, launchText]);

  const ptiBaseline = useMemo(() => {
    return versions.find((v) => v.source === 'pti') || null;
  }, [versions]);

  const diffBaseline = useMemo(() => {
    if (ptiBaseline) return ptiBaseline;
    return versions[0] || null;
  }, [ptiBaseline, versions]);
  const diffTabLabel = diffBaseline
    ? `Diff vs v${diffBaseline.version} (${diffBaseline.source || 'unknown'})`
    : 'Diff vs v1';

  const [diffBaselineText, setDiffBaselineText] = useState('');
  useEffect(() => {
    if (mode !== 2 || !diffBaseline || !sessionId || !task?.id) return undefined;
    let cancelled = false;
    fetchVersion(task.id, sessionId, diffBaseline.version)
      .then((v) => {
        if (!cancelled) setDiffBaselineText(v.scriptContent || '');
      })
      .catch(() => {
        if (!cancelled) setDiffBaselineText('');
      });
    return () => {
      cancelled = true;
    };
  }, [mode, diffBaseline, sessionId, task?.id]);

  const isViewingOlder = useMemo(() => {
    if (!latest || !selectedVersion) return false;
    return selectedVersion < latest.version;
  }, [latest, selectedVersion]);
  const nextVersionNumber = (latest?.version || 0) + 1;
  const handleRestore = useCallback(async () => {
    if (!task?.id || !selectedVersion || typeof saveScriptVersion !== 'function') return;
    setSaving(true);
    setError('');
    setInfo('');
    try {
      const v = await fetchVersion(task.id, sessionId, selectedVersion);
      const res = await saveScriptVersion(
        task.id,
        v.scriptContent || '',
        v.launchContent || '',
      );
      if (res?.error) {
        setError(`Restore failed: ${res.detail || res.error}`);
      } else {
        const latestAfter = pickLatestVersion(res.setup);
        if (latestAfter) {
          setSelectedVersion(latestAfter.version);
          setInfo(`Restored as v${latestAfter.version}.`);
        } else {
          setInfo('Restored.');
        }
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  }, [task?.id, selectedVersion, sessionId, saveScriptVersion]);

  return (
    <Drawer
      anchor="right"
      open={open}
      onClose={onClose}
      PaperProps={{ sx: { width: DRAWER_WIDTH, maxWidth: '90vw' } }}
    >
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <Box sx={{ px: 2, py: 1.5, borderBottom: '1px solid rgba(255,255,255,0.08)', display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography sx={{ fontSize: '0.95rem', fontWeight: 600, flex: 1 }}>
            {setup?.setupName ? `Setup: ${setup.setupName}` : 'Experiment runner'}
          </Typography>
          <IconButton size="small" onClick={onClose}>
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>

        <Box sx={{ px: 2, py: 1, borderBottom: '1px solid rgba(255,255,255,0.06)', display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
          <Typography variant="caption" color="text.secondary">Version:</Typography>
          <Select
            size="small"
            value={selectedVersion || ''}
            onChange={handleSelectVersion}
            sx={{ minWidth: 120, fontSize: '0.78rem' }}
          >
            {versions.map((v) => (
              <MenuItem key={v.version} value={v.version} sx={{ fontSize: '0.78rem' }}>
                v{v.version} ({v.source || 'unknown'})
              </MenuItem>
            ))}
          </Select>
          <Box sx={{ flex: 1 }} />
          <Tabs
            value={mode}
            onChange={(_e, v) => setMode(v)}
            sx={{ minHeight: 32, '& .MuiTab-root': { minHeight: 32, fontSize: '0.74rem', textTransform: 'none' } }}
          >
            <Tab label="Edit" />
            <Tab label="View" />
            <Tab label={diffTabLabel} disabled={!diffBaseline} />
          </Tabs>
          {isViewingOlder && (
            <Button
              variant="outlined"
              size="small"
              onClick={handleRestore}
              disabled={saving}
              sx={{ ml: 1, fontSize: '0.74rem', textTransform: 'none' }}
            >
              {saving ? 'Restoring…' : `Restore as new v${nextVersionNumber}`}
            </Button>
          )}
        </Box>

        <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
          {loading && (
            <Typography variant="caption" color="text.secondary">Loading…</Typography>
          )}
          {error && (
            <Alert severity="error" sx={{ mb: 1, fontSize: '0.78rem' }}>{error}</Alert>
          )}
          {info && (
            <Alert severity="info" sx={{ mb: 1, fontSize: '0.78rem' }}>{info}</Alert>
          )}

          {mode === 2 && diffBaseline ? (
            <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1, height: '100%' }}>
              <Box>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                  v{diffBaseline.version} ({diffBaseline.source || 'unknown'})
                </Typography>
                <Box component="pre" sx={{
                  m: 0, p: 1, fontSize: '0.72rem', fontFamily: 'monospace',
                  backgroundColor: 'rgba(255,255,255,0.02)', overflow: 'auto', height: '60vh',
                }}>{diffBaselineText}</Box>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                  Current (v{selectedVersion})
                </Typography>
                <Box component="pre" sx={{
                  m: 0, p: 1, fontSize: '0.72rem', fontFamily: 'monospace',
                  backgroundColor: 'rgba(255,255,255,0.02)', overflow: 'auto', height: '60vh',
                }}>{scriptText}</Box>
              </Box>
            </Box>
          ) : (
            <Box
              component="textarea"
              value={scriptText}
              onChange={(e) => setScriptText(e.target.value)}
              readOnly={mode === 1}
              spellCheck={false}
              sx={{
                width: '100%',
                minHeight: '60vh',
                fontFamily: 'Menlo, Consolas, monospace',
                fontSize: '0.78rem',
                lineHeight: 1.4,
                color: 'inherit',
                backgroundColor: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: 1,
                p: 1.25,
                resize: 'vertical',
                outline: 'none',
              }}
            />
          )}

          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
            {scriptText.length} chars · {lineCount} lines · {dirty ? 'unsaved changes' : 'saved'}
          </Typography>

          <Box sx={{ mt: 2 }}>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
              launch.json (shared across versions; Re-generate via PTI to change buck target)
            </Typography>
            <Box
              component="pre"
              sx={{
                m: 0, p: 1, fontSize: '0.7rem', fontFamily: 'monospace',
                backgroundColor: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255,255,255,0.06)',
                borderRadius: 1,
                overflow: 'auto', maxHeight: '20vh',
              }}
            >
              {launchText}
            </Box>
          </Box>
        </Box>

        <Box sx={{ px: 2, py: 1.5, borderTop: '1px solid rgba(255,255,255,0.08)', display: 'flex', gap: 1 }}>
          <Button
            size="small"
            variant="contained"
            onClick={handleSave}
            disabled={saving || !dirty || mode === 1}
            sx={{ textTransform: 'none' }}
          >
            {saving ? 'Saving…' : 'Save (creates new version)'}
          </Button>
          <Button
            size="small"
            variant="outlined"
            onClick={handleRevert}
            disabled={saving || !dirty}
            sx={{ textTransform: 'none' }}
          >
            Revert
          </Button>
          <Box sx={{ flex: 1 }} />
          <Button size="small" onClick={onClose} sx={{ textTransform: 'none' }}>
            Close
          </Button>
        </Box>
      </Box>
    </Drawer>
  );
}
