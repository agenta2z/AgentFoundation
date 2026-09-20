/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * DashboardPanel — generic, context-free multi-view container for a dashboard
 * subtab. Ported from RankEvolve's `agent/MultiViewTaskPanel`, but with NO
 * `useSession()` dependency: it runs its own `useReducer`, subscribes to a
 * per-hub WS event stream (`wsEvents$`), and threads
 * `{dashboardState, dispatch, sessionId, hubId, apiClient, wsEvents$, config}`
 * to each view. The tab manifest is supplied by the host (canonically the
 * Python `dashboard_config.view_manifest` delivered in `dashboard_open`).
 */

import React, { useReducer, useEffect, useCallback, useMemo } from 'react';
import { Alert, Box, Button, Chip, Typography } from '@mui/material';
import { ArrowBack as BackIcon } from '@mui/icons-material';
import ViewTabBar from '../nav/ViewTabBar';
import PipelineStatusBar from './PipelineStatusBar';
import { getView } from './ViewRegistry';
import { getDashboard, createDashboardReducer } from './DashboardRegistry';
import { normalizeManifest } from './manifest';
import { evaluateViewRules, getPipelineStages, getDashboardStatus } from './lifecycle';

const _fallbackReducer = createDashboardReducer();

function readSavedActiveView(key) {
  try {
    if (typeof window === 'undefined' || !window.localStorage) return null;
    const raw = window.localStorage.getItem(key);
    return raw == null ? null : parseInt(raw, 10);
  } catch (_e) { return null; }
}
function saveActiveView(key, idx) {
  try {
    if (typeof window === 'undefined' || !window.localStorage) return;
    window.localStorage.setItem(key, String(idx));
  } catch (_e) { /* ignore */ }
}

export default function DashboardPanel({
  dashboardId,
  manifest: manifestProp,
  reducer: reducerProp,
  initialState: initialStateProp,
  sessionId,
  hubId,
  apiClient,
  wsEvents$,
  onBack,
}) {
  const registered = dashboardId ? getDashboard(dashboardId) : null;
  const reducer = reducerProp || (registered && registered.reducer) || _fallbackReducer;
  const manifest = useMemo(
    () => normalizeManifest(manifestProp || (registered && registered.manifest)),
    [manifestProp, registered]
  );

  const storageKey = `ot_hub_view_${sessionId || 'na'}_${hubId || dashboardId || 'na'}`;
  const initialState = useMemo(() => {
    const seed = initialStateProp || (registered && registered.initialState) || {};
    const merged = { activeView: 0, unlockedViewIds: [0], ...seed };
    const saved = readSavedActiveView(storageKey);
    if (saved != null && !Number.isNaN(saved)) merged.activeView = saved;
    const unlocked = new Set(merged.unlockedViewIds || [0]);
    unlocked.add(merged.activeView || 0);
    merged.unlockedViewIds = Array.from(unlocked).sort((a, b) => a - b);
    return merged;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const [state, dispatch] = useReducer(reducer, initialState);

  // Live updates: feed the per-hub WS event stream into the reducer.
  useEffect(() => {
    if (!wsEvents$ || typeof wsEvents$.subscribe !== 'function') return undefined;
    const unsub = wsEvents$.subscribe((event) => dispatch({ type: 'DASHBOARD_EVENT', event }));
    return typeof unsub === 'function' ? unsub : undefined;
  }, [wsEvents$]);

  // Progressive unlock / auto-activate (newly-unlockable only → no render loop).
  useEffect(() => {
    const { viewsToUnlock, viewToActivate } = evaluateViewRules(state, manifest.views);
    viewsToUnlock.forEach((i) => dispatch({ type: 'UNLOCK_VIEW', viewIndex: i }));
    if (viewToActivate != null) dispatch({ type: 'SET_ACTIVE_VIEW', viewIndex: viewToActivate });
  }, [state, manifest.views]);

  // Persist the active view per (session, hub).
  useEffect(() => { saveActiveView(storageKey, state.activeView || 0); }, [state.activeView, storageKey]);

  const handleViewSwitch = useCallback((viewIndex) => {
    dispatch({ type: 'SET_ACTIVE_VIEW', viewIndex });
  }, []);

  const activeViewIndex = Math.min(state.activeView || 0, Math.max(manifest.views.length - 1, 0));
  const activeViewDef = manifest.views[activeViewIndex];
  const viewReg = activeViewDef ? getView(activeViewDef.type) : null;
  const ActiveViewComponent = viewReg && viewReg.component;

  const viewDefs = manifest.views.map((v, i) => {
    const reg = getView(v.type);
    return {
      label: v.label || (v.config && v.config.label) || (reg && reg.defaultLabel) || `View ${i}`,
      isUnlocked: (state.unlockedViewIds || [0]).includes(i),
      isActive: i === activeViewIndex,
    };
  });

  const pipelineStages = manifest.pipeline ? getPipelineStages(state, manifest.pipeline) : null;
  const statusInfo = getDashboardStatus(state);
  const notices = (state.metadata && state.metadata.notices) || [];

  const viewProps = {
    dashboardState: state,
    config: (activeViewDef && activeViewDef.config) || {},
    dispatch,
    sessionId,
    hubId,
    apiClient,
    wsEvents$,
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {(manifest.label || onBack) && (
        <Box sx={{
          px: 2, py: 1, borderBottom: '1px solid', borderColor: 'divider',
          display: 'flex', alignItems: 'center', gap: 2, backgroundColor: 'rgba(0,0,0,0.2)',
        }}>
          {onBack && (
            <Button size="small" startIcon={<BackIcon />} onClick={onBack}
              sx={{ textTransform: 'none', fontSize: '0.8rem' }}>
              Back to session
            </Button>
          )}
          <Typography variant="subtitle2" sx={{
            fontWeight: 600, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {manifest.icon && `${manifest.icon} `}{manifest.label || 'Dashboard'}
          </Typography>
          {statusInfo && statusInfo.label && (
            <Chip
              label={`${statusInfo.icon || ''} ${statusInfo.label}`.trim()}
              size="small" color={statusInfo.color || 'default'} variant="outlined"
              sx={{ height: 22, fontSize: '0.7rem' }}
            />
          )}
        </Box>
      )}

      {Array.isArray(notices) && notices.length > 0 && (
        <Box sx={{ px: 2, py: 1 }}>
          {notices.map((n, i) => (
            <Alert key={(n && n.code) || i} severity="info" sx={{ fontSize: '0.78rem', mb: 0.5 }}>
              {(n && n.message) || ''}
            </Alert>
          ))}
        </Box>
      )}

      {pipelineStages && manifest.pipeline && (
        <PipelineStatusBar stages={manifest.pipeline} statuses={pipelineStages} />
      )}

      {manifest.views.length > 1 && (
        <ViewTabBar views={viewDefs} activeIndex={activeViewIndex} onSwitch={handleViewSwitch} />
      )}

      <Box sx={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {ActiveViewComponent ? (
          <ActiveViewComponent {...viewProps} />
        ) : (
          <Box sx={{ p: 2 }}>
            <Typography color="text.secondary">
              Unknown view type: {activeViewDef && activeViewDef.type}
            </Typography>
          </Box>
        )}
      </Box>
    </Box>
  );
}
