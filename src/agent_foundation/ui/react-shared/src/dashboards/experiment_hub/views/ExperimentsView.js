/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * ExperimentsView — the Experiment Hub's "Experiments" tab. Composes the
 * ported JobMonitorView (← grouped run lifecycle + AccumulatedLearningsDrawer +
 * the `hub/` Run/baseline flow). Maps the generic dashboard view props to the
 * ported component's `task` prop and threads `sessionId`, `dispatch`, and the
 * injected `apiClient`.
 */

import React, { useCallback } from 'react';
import { Box, Button, Tooltip } from '@mui/material';
import DoneAllIcon from '@mui/icons-material/DoneAll';
import JobMonitorView from '../components/views/JobMonitorView';

export default function ExperimentsView({
  dashboardState, config, dispatch, sessionId, hubId, apiClient,
}) {
  // R4 — "Done — summarize & evolve". Emits a generic terminal signal
  // (completeEvolution) that the HOST maps to the real Phase-3 completion
  // (writes the declared Phase-3 output + _check_phase_completion → 3→3b). The
  // hub stays SOP-agnostic; guarded so a host that doesn't provide the method
  // (standalone AF webui / no SOP) simply doesn't render the control.
  const resolvedHubId = hubId || dashboardState?.id || dashboardState?.multiTaskId || null;
  const hasComplete = !!(apiClient && typeof apiClient.completeEvolution === 'function');
  const onDone = useCallback(() => {
    if (apiClient && typeof apiClient.completeEvolution === 'function') {
      apiClient.completeEvolution(resolvedHubId);
    }
  }, [apiClient, resolvedHubId]);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
      <Box sx={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
        <JobMonitorView
          task={dashboardState}
          config={config || {}}
          dispatch={dispatch}
          sessionId={sessionId}
          apiClient={apiClient}
        />
      </Box>
      {hasComplete && (
        <Box
          sx={{
            flexShrink: 0,
            display: 'flex',
            justifyContent: 'flex-end',
            gap: 1,
            p: 1.5,
            borderTop: 1,
            borderColor: 'divider',
          }}
        >
          <Tooltip
            title="Mark the evolution cycle complete — the conversation summarizes results and decides whether to continue evolving."
            placement="top"
          >
            <Button
              variant="contained"
              size="small"
              color="success"
              startIcon={<DoneAllIcon />}
              onClick={onDone}
              sx={{ textTransform: 'none' }}
            >
              Done — summarize &amp; evolve
            </Button>
          </Tooltip>
        </Box>
      )}
    </Box>
  );
}
