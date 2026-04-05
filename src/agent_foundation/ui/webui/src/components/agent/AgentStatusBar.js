/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * AgentStatusBar — Shows connection status, model, target path, and task phase.
 */

import React, { useState, useEffect, useRef } from 'react';
import { Box, Typography, Chip } from '@mui/material';

export function AgentStatusBar({ connectionStatus, model, targetPath, taskPhase, isStreaming }) {
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef(null);

  useEffect(() => {
    if (isStreaming) {
      setElapsed(0);
      timerRef.current = setInterval(() => {
        setElapsed(prev => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isStreaming]);

  const connectionIcon = connectionStatus === 'connected' ? '🟢' : connectionStatus === 'connecting' ? '🟡' : '🔴';
  const connectionLabel = connectionStatus === 'connected' ? 'Connected' : connectionStatus === 'connecting' ? 'Connecting...' : 'Disconnected';

  const formatElapsed = (seconds) => {
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  const phaseLabel = taskPhase ? {
    plan: '📋 Planning',
    implementation: '🔧 Implementation',
    analysis: '🔍 Analysis',
    complete: '✅ Complete',
    cancelled: '⚪ Cancelled',
    error: '❌ Error',
  }[taskPhase.phase] || taskPhase.phase : null;

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        px: 2,
        py: 0.75,
        backgroundColor: 'rgba(0, 0, 0, 0.3)',
        borderBottom: '1px solid',
        borderColor: 'divider',
        flexWrap: 'wrap',
      }}
    >
      <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
        {connectionIcon} {connectionLabel}
      </Typography>

      {model && (
        <Chip
          label={model}
          size="small"
          variant="outlined"
          sx={{ height: 22, fontSize: '0.7rem', borderColor: 'rgba(255,255,255,0.2)' }}
        />
      )}

      {targetPath && (
        <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>
          📁 {targetPath}
        </Typography>
      )}

      {phaseLabel && (
        <Chip
          label={phaseLabel}
          size="small"
          color={taskPhase?.state === 'running' ? 'primary' : 'default'}
          sx={{ height: 22, fontSize: '0.7rem' }}
        />
      )}

      {isStreaming && (
        <Typography variant="caption" sx={{ color: 'text.secondary', ml: 'auto' }}>
          ⏱ {formatElapsed(elapsed)}
        </Typography>
      )}
    </Box>
  );
}

export default AgentStatusBar;
