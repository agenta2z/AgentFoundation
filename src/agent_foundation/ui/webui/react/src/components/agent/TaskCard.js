/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * TaskCard — Inline task reference card rendered in session chat.
 * Shows task label, status chip, and "Open Task" button.
 */

import React from 'react';
import { Box, Paper, Typography, Chip, Button, Tooltip } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { OpenInNew as OpenIcon } from '@mui/icons-material';
import { useSession } from '../../contexts/SessionContext';

const STATUS_CONFIG = {
  starting: { label: 'Starting...', color: 'warning' },
  running: { label: 'Running', color: 'info' },
  completed: { label: 'Complete', color: 'success' },
  error: { label: 'Error', color: 'error' },
};

const AGENT_LABELS = {
  review: 'Review Agent',
  base: 'Base Agent',
  plan: 'Plan Agent',
  implementation: 'Implementation Agent',
};

export function TaskCard({ taskId, label, status }) {
  const theme = useTheme();
  const { switchTab, tasks } = useSession();
  const task = tasks[taskId];
  let statusInfo = STATUS_CONFIG[status] || STATUS_CONFIG.running;
  if (task?.currentAgent && status !== 'completed' && status !== 'error') {
    const agentLabel = AGENT_LABELS[task.currentAgent] || task.currentAgent;
    statusInfo = { label: agentLabel, color: 'info' };
  }
  const displayLabel = label && label.length > 50 ? label.slice(0, 50) + '...' : label;

  return (
    <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
      <Paper
        elevation={0}
        sx={{
          p: 2,
          maxWidth: '95%',
          backgroundColor: theme.custom.surfaces.highlightSubtle,
          borderRadius: 2,
          border: '1px solid',
          borderColor: 'primary.dark',
          display: 'flex',
          alignItems: 'center',
          gap: 2,
        }}
      >
        <Tooltip title={label || ''} placement="top" arrow>
          <Typography variant="body2" sx={{ fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 400 }}>
            Task: {displayLabel}
          </Typography>
        </Tooltip>
        <Chip
          label={statusInfo.label}
          size="small"
          color={statusInfo.color}
          variant="outlined"
          sx={{ height: 22, fontSize: '0.7rem' }}
        />
        <Button
          size="small"
          endIcon={<OpenIcon sx={{ fontSize: 14 }} />}
          onClick={() => switchTab(taskId, 'task')}
          sx={{
            fontSize: '0.75rem',
            textTransform: 'none',
            ml: 'auto',
          }}
        >
          Open Task
        </Button>
      </Paper>
    </Box>
  );
}

export default TaskCard;
