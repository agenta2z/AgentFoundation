/**
 * StatusBadge — colored chip for any status string.
 *
 * Maps status values to semantic colors. Used across all views
 * for projects, tasks, employees, and agents.
 *
 * Props:
 *   status   - string (e.g. "in-progress", "active", "blocked")
 *   variant  - "filled" | "outlined" (default: "filled")
 *   size     - "small" | "medium" (default: "small")
 */

import React from 'react';
import Chip from '@mui/material/Chip';

const STATUS_COLORS = {
  // Project statuses
  'in-progress': { bg: '#1e3a5f', color: '#4a90d9', label: 'In Progress' },
  'planning': { bg: '#3e2723', color: '#ff9800', label: 'Planning' },
  'completed': { bg: '#1b5e20', color: '#4caf50', label: 'Completed' },
  'on-hold': { bg: '#37474f', color: '#90a4ae', label: 'On Hold' },

  // Task statuses
  'backlog': { bg: '#37474f', color: '#90a4ae', label: 'Backlog' },
  'in-review': { bg: '#4a148c', color: '#ce93d8', label: 'In Review' },
  'done': { bg: '#1b5e20', color: '#4caf50', label: 'Done' },
  'blocked': { bg: '#b71c1c', color: '#ef9a9a', label: 'Blocked' },

  // Employee statuses
  'active': { bg: '#1b5e20', color: '#4caf50', label: 'Active' },
  'idle': { bg: '#37474f', color: '#90a4ae', label: 'Idle' },
  'away': { bg: '#e65100', color: '#ffb74d', label: 'Away' },
  'queued': { bg: '#f57f17', color: '#fff176', label: 'Queued' },
};

function getStatusConfig(status) {
  const key = status?.toLowerCase() || '';
  return STATUS_COLORS[key] || { bg: '#37474f', color: '#90a4ae', label: status || 'Unknown' };
}

export function StatusBadge({ status, variant = 'filled', size = 'small' }) {
  const config = getStatusConfig(status);

  if (variant === 'outlined') {
    return (
      <Chip
        label={config.label}
        size={size}
        variant="outlined"
        sx={{ borderColor: config.color, color: config.color }}
      />
    );
  }

  return (
    <Chip
      label={config.label}
      size={size}
      sx={{ backgroundColor: config.bg, color: config.color, fontWeight: 500 }}
    />
  );
}

export default StatusBadge;
