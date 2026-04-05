/**
 * StatusBadge — colored chip for any status string.
 *
 * Maps status values to semantic palette keys and derives colors
 * from the active MUI theme. Used across all views for projects,
 * tasks, employees, and agents.
 *
 * Props:
 *   status   - string (e.g. "in-progress", "active", "blocked")
 *   variant  - "filled" | "outlined" (default: "filled")
 *   size     - "small" | "medium" (default: "small")
 */

import React from 'react';
import Chip from '@mui/material/Chip';
import { useTheme, alpha } from '@mui/material/styles';

// Domain → palette mapping lives IN the component, not in the theme.
const STATUS_PALETTE = {
  'in-progress': 'primary',
  'planning':    'warning',
  'completed':   'success',
  'on-hold':     'neutral',
  'backlog':     'neutral',
  'in-review':   'info',
  'done':        'success',
  'blocked':     'error',
  'pending':     'secondary',
  'active':      'success',
  'idle':        'neutral',
  'away':        'warning',
  'queued':      'warning',
};

// Human-readable labels for known statuses.
const STATUS_LABELS = {
  'in-progress': 'In Progress',
  'planning':    'Planning',
  'completed':   'Completed',
  'on-hold':     'On Hold',
  'backlog':     'Backlog',
  'in-review':   'In Review',
  'done':        'Done',
  'blocked':     'Blocked',
  'pending':     'Pending',
  'active':      'Active',
  'idle':        'Idle',
  'away':        'Away',
  'queued':      'Queued',
};

function getStatusLabel(status) {
  const key = status?.toLowerCase() || '';
  return STATUS_LABELS[key] || status || 'Unknown';
}

export function StatusBadge({ status, variant = 'filled', size = 'small' }) {
  const theme = useTheme();
  const key = status?.toLowerCase() || '';
  const paletteKey = STATUS_PALETTE[key] || 'neutral';
  const color = theme.palette[paletteKey];
  const fg = color.main;
  const bg = alpha(color.main, 0.15);
  const label = getStatusLabel(status);

  if (variant === 'outlined') {
    return (
      <Chip
        label={label}
        size={size}
        variant="outlined"
        sx={{ borderColor: fg, color: fg }}
      />
    );
  }

  return (
    <Chip
      label={label}
      size={size}
      sx={{ backgroundColor: bg, color: fg, fontWeight: 500 }}
    />
  );
}

export default StatusBadge;
