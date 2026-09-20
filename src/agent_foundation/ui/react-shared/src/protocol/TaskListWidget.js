/**
 * TaskListWidget — compact task list with priority dots and progress.
 *
 * Shows a titled list of tasks, each with a colored priority indicator,
 * status badge, small progress bar, and assignee name.
 *
 * Props:
 *   data.title - string (e.g. "Sprint 4 Priorities")
 *   data.tasks - array of { title, status, priority, percent, assignee }
 */

import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import { useTheme } from '@mui/material/styles';

import { StatusBadge } from '../common/StatusBadge';
import { ProgressBar } from '../common/ProgressBar';

const PRIORITY_PALETTE = { high: 'error', medium: 'warning', low: 'success' };

function PriorityDot({ priority }) {
  const theme = useTheme();
  const paletteKey = PRIORITY_PALETTE[priority?.toLowerCase()] || 'neutral';
  const color = theme.palette[paletteKey]?.main || theme.palette.neutral.main;
  return (
    <Box
      sx={{
        width: 8,
        height: 8,
        borderRadius: '50%',
        backgroundColor: color,
        flexShrink: 0,
      }}
    />
  );
}

export default function TaskListWidget({ data }) {
  const theme = useTheme();
  const { title, tasks = [] } = data || {};

  const widgetBoxSx = {
    backgroundColor: theme.custom.surfaces.overlayLight,
    border: `1px solid ${theme.custom.surfaces.cardBorder}`,
    borderRadius: 2,
    p: 2,
    mt: 1.5,
  };

  return (
    <Box sx={widgetBoxSx}>
      {/* Header */}
      {title && (
        <Typography variant="subtitle2" sx={{ color: 'text.primary', fontWeight: 600, mb: 1.5 }}>
          {title}
        </Typography>
      )}

      {/* Task rows */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75 }}>
        {tasks.map((task, i) => (
          <Box
            key={i}
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1.5,
              py: 0.5,
              borderTop: i > 0 ? `1px solid ${theme.custom.surfaces.cardBorder}` : 'none',
            }}
          >
            <PriorityDot priority={task.priority} />
            <Typography
              variant="body2"
              sx={{ color: 'text.primary', flex: '1 1 auto', minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
            >
              {task.title}
            </Typography>
            <Box sx={{ flex: '0 0 auto' }}>
              <StatusBadge status={task.status} size="small" />
            </Box>
            <Box sx={{ flex: '0 0 90px' }}>
              <ProgressBar percent={task.percent || 0} height={4} showLabel />
            </Box>
            <Typography variant="caption" sx={{ color: 'text.secondary', flex: '0 0 80px', textAlign: 'right', whiteSpace: 'nowrap' }}>
              {task.assignee}
            </Typography>
          </Box>
        ))}
      </Box>
    </Box>
  );
}
