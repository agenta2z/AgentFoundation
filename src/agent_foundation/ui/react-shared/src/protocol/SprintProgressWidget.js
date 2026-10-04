/**
 * SprintProgressWidget — renders sprint task breakdown with progress bars.
 *
 * Dark card showing sprint label, overall progress, and a table of tasks
 * with assignee, status badge, and individual progress bars.
 *
 * Props:
 *   data.sprint           - string, e.g. "Sprint 3 of 4"
 *   data.overall_percent  - number 0-100
 *   data.tasks            - array of { title, assignee, assignee_type, status, percent }
 */

import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import { useTheme } from '@mui/material/styles';

import { StatusBadge } from '../common/StatusBadge';
import { ProgressBar } from '../common/ProgressBar';

function AssigneeLabel({ name, type }) {
  const icon = type === 'ai' ? '\u{1F916}' : '\u{1F464}';
  return (
    <Typography variant="body2" sx={{ color: 'text.primary', whiteSpace: 'nowrap' }}>
      {icon} {name}
    </Typography>
  );
}

export default function SprintProgressWidget({ data }) {
  const theme = useTheme();
  const { sprint, overall_percent = 0, tasks = [] } = data || {};

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
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 1.5 }}>
        <Typography variant="subtitle2" sx={{ color: 'text.primary', fontWeight: 600, whiteSpace: 'nowrap' }}>
          {typeof sprint === 'object' ? `Sprint ${sprint.current} of ${sprint.total}` : sprint}
        </Typography>
        <Box sx={{ flexGrow: 1 }}>
          <ProgressBar percent={overall_percent} height={6} />
        </Box>
      </Box>

      {/* Task rows */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        {tasks.map((task, i) => (
          <Box
            key={i}
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1.5,
              py: 0.75,
              borderTop: i > 0 ? `1px solid ${theme.custom.surfaces.cardBorder}` : 'none',
            }}
          >
            <Typography
              variant="body2"
              sx={{ color: 'text.primary', flex: '1 1 auto', minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
            >
              {task.title}
            </Typography>
            <Box sx={{ flex: '0 0 120px' }}>
              <AssigneeLabel name={task.assignee} type={task.assignee_type} />
            </Box>
            <Box sx={{ flex: '0 0 auto' }}>
              <StatusBadge status={task.status} size="small" />
            </Box>
            <Box sx={{ flex: '0 0 120px' }}>
              <ProgressBar percent={task.percent || 0} height={5} showLabel />
            </Box>
          </Box>
        ))}
      </Box>
    </Box>
  );
}
