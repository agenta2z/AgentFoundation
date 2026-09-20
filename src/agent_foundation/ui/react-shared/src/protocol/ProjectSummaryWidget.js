/**
 * ProjectSummaryWidget — compact project status card.
 *
 * Shows project name, status badge, progress bar, team composition,
 * sprint info, and optional blocker count.
 *
 * Props:
 *   data.name             - string
 *   data.status           - string (e.g. "in-progress")
 *   data.progress_percent - number 0-100
 *   data.sprint           - string (e.g. "Sprint 3 of 4")
 *   data.team_composition - { humans: number, ai: number }
 *   data.blockers         - number (optional)
 */

import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import { useTheme } from '@mui/material/styles';

import { StatusBadge } from '../common/StatusBadge';
import { ProgressBar } from '../common/ProgressBar';

export default function ProjectSummaryWidget({ data }) {
  const theme = useTheme();
  const { name, status, progress_percent = 0, sprint, team_composition, blockers } = data || {};

  const widgetBoxSx = {
    backgroundColor: theme.custom.surfaces.overlayLight,
    border: `1px solid ${theme.custom.surfaces.cardBorder}`,
    borderRadius: 2,
    p: 2,
    mt: 1.5,
  };

  return (
    <Box sx={widgetBoxSx}>
      {/* Project name + status */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
        <Typography variant="subtitle2" sx={{ color: 'text.primary', fontWeight: 600, flexGrow: 1 }}>
          {name}
        </Typography>
        <StatusBadge status={status} size="small" />
      </Box>

      {/* Progress */}
      <Box sx={{ mb: 1.5 }}>
        <ProgressBar percent={progress_percent} height={6} />
      </Box>

      {/* Details row */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, flexWrap: 'wrap' }}>
        {/* Team composition */}
        {team_composition && (
          <Box sx={{ display: 'flex', gap: 0.75 }}>
            <Chip
              label={`\u{1F464} ${team_composition.humans || 0} humans`}
              size="small"
              sx={{ backgroundColor: theme.custom.surfaces.overlayMedium, color: 'text.secondary', fontSize: '0.75rem' }}
            />
            <Chip
              label={`\u{1F916} ${team_composition.ai_agents || team_composition.ai || 0} AI`}
              size="small"
              sx={{ backgroundColor: theme.custom.surfaces.overlayMedium, color: 'text.secondary', fontSize: '0.75rem' }}
            />
          </Box>
        )}

        {/* Sprint info */}
        {sprint && (
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            {typeof sprint === 'object' ? `Sprint ${sprint.current} of ${sprint.total}` : sprint}
          </Typography>
        )}

        {/* Blockers */}
        {blockers > 0 && (
          <Chip
            label={`${blockers} blocker${blockers > 1 ? 's' : ''}`}
            size="small"
            sx={{ backgroundColor: theme.custom.surfaces.overlayLight, color: 'error.light', fontSize: '0.75rem', fontWeight: 500 }}
          />
        )}
      </Box>
    </Box>
  );
}
