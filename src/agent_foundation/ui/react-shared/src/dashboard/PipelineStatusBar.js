/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * PipelineStatusBar — Horizontal workflow-stage indicator for a dashboard.
 * Shows stages like: Research ✅ → Selection ✅ → Implement ⏳ → Experiment ○ → Monitor ○
 * Fully generic (ported near-verbatim from RankEvolve's agent/PipelineStatusBar).
 */

import React from 'react';
import { Box, Typography } from '@mui/material';
import {
  CheckCircle as CheckIcon,
  RadioButtonChecked as ActiveIcon,
  RadioButtonUnchecked as PendingIcon,
} from '@mui/icons-material';

const STATUS_ICONS = {
  completed: <CheckIcon sx={{ fontSize: 16, color: 'success.main' }} />,
  active: <ActiveIcon sx={{ fontSize: 16, color: 'info.main' }} />,
  pending: <PendingIcon sx={{ fontSize: 16, color: 'text.disabled' }} />,
};

export function PipelineStatusBar({ stages, statuses }) {
  if (!stages || stages.length === 0) return null;

  return (
    <Box sx={{
      display: 'flex',
      alignItems: 'center',
      gap: 0.5,
      px: 2,
      py: 0.75,
      borderBottom: '1px solid',
      borderColor: 'divider',
      backgroundColor: 'rgba(0, 0, 0, 0.05)',
      flexWrap: 'wrap',
    }}>
      {stages.map((stage, i) => {
        const status = (statuses && statuses[i]) || 'pending';
        return (
          <React.Fragment key={stage}>
            {i > 0 && (
              <Typography sx={{ fontSize: '0.7rem', color: 'text.disabled', mx: 0.25 }}>
                →
              </Typography>
            )}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.25 }}>
              {STATUS_ICONS[status]}
              <Typography sx={{
                fontSize: '0.72rem',
                fontWeight: status === 'active' ? 600 : 400,
                color: status === 'pending' ? 'text.disabled'
                  : status === 'active' ? 'info.main'
                  : 'text.secondary',
              }}>
                {stage}
              </Typography>
            </Box>
          </React.Fragment>
        );
      })}
    </Box>
  );
}

export default PipelineStatusBar;
