/**
 * SessionContextBar — Visual progress bar showing session context usage.
 * 
 * Parses "Session context: ▮▮▮▮▮▮▮▮▮▮ 33.8K/1M" from acli output
 * and renders it as a clean MUI LinearProgress bar.
 */

import React from 'react';
import { Box, Typography, LinearProgress, Tooltip } from '@mui/material';
import { Memory as ContextIcon } from '@mui/icons-material';

export function SessionContextBar({ usedLabel, totalLabel, percentage }) {
  if (!usedLabel || !totalLabel) return null;

  const getColor = (pct) => {
    if (pct > 80) return 'error';
    if (pct > 60) return 'warning';
    return 'primary';
  };

  return (
    <Tooltip title={`Context window: ${usedLabel} of ${totalLabel} used`} placement="top">
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          px: 1.5,
          py: 0.5,
          borderRadius: 1,
          backgroundColor: 'rgba(255, 255, 255, 0.03)',
        }}
      >
        <ContextIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
        <Typography
          variant="caption"
          sx={{ color: 'text.secondary', whiteSpace: 'nowrap', fontSize: '0.7rem' }}
        >
          {usedLabel}/{totalLabel}
        </Typography>
        <LinearProgress
          variant="determinate"
          value={Math.min(percentage, 100)}
          color={getColor(percentage)}
          sx={{
            flexGrow: 1,
            height: 4,
            borderRadius: 2,
            minWidth: 60,
            backgroundColor: 'rgba(255, 255, 255, 0.08)',
          }}
        />
        <Typography
          variant="caption"
          sx={{ color: 'text.disabled', whiteSpace: 'nowrap', fontSize: '0.65rem' }}
        >
          {percentage.toFixed(0)}%
        </Typography>
      </Box>
    </Tooltip>
  );
}

export default SessionContextBar;
