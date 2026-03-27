/**
 * ProgressBar — linear progress with optional percentage label.
 *
 * Automatically colors based on percentage (green > yellow > red)
 * unless an explicit color is provided.
 *
 * Props:
 *   percent    - number 0-100
 *   color      - optional override color string
 *   showLabel  - boolean (default: true)
 *   height     - number in px (default: 8)
 */

import React from 'react';
import Box from '@mui/material/Box';
import LinearProgress from '@mui/material/LinearProgress';
import Typography from '@mui/material/Typography';

function getAutoColor(percent) {
  if (percent >= 75) return '#4caf50';
  if (percent >= 40) return '#ff9800';
  return '#f44336';
}

export function ProgressBar({ percent = 0, color, showLabel = true, height = 8 }) {
  const barColor = color || getAutoColor(percent);
  const clampedPercent = Math.min(100, Math.max(0, percent));

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
      <Box sx={{ flexGrow: 1 }}>
        <LinearProgress
          variant="determinate"
          value={clampedPercent}
          sx={{
            height,
            borderRadius: height / 2,
            backgroundColor: 'rgba(255, 255, 255, 0.08)',
            '& .MuiLinearProgress-bar': {
              borderRadius: height / 2,
              backgroundColor: barColor,
            },
          }}
        />
      </Box>
      {showLabel && (
        <Typography variant="body2" sx={{ color: 'text.secondary', minWidth: 40, textAlign: 'right' }}>
          {Math.round(clampedPercent)}%
        </Typography>
      )}
    </Box>
  );
}

export default ProgressBar;
