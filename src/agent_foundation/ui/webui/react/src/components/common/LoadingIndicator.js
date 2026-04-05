/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Loading indicator component.
 */

import React from 'react';
import { Box, Paper, Typography, CircularProgress } from '@mui/material';

/**
 * Renders a loading/thinking indicator
 * @param {object} props
 * @param {string} props.text - Loading text to display
 */
export function LoadingIndicator({ text = 'Thinking...' }) {
  return (
    <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
      <Paper
        elevation={0}
        sx={{
          p: 2,
          backgroundColor: 'background.paper',
          borderRadius: 2,
          border: '1px solid',
          borderColor: 'divider',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <CircularProgress size={16} thickness={4} />
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            {text}
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
}

export default LoadingIndicator;
