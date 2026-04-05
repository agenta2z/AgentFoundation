/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pre-animation message component (shown during pre_messages phase).
 */

import React from 'react';
import { Box, Paper, CircularProgress } from '@mui/material';
import { MarkdownRenderer } from '../common/MarkdownRenderer';

/**
 * Renders a pre-animation message with loading indicator
 * @param {object} props
 * @param {object} props.message - Message object with content
 */
export function PreMessage({ message }) {
  if (!message) return null;

  return (
    <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
      <Paper
        elevation={0}
        sx={{
          p: 2,
          maxWidth: '80%',
          backgroundColor: 'background.paper',
          borderRadius: 2,
          border: '1px solid',
          borderColor: 'divider',
          animation: 'fadeIn 0.3s ease-in-out',
          '@keyframes fadeIn': {
            '0%': { opacity: 0, transform: 'translateY(10px)' },
            '100%': { opacity: 1, transform: 'translateY(0)' },
          },
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
          <CircularProgress size={16} thickness={4} sx={{ mt: 0.5, flexShrink: 0 }} />
          <Box className="markdown-content" sx={{ '& p': { margin: 0 } }}>
            <MarkdownRenderer content={message.content} />
          </Box>
        </Box>
      </Paper>
    </Box>
  );
}

export default PreMessage;
