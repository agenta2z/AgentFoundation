/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * StreamingMessage — Renders partial markdown content with blinking cursor.
 * Shows phase metadata badge during /task execution.
 */

import React from 'react';
import { Box, Paper, Chip } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { MarkdownRenderer } from '../common/MarkdownRenderer';

export function StreamingMessage({ content, metadata, taskPhase, isPlaceholder = false }) {
  const theme = useTheme();
  const agentLabel = metadata?.agent_id ? {
    base: '🔵 Base Agent',
    review: '🟣 Review Agent',
  }[metadata.agent_id] || metadata.agent_id : null;

  const phaseLabel = metadata?.phase ? {
    plan: '📋 Planning',
    implementation: '🔧 Implementation',
    analysis: '🔍 Analysis',
  }[metadata.phase] || metadata.phase : null;

  return (
    <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
      <Paper
        elevation={0}
        sx={{
          p: 2,
          maxWidth: '95%',
          backgroundColor: 'background.paper',
          borderRadius: 2,
          border: '1px solid',
          borderColor: 'primary.main',
        }}
      >
        {(phaseLabel || agentLabel) && (
          <Box sx={{ mb: 1, display: 'flex', gap: 1 }}>
            {phaseLabel && (
              <Chip
                label={phaseLabel}
                size="small"
                variant="outlined"
                sx={{ height: 20, fontSize: '0.65rem', borderColor: 'primary.main' }}
              />
            )}
            {agentLabel && (
              <Chip
                label={agentLabel}
                size="small"
                variant="outlined"
                sx={{ height: 20, fontSize: '0.65rem', borderColor: theme.custom.surfaces.inputBorderHover }}
              />
            )}
          </Box>
        )}

        <Box sx={{
          '& p': { m: 0 },
          '& pre': { overflow: 'auto' },
          ...(isPlaceholder && { opacity: 0.6, fontStyle: 'italic' }),
        }}>
          <MarkdownRenderer content={content || ''} />
          <Box
            component="span"
            sx={{
              display: 'inline-block',
              width: 8,
              height: 16,
              backgroundColor: 'primary.main',
              ml: 0.5,
              verticalAlign: 'text-bottom',
              animation: 'blink 1s step-end infinite',
              '@keyframes blink': {
                '0%, 100%': { opacity: 1 },
                '50%': { opacity: 0 },
              },
            }}
          />
        </Box>
      </Paper>
    </Box>
  );
}

export default StreamingMessage;
