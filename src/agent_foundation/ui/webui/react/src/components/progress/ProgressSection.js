/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Single progress section card component.
 */

import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Collapse,
  CircularProgress,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import {
  Description as FileIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  CheckCircle as CheckCircleIcon,
} from '@mui/icons-material';

/**
 * Renders a single progress section card
 * @param {object} props
 * @param {object} props.section - Section data with title, messages, slot, prompt_file
 * @param {number} props.revealedCount - Number of messages revealed so far
 * @param {boolean} props.isComplete - Whether all messages are revealed
 * @param {boolean} props.isCollapsed - Whether section content is collapsed
 * @param {boolean} props.isAnimating - Whether animation is in progress
 * @param {boolean} props.justAppeared - Whether section just appeared (for animation)
 * @param {Function} props.onToggle - Callback when header is clicked
 * @param {Function} props.onContextMenu - Callback for right-click context menu
 * @param {Function} props.onOpenFile - Callback when prompt file is clicked
 */
export function ProgressSection({
  section,
  revealedCount = 0,
  isComplete = false,
  isCollapsed = false,
  isAnimating = false,
  justAppeared = false,
  onToggle,
  onContextMenu,
  onOpenFile,
}) {
  const theme = useTheme();
  const visibleMessages = section.messages.slice(0, revealedCount);

  return (
    <Paper
      elevation={0}
      className={`progress-section ${justAppeared ? 'section-appear' : ''}`}
      onContextMenu={(e) => onContextMenu?.(e, section)}
      sx={{
        flex: '1 1 300px',
        maxWidth: '48%',
        minWidth: '280px',
        backgroundColor: theme.custom.surfaces.highlight,
        borderRadius: 2,
        border: '1px solid',
        borderColor: isComplete ? 'success.main' : 'primary.dark',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <Box
        onClick={() => onToggle?.(section.slot)}
        sx={{
          p: 1.5,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          cursor: 'pointer',
          backgroundColor: theme.custom.surfaces.sidebarBg,
          borderBottom: isCollapsed ? 'none' : `1px solid ${theme.custom.surfaces.overlayActive}`,
          '&:hover': { backgroundColor: theme.custom.surfaces.scrim },
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {isCollapsed ? (
            <ExpandMoreIcon sx={{ color: 'primary.light', fontSize: 20 }} />
          ) : (
            <ExpandLessIcon sx={{ color: 'primary.light', fontSize: 20 }} />
          )}
          <Typography
            variant="subtitle2"
            sx={{ color: 'primary.light', fontWeight: 600 }}
          >
            {section.title}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {section.prompt_file && (
            <Button
              size="small"
              startIcon={<FileIcon sx={{ fontSize: 14 }} />}
              onClick={(e) => {
                e.stopPropagation();
                onOpenFile?.(section.prompt_file);
              }}
              sx={{
                fontSize: '0.7rem',
                textTransform: 'none',
                color: 'text.secondary',
                minWidth: 'auto',
                py: 0.25,
                px: 0.75,
                '&:hover': {
                  color: 'primary.light',
                  backgroundColor: theme.custom.surfaces.overlayActive,
                },
              }}
            >
              View Prompt
            </Button>
          )}
          {isComplete ? (
            <CheckCircleIcon sx={{ color: 'success.main', fontSize: 18 }} />
          ) : (
            <CircularProgress size={14} thickness={4} />
          )}
        </Box>
      </Box>

      {/* Content */}
      <Collapse in={!isCollapsed}>
        <Box sx={{ p: 2, maxHeight: 300, overflow: 'auto' }}>
          {visibleMessages.map((msg, msgIndex) => {
            const isLatest = msgIndex === visibleMessages.length - 1 && !isComplete;
            return (
              <Typography
                key={msgIndex}
                variant="body2"
                sx={{
                  color: isLatest ? 'primary.light' : 'text.secondary',
                  mb: 0.5,
                  opacity: isLatest ? 1 : 0.7,
                  transition: 'all 0.3s ease-in-out',
                  pl: 1,
                  borderLeft: isLatest ? '2px solid' : '2px solid transparent',
                  borderColor: isLatest ? 'primary.main' : 'transparent',
                  animation: isLatest ? 'pulse 1.5s infinite' : 'none',
                  '@keyframes pulse': {
                    '0%, 100%': { opacity: 1 },
                    '50%': { opacity: 0.6 },
                  },
                }}
              >
                {isComplete || msgIndex < visibleMessages.length - 1 ? '✓ ' : '⏳ '}
                {msg.content}
              </Typography>
            );
          })}
          {isAnimating && !isComplete && revealedCount < section.messages.length && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1, pl: 1 }}>
              <CircularProgress size={12} thickness={4} />
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                Processing...
              </Typography>
            </Box>
          )}
        </Box>
      </Collapse>
    </Paper>
  );
}

export default ProgressSection;
