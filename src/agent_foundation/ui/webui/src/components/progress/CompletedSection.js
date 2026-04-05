/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Completed progress section card component.
 */

import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Collapse,
} from '@mui/material';
import {
  Description as FileIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  CheckCircle as CheckCircleIcon,
} from '@mui/icons-material';

/**
 * Renders a completed progress section card
 * @param {object} props
 * @param {object} props.section - Section data
 * @param {boolean} props.isCollapsed - Whether section is collapsed
 * @param {Function} props.onToggle - Callback when header is clicked
 * @param {Function} props.onContextMenu - Callback for right-click context menu
 * @param {Function} props.onOpenFile - Callback when prompt file is clicked
 */
export function CompletedSection({
  section,
  isCollapsed = true,
  onToggle,
  onContextMenu,
  onOpenFile,
}) {
  const revealedCount = section.revealed_count || section.messages.length;
  const visibleMessages = section.messages.slice(0, revealedCount);

  return (
    <Paper
      elevation={0}
      onContextMenu={(e) => onContextMenu?.(e, section)}
      sx={{
        flex: '1 1 300px',
        maxWidth: '48%',
        minWidth: '280px',
        backgroundColor: 'rgba(74, 144, 217, 0.1)',
        borderRadius: 2,
        border: '1px solid',
        borderColor: 'success.main',
        overflow: 'hidden',
        opacity: 0.85,
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
          backgroundColor: 'rgba(0, 0, 0, 0.2)',
          borderBottom: isCollapsed ? 'none' : '1px solid rgba(255,255,255,0.1)',
          '&:hover': { backgroundColor: 'rgba(0, 0, 0, 0.3)' },
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
                  backgroundColor: 'rgba(255,255,255,0.1)',
                },
              }}
            >
              View Prompt
            </Button>
          )}
          <CheckCircleIcon sx={{ color: 'success.main', fontSize: 18 }} />
        </Box>
      </Box>

      {/* Content */}
      <Collapse in={!isCollapsed}>
        <Box sx={{ p: 2, maxHeight: 300, overflow: 'auto' }}>
          {visibleMessages.map((msg, msgIndex) => (
            <Typography
              key={msgIndex}
              variant="body2"
              sx={{
                color: 'text.secondary',
                mb: 0.5,
                opacity: 0.7,
                pl: 1,
                borderLeft: '2px solid transparent',
              }}
            >
              ✓ {msg.content}
            </Typography>
          ))}
        </Box>
      </Collapse>
    </Paper>
  );
}

export default CompletedSection;
