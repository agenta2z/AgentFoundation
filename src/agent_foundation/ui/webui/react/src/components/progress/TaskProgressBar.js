/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Individual task progress bar component with color-coded completion status.
 */

import React from 'react';
import {
  Box,
  Paper,
  Typography,
  LinearProgress,
  CircularProgress,
  Checkbox,
} from '@mui/material';
import { useTheme, alpha } from '@mui/material/styles';
import {
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
} from '@mui/icons-material';

/**
 * Renders a single task progress bar with status label
 * @param {object} props
 * @param {string} props.id - Task identifier
 * @param {string} props.title - Display title with emoji
 * @param {number} props.progress - Progress percentage (0-100)
 * @param {string} props.status - "pending" | "running" | "complete"
 * @param {object} props.result - Completion result {delta_latency, is_better, result_message}
 * @param {string} props.message - Current progress message
 * @param {boolean} props.selectable - Whether to show selection checkbox
 * @param {boolean} props.selected - Whether this task is selected
 * @param {Function} props.onSelectChange - Callback when selection changes (id, selected) => void
 */
export function TaskProgressBar({
  id,
  title,
  progress = 0,
  status = 'pending',
  result = null,
  message = '',
  selectable = false,
  selected = false,
  onSelectChange,
}) {
  const theme = useTheme();

  const getStatusColor = () => {
    if (status === 'running' || status === 'pending') return 'text.primary';
    if (result?.is_better) return 'success.main';
    return 'error.main';
  };

  const getBarColor = () => {
    if (status === 'complete') {
      return result?.is_better ? 'success' : 'error';
    }
    return 'primary';
  };

  const getBorderColor = () => {
    if (status === 'complete') {
      return result?.is_better ? 'success.main' : 'error.main';
    }
    return 'primary.dark';
  };

  return (
    <Paper
      elevation={2}
      sx={{
        mb: 1.5,
        p: 1.5,
        backgroundColor: theme.custom.surfaces.scrim,
        border: '1px solid',
        borderColor: getBorderColor(),
        borderRadius: 1,
        transition: 'border-color 0.3s ease, box-shadow 0.3s ease',
        ...(status === 'complete' && {
          boxShadow: result?.is_better
            ? `0 0 10px ${alpha(theme.palette.success.main, 0.3)}`
            : `0 0 10px ${alpha(theme.palette.error.main, 0.3)}`,
        }),
      }}
    >
      {/* Title row with progress bar */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 0.5 }}>
        <Typography
          variant="subtitle2"
          sx={{ minWidth: 160, fontWeight: 600, color: 'text.primary' }}
        >
          {title}
        </Typography>
        <LinearProgress
          variant="determinate"
          value={progress}
          color={getBarColor()}
          sx={{
            flex: 1,
            height: 8,
            borderRadius: 4,
            backgroundColor: theme.custom.surfaces.overlayActive,
          }}
        />
        <Typography
          variant="caption"
          sx={{ minWidth: 40, textAlign: 'right', color: 'text.secondary' }}
        >
          {Math.round(progress)}%
        </Typography>
        {status === 'complete' ? (
          result?.is_better ? (
            <CheckCircleIcon sx={{ color: 'success.main', fontSize: 20 }} />
          ) : (
            <CancelIcon sx={{ color: 'error.main', fontSize: 20 }} />
          )
        ) : (
          <CircularProgress size={18} thickness={4} />
        )}
        {status === 'complete' && selectable && (
          <Checkbox
            checked={selected}
            onChange={(e) => onSelectChange?.(id, e.target.checked)}
            size="small"
            sx={{
              p: 0.5,
              color: 'primary.light',
              '&.Mui-checked': { color: 'success.main' },
            }}
          />
        )}
      </Box>

      {/* Status label */}
      <Typography
        variant="body2"
        sx={{
          color: getStatusColor(),
          pl: 1,
          fontSize: '0.85rem',
          transition: 'color 0.3s ease',
          animation: status === 'running' ? 'taskPulse 1.5s infinite' : 'none',
          '@keyframes taskPulse': {
            '0%, 100%': { opacity: 1 },
            '50%': { opacity: 0.6 },
          },
        }}
      >
        {status === 'complete' && result ? result.result_message : message}
      </Typography>
    </Paper>
  );
}

export default TaskProgressBar;
