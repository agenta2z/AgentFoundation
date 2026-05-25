/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Container panel for task progress bars with summary.
 */

import React from 'react';
import {
  Box,
  Paper,
  Typography,
  CircularProgress,
  Button,
  Checkbox,
  FormControlLabel,
} from '@mui/material';
import { useTheme, alpha } from '@mui/material/styles';
import { TaskProgressBar } from './TaskProgressBar';

/**
 * Renders the task progress panel with all progress bars and summary
 * @param {object} props
 * @param {Array} props.tasks - Array of task data objects
 * @param {boolean} props.isAnimating - Whether animation is in progress
 * @param {string} props.title - Panel title
 * @param {boolean} props.selectable - Whether to show selection checkboxes
 * @param {Array} props.selectedTasks - Array of selected task IDs
 * @param {Function} props.onTaskSelectionChange - Callback when task selection changes
 * @param {Function} props.onProceedWithSelected - Callback when proceed button is clicked
 */
export function TaskProgressPanel({
  tasks = [],
  isAnimating = false,
  title = '🧪 Running Tasks',
  selectable = false,
  selectedTasks = [],
  onTaskSelectionChange,
  onProceedWithSelected,
}) {
  const theme = useTheme();
  const allComplete = tasks.length > 0 && tasks.every(t => t.status === 'complete');
  const improvedCount = tasks.filter(t => t.result?.is_better).length;
  const regressedCount = tasks.filter(t => t.status === 'complete' && !t.result?.is_better).length;
  const bestImprovement = tasks
    .filter(t => t.result?.is_better)
    .sort((a, b) => (a.result?.delta_latency || 0) - (b.result?.delta_latency || 0))[0];

  const improvedTaskIds = tasks.filter(t => t.result?.is_better).map(t => t.id);
  const allImprovedSelected = improvedTaskIds.length > 0 &&
    improvedTaskIds.every(id => selectedTasks.includes(id));

  const handleSelectAllImproved = () => {
    if (allImprovedSelected) {
      improvedTaskIds.forEach(id => onTaskSelectionChange?.(id, false));
    } else {
      improvedTaskIds.forEach(id => onTaskSelectionChange?.(id, true));
    }
  };

  return (
    <Paper
      elevation={3}
      sx={{
        p: 2,
        width: '100%',
        maxWidth: 700,
        backgroundColor: alpha(theme.palette.background.default, 0.95),
        borderRadius: 2,
        border: `1px solid ${theme.custom.surfaces.overlayActive}`,
      }}
    >
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary' }}>
          {title}
        </Typography>
        {isAnimating && !allComplete && (
          <CircularProgress size={18} thickness={4} sx={{ ml: 1 }} />
        )}
      </Box>

      {/* Progress bars */}
      <Box sx={{ mb: allComplete ? 2 : 0 }}>
        {tasks.map((task) => (
          <TaskProgressBar
            key={task.id}
            {...task}
            selectable={selectable && allComplete}
            selected={selectedTasks.includes(task.id)}
            onSelectChange={onTaskSelectionChange}
          />
        ))}
      </Box>

      {/* Summary section (appears after all complete) */}
      {allComplete && (
        <Paper
          elevation={1}
          sx={{
            p: 2,
            backgroundColor: improvedCount > regressedCount
              ? alpha(theme.palette.success.main, 0.1)
              : alpha(theme.palette.error.main, 0.1),
            border: '1px solid',
            borderColor: improvedCount > regressedCount ? 'success.main' : 'error.main',
            borderRadius: 1,
          }}
        >
          <Typography
            variant="subtitle1"
            sx={{ fontWeight: 600, mb: 1, color: 'text.primary' }}
          >
            📊 SUMMARY
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.primary', mb: 0.5 }}>
            Tasks Improved:{' '}
            <Box component="span" sx={{ color: 'success.main', fontWeight: 600 }}>
              {improvedCount}/{tasks.length} ✓
            </Box>
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.primary', mb: 0.5 }}>
            Tasks Regressed:{' '}
            <Box component="span" sx={{ color: 'error.main', fontWeight: 600 }}>
              {regressedCount}/{tasks.length}
            </Box>
          </Typography>
          {bestImprovement && (
            <Typography variant="body2" sx={{ color: 'text.primary' }}>
              Best Improvement:{' '}
              <Box component="span" sx={{ color: 'success.main', fontWeight: 600 }}>
                {bestImprovement.title} ({bestImprovement.result?.delta_latency}% latency)
              </Box>
            </Typography>
          )}
        </Paper>
      )}

      {/* Selection CTA section (appears after completion when selectable) */}
      {allComplete && selectable && (
        <Box sx={{ mt: 2, pt: 2, borderTop: `1px solid ${theme.custom.surfaces.overlayActive}` }}>
          <Typography variant="body2" sx={{ mb: 1.5, color: 'text.secondary' }}>
            Select proposals for full experiment submission (MAST jobs):
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap' }}>
            <FormControlLabel
              control={
                <Checkbox
                  checked={allImprovedSelected}
                  onChange={handleSelectAllImproved}
                  size="small"
                  sx={{ color: 'primary.light', '&.Mui-checked': { color: 'success.main' } }}
                />
              }
              label={
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  Select all improved ({improvedCount})
                </Typography>
              }
            />
            <Button
              variant="contained"
              color="primary"
              size="small"
              disabled={selectedTasks.length === 0}
              onClick={() => onProceedWithSelected?.(selectedTasks)}
              sx={{ textTransform: 'none' }}
            >
              Submit {selectedTasks.length} Selected for Full Experiments
            </Button>
          </Box>
        </Box>
      )}
    </Paper>
  );
}

export default TaskProgressPanel;
