/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * RunQueueList — Generic left-panel clickable run queue list with status
 * badges. Per-row affordances for nested implhyp wrapper rows (Resume +
 * View-error).
 *
 * Ported from RankEvolve (components/views/RunQueueList.js). The one change:
 * `resumeImplementTask` is now an INJECTED prop (the apiClient action) rather
 * than read from `useSession()`.
 */

import React, { useState } from 'react';
import {
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  List,
  ListItemButton,
  Tooltip,
  Typography,
} from '@mui/material';
import ReplayIcon from '@mui/icons-material/Replay';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

const STATUS_CONFIG = {
  queued: { label: 'Queued', color: 'default', icon: '⏳' },
  running: { label: 'Running', color: 'info', icon: '▶' },
  completed: { label: 'Done', color: 'success', icon: '✅' },
  error: { label: 'Error', color: 'error', icon: '❌' },
};

export default function RunQueueList({
  runQueue,
  selectedIndex,
  currentRunIndex,
  onSelect,
  renderLabel,
  resumeImplementTask,
}) {
  const [errorDialog, setErrorDialog] = useState(null);

  return (
    <Box sx={{
      width: 220,
      minWidth: 220,
      borderRight: '1px solid',
      borderColor: 'divider',
      overflow: 'auto',
      backgroundColor: 'rgba(0, 0, 0, 0.05)',
    }}>
      <List dense disablePadding>
        {(runQueue || []).map((entry, i) => {
          const isSelected = i === selectedIndex;
          const isRunning = i === currentRunIndex;
          const status = STATUS_CONFIG[entry.status] || STATUS_CONFIG.queued;
          const meta = entry.metadata || {};
          const isImplhypWrapper =
            meta.implhyp_kind === 'wrapper'
            && entry.status === 'error'
            && typeof entry.subTaskId === 'string'
            && entry.subTaskId.startsWith('implhyp-')
            && typeof resumeImplementTask === 'function';
          const hasErrorMessage =
            entry.status === 'error' && typeof meta.error_message === 'string' && meta.error_message;

          return (
            <ListItemButton
              key={i}
              selected={isSelected}
              onClick={() => onSelect(i)}
              sx={{
                py: 1,
                px: 1.5,
                borderLeft: isSelected ? '3px solid' : '3px solid transparent',
                borderColor: isSelected ? 'primary.main' : 'transparent',
                opacity: entry.status === 'queued' ? 0.6 : 1,
              }}
            >
              <Box sx={{ flex: 1, minWidth: 0 }}>
                {renderLabel ? renderLabel(entry) : (
                  <Typography variant="body2" sx={{
                    fontSize: '0.8rem',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}>
                    {entry.label}
                  </Typography>
                )}
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, ml: 1, flexShrink: 0 }}>
                <Chip
                  label={status.label}
                  size="small"
                  color={status.color}
                  variant="outlined"
                  sx={{ height: 18, fontSize: '0.6rem', '& .MuiChip-label': { px: 0.5 } }}
                />
                {isRunning && (
                  <Typography sx={{ fontSize: '0.7rem', color: 'info.main' }}>←</Typography>
                )}
                {hasErrorMessage && (
                  <Tooltip title="View error message">
                    <IconButton
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        setErrorDialog({
                          title: entry.label || entry.subTaskId || 'Run error',
                          body: meta.error_message,
                        });
                      }}
                      sx={{ p: 0.25 }}
                    >
                      <ErrorOutlineIcon fontSize="small" color="error" />
                    </IconButton>
                  </Tooltip>
                )}
                {isImplhypWrapper && (
                  <Tooltip title="Re-run /implement-hypothesis with --reuse-task — preserves workspace; only failed batches re-execute.">
                    <IconButton
                      size="small"
                      color="warning"
                      onClick={(e) => {
                        e.stopPropagation();
                        resumeImplementTask({
                          id: entry.subTaskId,
                          metadata: meta,
                        });
                      }}
                      sx={{ p: 0.25 }}
                    >
                      <ReplayIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                )}
              </Box>
            </ListItemButton>
          );
        })}
      </List>
      <Dialog
        open={errorDialog !== null}
        onClose={() => setErrorDialog(null)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Error: {errorDialog?.title}</DialogTitle>
        <DialogContent>
          <Box
            component="pre"
            sx={{
              fontFamily: 'monospace',
              fontSize: '0.75rem',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              backgroundColor: 'rgba(0,0,0,0.2)',
              p: 1.5,
              borderRadius: 1,
              maxHeight: '60vh',
              overflow: 'auto',
            }}
          >
            {errorDialog?.body}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setErrorDialog(null)} sx={{ textTransform: 'none' }}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
