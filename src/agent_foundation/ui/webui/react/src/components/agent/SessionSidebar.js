/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * SessionSidebar — Left sidebar with session/task tree navigation.
 * ~250px fixed panel. Shows sessions with nested tasks, status chips,
 * and connection status.
 */

import React from 'react';
import {
  Box,
  Typography,
  Button,
  Chip,
  Divider,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import {
  Add as AddIcon,
} from '@mui/icons-material';
import { useSession } from '../../contexts/SessionContext';

const TASK_STATUS_CHIPS = {
  starting: { label: 'Starting', color: 'warning', icon: '...' },
  running: { label: 'Running', color: 'info', icon: '...' },
  completed: { label: 'Done', color: 'success', icon: '' },
  error: { label: 'Error', color: 'error', icon: '' },
};

export function SessionSidebar() {
  const theme = useTheme();
  const {
    sessionList,
    tasks,
    activeTabId,
    activeTabType,
    serverStatus,
    createSession,
    switchTab,
  } = useSession();

  const serverStatusDisplay = {
    connected: { icon: '', label: 'Connected' },
    server_down: { icon: '', label: 'Server down' },
    reconnecting: { icon: '', label: 'Reconnecting...' },
    syncing: { icon: '', label: 'Syncing...' },
  }[serverStatus] || { icon: '', label: serverStatus };

  const isDisabled = serverStatus !== 'connected';

  return (
    <Box
      sx={{
        width: 250,
        minWidth: 250,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        borderRight: '1px solid',
        borderColor: 'divider',
        backgroundColor: theme.custom.surfaces.sidebarBg,
        overflow: 'hidden',
        opacity: isDisabled ? 0.6 : 1,
        transition: 'opacity 0.2s',
      }}
    >
      {/* Header */}
      <Box sx={{ p: 1.5, borderBottom: '1px solid', borderColor: 'divider' }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'text.secondary', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: 1 }}>
          Sessions
        </Typography>
      </Box>

      {/* Session list */}
      <Box sx={{ flex: 1, overflow: 'auto', py: 0.5 }}>
        {sessionList.map((session) => {
          const isActiveSession = activeTabType === 'session' && activeTabId === session.id;
          const isCreating = session.status === 'creating';
          const sessionTasks = session.taskIds
            .map(tid => tasks[tid])
            .filter(Boolean);

          return (
            <Box key={session.id}>
              {/* Session entry */}
              <Box
                onClick={() => !isCreating && switchTab(session.id, 'session')}
                sx={{
                  px: 1.5,
                  py: 1,
                  cursor: isCreating ? 'default' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                  backgroundColor: isActiveSession ? theme.custom.surfaces.activeHighlight : 'transparent',
                  borderLeft: isActiveSession ? '3px solid' : '3px solid transparent',
                  borderColor: isActiveSession ? 'primary.main' : 'transparent',
                  opacity: isCreating ? 0.5 : 1,
                  '&:hover': {
                    backgroundColor: isCreating ? undefined
                      : isActiveSession ? theme.custom.surfaces.highlight : theme.custom.surfaces.inputBg,
                  },
                }}
              >
                <Typography variant="body2" sx={{
                  fontWeight: isActiveSession ? 600 : 400,
                  color: isCreating ? 'text.disabled' : isActiveSession ? 'primary.light' : 'text.primary',
                  fontSize: '0.85rem',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  flex: 1,
                  fontStyle: isCreating ? 'italic' : 'normal',
                }}>
                  {isCreating ? 'Creating New Session...' : session.label}
                </Typography>
              </Box>

              {/* Task children */}
              {sessionTasks.map((task) => {
                const isActiveTask = activeTabType === 'task' && activeTabId === task.id;
                const statusInfo = TASK_STATUS_CHIPS[task.status] || TASK_STATUS_CHIPS.running;

                return (
                  <Box
                    key={task.id}
                    onClick={() => switchTab(task.id, 'task')}
                    sx={{
                      pl: 3.5,
                      pr: 1.5,
                      py: 0.75,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 0.5,
                      backgroundColor: isActiveTask ? theme.custom.surfaces.highlight : 'transparent',
                      borderLeft: isActiveTask ? '3px solid' : '3px solid transparent',
                      borderColor: isActiveTask ? 'primary.light' : 'transparent',
                      '&:hover': {
                        backgroundColor: isActiveTask ? theme.custom.surfaces.activeHighlight : theme.custom.surfaces.cardBg,
                      },
                    }}
                  >
                    <Typography variant="body2" sx={{
                      fontSize: '0.8rem',
                      color: isActiveTask ? 'primary.light' : 'text.secondary',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      flex: 1,
                    }}>
                      {task.label}
                    </Typography>
                    <Chip
                      label={`${statusInfo.icon} ${statusInfo.label}`}
                      size="small"
                      color={statusInfo.color}
                      variant="outlined"
                      sx={{
                        height: 18,
                        fontSize: '0.6rem',
                        '& .MuiChip-label': { px: 0.5 },
                      }}
                    />
                  </Box>
                );
              })}
            </Box>
          );
        })}
      </Box>

      <Divider />

      {/* New session button */}
      <Box sx={{ p: 1 }}>
        <Button
          fullWidth
          size="small"
          startIcon={<AddIcon />}
          onClick={() => createSession()}
          disabled={isDisabled}
          sx={{
            justifyContent: 'flex-start',
            textTransform: 'none',
            fontSize: '0.8rem',
            color: 'text.secondary',
            '&:hover': { backgroundColor: theme.custom.surfaces.overlayMedium },
          }}
        >
          New Session
        </Button>
      </Box>

      {/* Connection status */}
      <Box sx={{
        p: 1,
        borderTop: '1px solid',
        borderColor: 'divider',
        display: 'flex',
        alignItems: 'center',
        gap: 0.5,
      }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>
          {serverStatusDisplay.icon} {serverStatusDisplay.label}
        </Typography>
      </Box>
    </Box>
  );
}

export default SessionSidebar;
