/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * AgentStatusBar — Shows connection status, model, target path, task phase,
 * and queue status dropdown.
 */

import React, { useState, useEffect, useRef } from 'react';
import { Box, Button, Typography, Chip, Popover, Divider, IconButton, Switch, FormControlLabel, Tabs, Tab, TextField } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import SettingsIcon from '@mui/icons-material/Settings';

function formatUptime(seconds) {
  if (seconds == null || seconds < 0) return 'N/A';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

export function AgentStatusBar({ connectionStatus, model, targetPath, taskPhase, isStreaming, queueStatus, serverSessionId, serverId, sessionSettings, onSettingsChange, globalSettings, onGlobalSettingsChange, onResetWelcomeMessage }) {
  const theme = useTheme();
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef(null);
  const [anchorEl, setAnchorEl] = useState(null);
  const [settingsAnchor, setSettingsAnchor] = useState(null);
  const [settingsTab, setSettingsTab] = useState(0);

  useEffect(() => {
    if (isStreaming) {
      setElapsed(0);
      timerRef.current = setInterval(() => {
        setElapsed(prev => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isStreaming]);

  const connectionIcon = connectionStatus === 'connected' ? '🟢'
    : connectionStatus === 'connecting' ? '🟡'
    : '🔴';
  const connectionLabel = connectionStatus === 'connected' ? 'Connected'
    : connectionStatus === 'connecting' ? 'Connecting...'
    : 'Disconnected';
  const isConnecting = connectionStatus !== 'connected';

  const formatElapsed = (seconds) => {
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  const phaseLabel = taskPhase ? {
    plan: '📋 Planning',
    implementation: '🔧 Implementation',
    analysis: '🔍 Analysis',
    complete: '✅ Complete',
    cancelled: '⚪ Cancelled',
    error: '❌ Error',
  }[taskPhase.phase] || taskPhase.phase : null;

  const handleQueueClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleQueueClose = () => {
    setAnchorEl(null);
  };

  const queueOpen = Boolean(anchorEl);

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        px: 2,
        py: 0.75,
        backgroundColor: theme.custom.surfaces.scrim,
        borderBottom: '1px solid',
        borderColor: 'divider',
        flexWrap: 'wrap',
      }}
    >
      <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
        {connectionIcon} {connectionLabel}
        {serverId && (
          <span style={{ fontFamily: 'monospace', opacity: 0.6, fontSize: '0.65rem' }}>
            [{serverId}]
          </span>
        )}
      </Typography>

      <Chip
        label={isConnecting ? '🤖 ...' : `🤖 ${model || '(no model)'}`}
        size="small"
        variant="outlined"
        sx={{ height: 22, fontSize: '0.7rem', borderColor: theme.custom.surfaces.inputBorder, opacity: isConnecting ? 0.3 : model ? 1 : 0.5 }}
      />

      <Typography variant="caption" sx={{ color: isConnecting ? 'text.disabled' : model ? 'text.secondary' : 'text.disabled', fontSize: '0.7rem', opacity: isConnecting ? 0.3 : 1 }}>
        📁 {isConnecting ? '...' : targetPath || '(no target path)'}
      </Typography>

      {phaseLabel && (
        <Chip
          label={phaseLabel}
          size="small"
          color={taskPhase?.state === 'running' ? 'primary' : 'default'}
          sx={{ height: 22, fontSize: '0.7rem' }}
        />
      )}

      {isConnecting ? (
        <Chip
          label="📊 ..."
          size="small"
          variant="outlined"
          sx={{ height: 22, fontSize: '0.7rem', borderColor: theme.custom.surfaces.inputBorder, opacity: 0.3 }}
        />
      ) : queueStatus ? (
        <>
          <Chip
            label={`📊 Queues (${queueStatus.global?.total_sessions ?? '?'} sessions) ▾`}
            size="small"
            variant="outlined"
            onClick={handleQueueClick}
            sx={{
              height: 22,
              fontSize: '0.7rem',
              borderColor: theme.custom.surfaces.inputBorder,
              cursor: 'pointer',
              '&:hover': { borderColor: theme.custom.surfaces.mutedText },
            }}
          />
          <Popover
            open={queueOpen}
            anchorEl={anchorEl}
            onClose={handleQueueClose}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
            transformOrigin={{ vertical: 'top', horizontal: 'left' }}
          >
            <Box sx={{ p: 2, minWidth: 320, backgroundColor: 'background.paper' }}>
              <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                📊 Queue Status
              </Typography>
              <Divider sx={{ mb: 1 }} />

              {queueStatus.global && (
                <>
                  <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                    Global
                  </Typography>
                  <Box sx={{ pl: 1, mb: 1 }}>
                    <Typography variant="caption" component="div">
                      Total sessions: {queueStatus.global.total_sessions ?? 'N/A'}
                    </Typography>
                    <Typography variant="caption" component="div">
                      Control queue depth: {queueStatus.global.server_control_depth ?? 'N/A'}
                    </Typography>
                  </Box>
                  <Divider sx={{ mb: 1 }} />
                </>
              )}

              {queueStatus.session && (
                <>
                  <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                    Current Session: {queueStatus.session.session_id}
                  </Typography>
                  <Box sx={{ pl: 1 }}>
                    <Typography variant="caption" component="div" sx={{ fontFamily: 'monospace', fontSize: '0.7rem' }}>
                      Input: {queueStatus.session.input_queue} ({queueStatus.session.input_depth ?? '?'})
                    </Typography>
                    <Typography variant="caption" component="div" sx={{ fontFamily: 'monospace', fontSize: '0.7rem' }}>
                      Output: {queueStatus.session.response_queue} ({queueStatus.session.response_depth ?? '?'})
                    </Typography>
                    {queueStatus.session.active_task_id && (
                      <Typography variant="caption" component="div">
                        Active task: {queueStatus.session.active_task_id}
                      </Typography>
                    )}
                  </Box>
                </>
              )}
            </Box>
          </Popover>
        </>
      ) : (
        <Chip
          label="📊 Queues (loading...)"
          size="small"
          variant="outlined"
          sx={{ height: 22, fontSize: '0.7rem', borderColor: theme.custom.surfaces.inputBorder, opacity: 0.5 }}
        />
      )}

      {/* Session Settings */}
      <IconButton
        size="small"
        onClick={(e) => setSettingsAnchor(e.currentTarget)}
        sx={{
          ml: 'auto',
          color: 'text.secondary',
          '&:hover': { color: 'primary.light' },
        }}
      >
        <SettingsIcon sx={{ fontSize: 18 }} />
      </IconButton>
      <Popover
        open={Boolean(settingsAnchor)}
        anchorEl={settingsAnchor}
        onClose={() => setSettingsAnchor(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <Box sx={{ minWidth: 340, backgroundColor: 'background.paper' }}>
          <Tabs
            value={settingsTab}
            onChange={(_, v) => setSettingsTab(v)}
            variant="fullWidth"
            sx={{ borderBottom: 1, borderColor: 'divider', minHeight: 36 }}
          >
            <Tab label="Session" sx={{ minHeight: 36, fontSize: '0.8rem', textTransform: 'none' }} />
            <Tab label="Global" sx={{ minHeight: 36, fontSize: '0.8rem', textTransform: 'none' }} />
          </Tabs>

          {/* Session Settings Tab */}
          {settingsTab === 0 && (
            <Box sx={{ p: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    size="small"
                    checked={sessionSettings?.showStreamingSections !== false}
                    onChange={(e) => onSettingsChange?.({
                      ...sessionSettings,
                      showStreamingSections: e.target.checked,
                    })}
                  />
                }
                label={
                  <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
                    Show agent streaming sections
                  </Typography>
                }
              />
              <Typography variant="caption" component="div" sx={{ color: 'text.secondary', ml: 4, mt: -0.5, mb: 1 }}>
                Keep streaming output as foldable boxes after completion
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    size="small"
                    checked={sessionSettings?.streamSectionsCollapsed !== false}
                    onChange={(e) => onSettingsChange?.({
                      ...sessionSettings,
                      streamSectionsCollapsed: e.target.checked,
                    })}
                  />
                }
                label={
                  <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
                    Auto-collapse completed sections
                  </Typography>
                }
              />
              <Typography variant="caption" component="div" sx={{ color: 'text.secondary', ml: 4, mt: -0.5, mb: 1 }}>
                Collapse streaming sections after they finish
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    size="small"
                    checked={sessionSettings?.separateFollowUpTurns !== false}
                    onChange={(e) => onSettingsChange?.({
                      ...sessionSettings,
                      separateFollowUpTurns: e.target.checked,
                    })}
                  />
                }
                label={
                  <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
                    Separate follow-up agent turns
                  </Typography>
                }
              />
              <Typography variant="caption" component="div" sx={{ color: 'text.secondary', ml: 4, mt: -0.5 }}>
                Create new streaming boxes for each agent turn after widget submissions
              </Typography>
            </Box>
          )}

          {/* Global Settings Tab */}
          {settingsTab === 1 && (
            <Box sx={{ p: 2 }}>
              <Typography variant="body2" sx={{ fontSize: '0.8rem', mb: 1, fontWeight: 600 }}>
                Welcome Message
              </Typography>
              <Typography variant="caption" component="div" sx={{ color: 'text.secondary', mb: 1 }}>
                Shown when a new session starts (supports Markdown). Not part of the conversation context.
              </Typography>
              <TextField
                multiline
                minRows={4}
                maxRows={12}
                fullWidth
                size="small"
                value={globalSettings?.welcomeMessage || ''}
                onChange={(e) => onGlobalSettingsChange?.({
                  welcomeMessage: e.target.value,
                })}
                sx={{
                  '& .MuiInputBase-root': { fontSize: '0.8rem', fontFamily: 'monospace' },
                }}
              />
              {globalSettings?.welcomeMessageIsCustom && (
                <Button
                  size="small"
                  onClick={onResetWelcomeMessage}
                  sx={{ mt: 1, fontSize: '0.75rem', textTransform: 'none' }}
                >
                  Reset to default
                </Button>
              )}
            </Box>
          )}
        </Box>
      </Popover>

      {isStreaming && (
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          ⏱ {formatElapsed(elapsed)}
        </Typography>
      )}
    </Box>
  );
}

export default AgentStatusBar;
