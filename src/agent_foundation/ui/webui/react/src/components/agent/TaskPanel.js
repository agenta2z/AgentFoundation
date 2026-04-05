/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * TaskPanel — Read-only task dashboard view.
 * Shows task header, streaming sections, and back-to-session link.
 * No ChatInput (dashboard mode). Reuses AgentStreamSection.
 */

import React, { useRef, useEffect, useCallback, useState } from 'react';
import { Box, Container, Typography, Button, Chip, Popover, Snackbar } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { ArrowBack as BackIcon, ContentCopy as CopyIcon } from '@mui/icons-material';
import { useSession } from '../../contexts/SessionContext';
import { AgentStreamSection } from './AgentStreamSection';
import { AgentStreamDrawer } from './AgentStreamDrawer';

const STATUS_CONFIG = {
  starting: { label: 'Starting...', color: 'warning' },
  running: { label: 'Running', color: 'info' },
  completed: { label: 'Complete', color: 'success' },
  error: { label: 'Error', color: 'error' },
};

const AGENT_LABELS = {
  review: 'Review Agent',
  base: 'Base Agent',
  plan: 'Plan Agent',
  implementation: 'Implementation Agent',
};

function getStatusInfo(task) {
  if (task.currentAgent && task.status !== 'completed' && task.status !== 'error') {
    const agentLabel = AGENT_LABELS[task.currentAgent] || task.currentAgent;
    return { label: agentLabel, color: 'info' };
  }
  return STATUS_CONFIG[task.status] || STATUS_CONFIG.running;
}

function truncateLabel(text, maxLen = 60) {
  if (!text || text.length <= maxLen) return text;
  return text.slice(0, maxLen) + '...';
}

export function TaskPanel() {
  const theme = useTheme();
  const {
    activeTask,
    streamingSections,
    isStreaming,
    switchTab,
  } = useSession();

  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerAgent, setDrawerAgent] = useState({ agentId: '', content: '' });
  const [cmdAnchor, setCmdAnchor] = useState(null);
  const [copiedSnack, setCopiedSnack] = useState(false);
  const scrollRef = useRef(null);
  const bottomRef = useRef(null);

  const isNearBottom = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 150;
  }, []);

  useEffect(() => {
    if (isNearBottom()) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [streamingSections, isNearBottom]);

  const handleViewAll = useCallback((agentId, content) => {
    setDrawerAgent({ agentId, content });
    setDrawerOpen(true);
  }, []);

  if (!activeTask) {
    return (
      <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Typography color="text.secondary">No task selected</Typography>
      </Box>
    );
  }

  const statusInfo = getStatusInfo(activeTask);

  const handleCmdClick = (e) => setCmdAnchor(e.currentTarget);
  const handleCmdClose = () => setCmdAnchor(null);
  const handleCopyCmd = () => {
    navigator.clipboard.writeText(activeTask.request || activeTask.label);
    setCopiedSnack(true);
    setCmdAnchor(null);
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Task header */}
      <Box sx={{
        px: 2,
        py: 1,
        borderBottom: '1px solid',
        borderColor: 'divider',
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        backgroundColor: theme.custom.surfaces.sidebarBg,
      }}>
        <Button
          size="small"
          startIcon={<BackIcon />}
          onClick={() => switchTab(activeTask.parentSessionId, 'session')}
          sx={{ textTransform: 'none', fontSize: '0.8rem' }}
        >
          Back to session
        </Button>
        <Typography
          variant="subtitle2"
          onClick={handleCmdClick}
          sx={{
            fontWeight: 600,
            flex: 1,
            cursor: 'pointer',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            '&:hover': { textDecoration: 'underline dotted', opacity: 0.8 },
          }}
        >
          Task: {truncateLabel(activeTask.request || activeTask.label)}
        </Typography>
        <Popover
          open={Boolean(cmdAnchor)}
          anchorEl={cmdAnchor}
          onClose={handleCmdClose}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
          transformOrigin={{ vertical: 'top', horizontal: 'left' }}
        >
          <Box
            onClick={handleCopyCmd}
            sx={{
              p: 2,
              maxWidth: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'flex-start',
              gap: 1,
              '&:hover': { backgroundColor: 'action.hover' },
            }}
          >
            <CopyIcon sx={{ fontSize: 16, mt: 0.3, color: 'text.secondary', flexShrink: 0 }} />
            <Typography
              variant="body2"
              sx={{ fontFamily: 'monospace', fontSize: '0.8rem', wordBreak: 'break-all' }}
            >
              {activeTask.request || activeTask.label}
            </Typography>
          </Box>
        </Popover>
        <Chip
          label={statusInfo.label}
          size="small"
          color={statusInfo.color}
          variant="outlined"
          sx={{ height: 22, fontSize: '0.7rem' }}
        />
        <Snackbar
          open={copiedSnack}
          autoHideDuration={2000}
          onClose={() => setCopiedSnack(false)}
          message="Command copied to clipboard"
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        />
      </Box>

      {/* Streaming sections */}
      <Container maxWidth="xl" sx={{ flex: 1, display: 'flex', flexDirection: 'column', py: 2, overflow: 'hidden' }}>
        <Box ref={scrollRef} sx={{ flex: 1, overflow: 'auto', mb: 2, px: 1 }}>
          {streamingSections.length === 0 && !isStreaming && (
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', opacity: 0.5 }}>
              <Typography variant="body2" color="text.secondary">
                {activeTask.status === 'completed' ? 'Task completed.' : 'Waiting for task output...'}
              </Typography>
            </Box>
          )}

          {streamingSections.map((section, idx) => (
            <AgentStreamSection
              key={`${section.agentId}-${idx}`}
              agentId={section.agentId}
              content={section.content}
              isComplete={section.isComplete}
              onViewAll={handleViewAll}
            />
          ))}

          {isStreaming && streamingSections.length === 0 && (
            <AgentStreamSection
              agentId="system"
              content="*Thinking (may take a few minutes for initial thinking)...*"
              isComplete={false}
              isPlaceholder={true}
            />
          )}

          <div ref={bottomRef} />
        </Box>
      </Container>

      <AgentStreamDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        agentId={drawerAgent.agentId}
        content={drawerAgent.content}
      />
    </Box>
  );
}

export default TaskPanel;
