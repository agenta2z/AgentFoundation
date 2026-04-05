/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * AgentChatPanel — Main real-mode chat UI container.
 *
 * Uses useSession() context for multi-session state management.
 * Features smart auto-scroll, per-agent streaming boxes, side-panel drawer,
 * task cards, and task panel view switching.
 */

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { Box, Collapse, Container, IconButton, List, ListItem, ListItemText, Paper, Typography } from '@mui/material';
import { useTheme, alpha } from '@mui/material/styles';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import { useSession } from '../../contexts/SessionContext';
import { useWorkspace } from '../../hooks/useWorkspace';
import { MarkdownRenderer } from '../common/MarkdownRenderer';
import { ChatInput } from '../chat/ChatInput';
import { AgentStatusBar } from './AgentStatusBar';
import { AgentStreamSection } from './AgentStreamSection';
import { AgentStreamDrawer } from './AgentStreamDrawer';
import { CommandAutocomplete } from './CommandAutocomplete';
import { TaskCard } from './TaskCard';
import { TaskPanel } from './TaskPanel';
import { getWidget } from '../widgets/WidgetRegistry';

export function AgentChatPanel() {
  const theme = useTheme();
  const {
    activeSession,
    activeSessionId,
    activeTabType,
    activeTask,
    streamingSections,
    isStreaming,
    config,
    connectionStatus,
    taskPhase,
    serverStatus,
    serverSessionId,
    serverId,
    isConnected,
    queueStatus,
    sendMessage,
    cancelRequest,
    clearMessages,
    pendingInput,
    sendPendingInputResponse,
    globalSettings,
    saveWelcomeMessage,
    resetWelcomeToDefault,
    fetchTurnData,
    dispatch,
  } = useSession();

  const messages = activeSession?.messages || [];
  const sessionSettings = activeSession?.settings || {};

  // Workspace browser state
  const activeWorkspace = activeTask?.workspace || null;
  const isTaskRunning = activeTask?.status === 'starting' || activeTask?.status === 'running';
  const { tree: workspaceTree, fetchFile } = useWorkspace(activeWorkspace, isTaskRunning);
  const [workspaceExpanded, setWorkspaceExpanded] = useState(false);

  const [inputValue, setInputValue] = useState('');
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerAgent, setDrawerAgent] = useState({ agentId: '', content: '' });
  const messagesEndRef = useRef(null);
  const scrollContainerRef = useRef(null);

  const isNearBottom = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 150;
  }, []);

  useEffect(() => {
    if (isNearBottom()) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, streamingSections, isNearBottom]);

  const handleSubmit = useCallback((e) => {
    e.preventDefault();
    if (!inputValue.trim() || !isConnected) return;

    if (inputValue.trim() === '/clear') {
      sendMessage(inputValue);
      clearMessages();
    } else {
      sendMessage(inputValue);
    }
    setInputValue('');
    setShowAutocomplete(false);
  }, [inputValue, isConnected, sendMessage, clearMessages]);

  const handleInputChange = useCallback((value) => {
    setInputValue(value);
    setShowAutocomplete(value.startsWith('/') && !value.includes(' '));
  }, []);

  const handleAutocompleteSelect = useCallback((command) => {
    setInputValue(command + ' ');
    setShowAutocomplete(false);
  }, []);

  const handleViewAll = useCallback((agentId, content) => {
    setDrawerAgent({ agentId, content });
    setDrawerOpen(true);
  }, []);

  const handleViewPrompt = useCallback(async (turnNumber) => {
    if (!activeSessionId || !fetchTurnData) return;
    const data = await fetchTurnData(activeSessionId, turnNumber);
    if (data) {
      setDrawerAgent({
        agentId: `Prompt (Turn ${turnNumber})`,
        content: data.rendered_prompt || '',
        turnData: data,
      });
      setDrawerOpen(true);
    } else {
      console.error(`View Prompt: failed to fetch turn data for session=${activeSessionId} turn=${turnNumber}`);
      setDrawerAgent({
        agentId: `Prompt (Turn ${turnNumber})`,
        content: `Failed to load prompt data for turn ${turnNumber}.\n\nThis can happen if:\n- The server was restarted (session data is in a different server directory)\n- The WebUI backend was started with a different --queue-root\n- The turn data hasn't been persisted yet\n\nCheck the browser console and server logs for details.`,
      });
      setDrawerOpen(true);
    }
  }, [activeSessionId, fetchTurnData]);

  const handleSettingsChange = useCallback((settings) => {
    if (activeSessionId) {
      dispatch({ type: 'UPDATE_SESSION_SETTINGS', sessionId: activeSessionId, settings });
    }
  }, [activeSessionId, dispatch]);

  const welcomeSaveTimerRef = useRef(null);
  const handleGlobalSettingsChange = useCallback((settings) => {
    if ('welcomeMessage' in settings) {
      // Update UI immediately, debounce the server save
      dispatch({
        type: 'UPDATE_GLOBAL_SETTINGS',
        settings: { welcomeMessage: settings.welcomeMessage, welcomeMessageIsCustom: true },
      });
      if (welcomeSaveTimerRef.current) clearTimeout(welcomeSaveTimerRef.current);
      welcomeSaveTimerRef.current = setTimeout(() => {
        saveWelcomeMessage(settings.welcomeMessage);
      }, 800);
    } else {
      dispatch({ type: 'UPDATE_GLOBAL_SETTINGS', settings });
    }
  }, [dispatch, saveWelcomeMessage]);

  // If viewing a task tab, render TaskPanel instead
  if (activeTabType === 'task') {
    return <TaskPanel />;
  }

  const isServerDown = serverStatus === 'server_down';
  const isSessionCreating = activeSession?.status === 'creating';

  const renderMessage = (msg, index) => {
    if (msg.hidden) return null;

    // Render task reference cards
    if (msg.role === 'task_ref') {
      return (
        <TaskCard
          key={`task-ref-${index}`}
          taskId={msg.taskId}
          label={msg.label}
          status={msg.status}
        />
      );
    }

    // Render committed streaming sections as persistent foldable boxes
    if (msg.role === 'assistant_stream') {
      if (sessionSettings.showStreamingSections === false || !msg.sections?.length) {
        // Settings say hide, or no sections — fall through to plain text render
        if (!msg.content) return null;
        // Render as plain assistant message
        return (
          <Box key={index} sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
            <Paper
              elevation={0}
              sx={{
                p: 2, maxWidth: '95%', backgroundColor: 'background.paper',
                borderRadius: 2, border: '1px solid', borderColor: 'divider',
              }}
            >
              <Box sx={{ '& p': { m: 0 }, '& pre': { overflow: 'auto' } }}>
                <MarkdownRenderer content={msg.content} />
              </Box>
            </Paper>
          </Box>
        );
      }
      return (
        <Box key={`stream-${index}`}>
          {msg.sections.map((section, sIdx) => (
            <AgentStreamSection
              key={`${section.agentId}-${sIdx}`}
              agentId={section.agentId}
              content={section.content}
              isComplete={true}
              onViewAll={handleViewAll}
              onViewPrompt={handleViewPrompt}
              turnNumber={msg.turnNumber || (index + 1)}
              defaultCollapsed={sessionSettings.streamSectionsCollapsed !== false}
              thinkingContent={section.thinkingContent}
              responseContent={section.responseContent}
              responsePhase={section.responsePhase}
            />
          ))}
        </Box>
      );
    }

    const isUser = msg.role === 'user';
    const isSystem = msg.role === 'system';
    const isError = msg.role === 'error';

    let bgColor = 'background.paper';
    let borderColor = 'divider';
    if (isUser) {
      bgColor = 'primary.dark';
      borderColor = 'primary.main';
    } else if (isSystem) {
      bgColor = theme.custom.surfaces.inputBg;
      borderColor = theme.custom.surfaces.inputBorder;
    } else if (isError) {
      bgColor = alpha(theme.palette.error.main, 0.1);
      borderColor = 'error.main';
    }

    return (
      <Box
        key={index}
        sx={{
          display: 'flex',
          justifyContent: isUser ? 'flex-end' : 'flex-start',
          mb: 2,
        }}
      >
        <Paper
          elevation={0}
          sx={{
            p: 2,
            maxWidth: '95%',
            backgroundColor: bgColor,
            borderRadius: 2,
            border: '1px solid',
            borderColor: borderColor,
          }}
        >
          {isSystem ? (
            <Typography
              variant="body2"
              sx={{
                fontFamily: 'monospace',
                fontSize: '0.85rem',
                color: 'text.secondary',
                whiteSpace: 'pre-wrap',
              }}
            >
              {msg.content}
            </Typography>
          ) : isError ? (
            <Typography variant="body2" sx={{ color: 'error.main' }}>
              {msg.content}
            </Typography>
          ) : (
            <Box sx={{ '& p': { m: 0 }, '& pre': { overflow: 'auto' } }}>
              <MarkdownRenderer content={msg.content} />
            </Box>
          )}
        </Paper>
      </Box>
    );
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', position: 'relative' }}>
      <AgentStatusBar
        connectionStatus={connectionStatus}
        model={config.model}
        targetPath={config.target_path}
        taskPhase={taskPhase}
        isStreaming={isStreaming}
        queueStatus={queueStatus}
        serverSessionId={serverSessionId}
        serverId={serverId}
        sessionSettings={sessionSettings}
        onSettingsChange={handleSettingsChange}
        globalSettings={globalSettings}
        onGlobalSettingsChange={handleGlobalSettingsChange}
        onResetWelcomeMessage={resetWelcomeToDefault}
      />

      {/* Server down overlay */}
      {isServerDown && (
        <Box sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: theme.custom.surfaces.scrim,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 10,
        }}>
          <Typography variant="h6" color="error">
            Server is disconnected. Waiting for reconnection...
          </Typography>
        </Box>
      )}

      {/* Session creating overlay */}
      {isSessionCreating && !isServerDown && (
        <Box sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: theme.custom.surfaces.scrim,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 10,
        }}>
          <Typography variant="h6" sx={{ color: 'text.secondary', fontStyle: 'italic' }}>
            Creating New Session...
          </Typography>
        </Box>
      )}

      <Container maxWidth="xl" sx={{ flex: 1, display: 'flex', flexDirection: 'column', py: 2, overflow: 'hidden' }}>
        <Box ref={scrollContainerRef} sx={{ flex: 1, overflow: 'auto', mb: 2, px: 1 }}>
          {globalSettings?.welcomeMessage && (
            <AgentStreamSection
              agentId="welcome"
              content={globalSettings.welcomeMessage}
              isComplete={true}
              showStatus={false}
              fitContent={true}
              onViewAll={handleViewAll}
              defaultCollapsed={false}
            />
          )}

          {messages.map((msg, index) => renderMessage(msg, index))}

          {streamingSections.length > 0 && (() => {
            const currentTurnNumber = messages
              .filter(m => m.role === 'assistant_stream' && m.turnNumber)
              .reduce((max, m) => Math.max(max, m.turnNumber), 0) + 1;
            return streamingSections.map((section, idx) => (
              <AgentStreamSection
                key={`${section.agentId}-${idx}`}
                agentId={section.agentId}
                content={section.content}
                isComplete={section.isComplete}
                onViewAll={handleViewAll}
                onViewPrompt={handleViewPrompt}
                turnNumber={section.turnNumber || currentTurnNumber}
                thinkingContent={section.thinkingContent}
                responseContent={section.responseContent}
                responsePhase={section.responsePhase}
              />
            ));
          })()}

          {isStreaming && streamingSections.length === 0 && (
            <AgentStreamSection
              agentId="system"
              content="*Thinking (may take a few minutes for initial thinking)...*"
              isComplete={false}
              isPlaceholder={true}
            />
          )}

          {/* Workspace file browser */}
          {activeWorkspace && workspaceTree && (
            <Paper
              elevation={0}
              sx={{
                mt: 2,
                mb: 1,
                backgroundColor: theme.custom.surfaces.cardBg,
                border: '1px solid',
                borderColor: 'divider',
                borderRadius: 2,
              }}
            >
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  px: 2,
                  py: 1,
                  cursor: 'pointer',
                  '&:hover': { backgroundColor: theme.custom.surfaces.inputBg },
                }}
                onClick={() => setWorkspaceExpanded((v) => !v)}
              >
                <FolderOpenIcon sx={{ mr: 1, fontSize: 18, color: 'text.secondary' }} />
                <Typography variant="subtitle2" sx={{ flex: 1, color: 'text.secondary' }}>
                  Workspace Files
                </Typography>
                <IconButton size="small">
                  {workspaceExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                </IconButton>
              </Box>
              <Collapse in={workspaceExpanded}>
                <Box sx={{ px: 2, pb: 2 }}>
                  {['outputs', 'results', 'logs', 'analysis'].map((category) => {
                    const files = workspaceTree[category];
                    if (!files || files.length === 0) return null;
                    return (
                      <Box key={category} sx={{ mb: 1 }}>
                        <Typography
                          variant="caption"
                          sx={{ color: 'text.secondary', textTransform: 'uppercase', fontWeight: 600 }}
                        >
                          {category}
                        </Typography>
                        <List dense disablePadding>
                          {files.map((f) => (
                            <ListItem
                              key={f.name}
                              sx={{
                                py: 0.25,
                                cursor: 'pointer',
                                '&:hover': { backgroundColor: theme.custom.surfaces.inputBg },
                                borderRadius: 1,
                              }}
                              onClick={async () => {
                                const data = await fetchFile(`${category}/${f.name}`);
                                if (data) {
                                  setDrawerAgent({
                                    agentId: f.name,
                                    content: typeof data.content === 'string'
                                      ? data.content
                                      : JSON.stringify(data.content, null, 2),
                                  });
                                  setDrawerOpen(true);
                                }
                              }}
                            >
                              <ListItemText
                                primary={f.name}
                                secondary={`${(f.size / 1024).toFixed(1)} KB`}
                                primaryTypographyProps={{ variant: 'body2', sx: { fontFamily: 'monospace', fontSize: '0.8rem' } }}
                                secondaryTypographyProps={{ variant: 'caption' }}
                              />
                            </ListItem>
                          ))}
                        </List>
                      </Box>
                    );
                  })}
                </Box>
              </Collapse>
            </Paper>
          )}

          {/* Pending input widget */}
          {pendingInput && (() => {
            const isCompound = pendingInput.inputMode?.metadata?.compound;
            const widgetType = pendingInput.widget?.widget_type
              || (isCompound ? 'multi_input' : null)
              || pendingInput.inputMode?.metadata?.widget_type
              || pendingInput.inputMode?.mode
              || 'text_input';
            const WidgetComponent = getWidget(widgetType);
            const widgetConfig = pendingInput.widget || pendingInput.inputMode || {};
            return (
              <Box sx={{ mb: 2, px: 1 }}>
                <Paper
                  elevation={0}
                  sx={{
                    p: 2,
                    backgroundColor: theme.custom.surfaces.highlightSubtle,
                    border: '1px solid',
                    borderColor: 'primary.main',
                    borderRadius: 2,
                  }}
                >
                  <WidgetComponent
                    config={widgetConfig}
                    onSubmit={(response) => {
                      const msg = { user_input: response };
                      if (pendingInput.widget?.widget_id) {
                        msg.widget_id = pendingInput.widget.widget_id;
                        msg.values = response.values || response;
                        msg.action = 'submit';
                      }
                      sendPendingInputResponse(msg);
                    }}
                  />
                </Paper>
              </Box>
            );
          })()}

          <div ref={messagesEndRef} />
        </Box>

        <Box sx={{ position: 'relative' }}>
          {showAutocomplete && (
            <CommandAutocomplete
              input={inputValue}
              onSelect={handleAutocompleteSelect}
              onClose={() => setShowAutocomplete(false)}
            />
          )}
          <ChatInput
            value={inputValue}
            onChange={handleInputChange}
            onSubmit={handleSubmit}
            disabled={!isConnected || isServerDown || isSessionCreating || !!pendingInput}
          />
        </Box>
      </Container>

      <AgentStreamDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        agentId={drawerAgent.agentId}
        content={drawerAgent.content}
        turnData={drawerAgent.turnData}
      />
    </Box>
  );
}

export default AgentChatPanel;
