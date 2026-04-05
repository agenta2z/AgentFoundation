/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * SessionContext — React Context wiring useSessionManager + WebSocket connections.
 *
 * First React Context in this codebase. Provides session/task state,
 * streaming sections, and actions to all consuming components.
 */

import React, { createContext, useContext, useCallback, useMemo, useEffect, useRef } from 'react';
import { useSessionManager } from '../hooks/useSessionManager';
import { useAgentWebSocket } from '../hooks/useAgentWebSocket';
import { useSessionApi } from '../hooks/useSessionApi';

const SessionContext = createContext(null);

export function SessionProvider({ children }) {
  const {
    state,
    dispatch,
    streamingSections,
    isStreaming,
    handleStreamToken,
    handleStreamStart,
    handleStreamEnd,
    switchTab,
    getStreamState,
    remapStreamingRef,
    clearStreamingDisplay,
    persistSessionIds,
  } = useSessionManager();

  // HTTP API for reading session data from file store
  const {
    fetchSessionState,
    fetchSessionMessages,
    fetchWelcomeMessage,
    updateWelcomeMessage,
    resetWelcomeMessage,
    fetchTurnData,
  } = useSessionApi();

  // WebSocket message handler — routes to reducer or streaming refs
  const handleWsMessage = useCallback((data) => {
    const sessionId = state.activeSessionId;

    // Handle session-independent messages BEFORE sessionId check
    switch (data.type) {
      case 'session_init':
        // Server confirmed the session — this is the handshake completion.
        // Set status to "connected", store session_id, apply config, and sync session list.
        dispatch({ type: 'SET_SERVER_STATUS', status: 'connected' });
        if (data.session_id) {
          dispatch({ type: 'SET_SERVER_SESSION_ID', serverSessionId: data.session_id });
          // Only remap if the active session is still in 'creating' status
          // (i.e., a client-generated ID awaiting server confirmation).
          // Do NOT remap for switch_session responses — that creates an
          // infinite loop: CONFIRM_SESSION changes activeSessionId →
          // useEffect sends switch_session → server responds session_init →
          // CONFIRM_SESSION fires again → repeat.
          const activeSession = state.sessions?.[state.activeSessionId];
          if (activeSession?.status === 'creating') {
            dispatch({
              type: 'CONFIRM_SESSION',
              clientSessionId: state.activeSessionId,
              serverSessionId: data.session_id,
            });
          }
        }
        if (data.server_id) {
          dispatch({ type: 'SET_SERVER_ID', serverId: data.server_id });
        }
        if (data.config) {
          dispatch({ type: 'CONFIG_UPDATE', config: data.config });
        }
        // Note: we intentionally do NOT dispatch SYNC_SESSION_LIST here.
        // The server's session list includes sessions from previous page loads
        // (they persist on disk), which would confuse the UI by adding stale
        // sessions the user didn't create in this browser session. Session
        // restore will be a separate explicit feature.
        return;

      case 'config_update':
        if (data.config) {
          dispatch({ type: 'CONFIG_UPDATE', config: data.config });
        }
        return;

      case 'server_status':
        dispatch({ type: 'SET_SERVER_STATUS', status: data.status });
        return;

      case 'queue_status_response':
        dispatch({
          type: 'QUEUE_STATUS_UPDATE',
          queueStatus: { global: data.global, session: data.session },
        });
        return;

      case 'session_notification':
        // Server wrote state to file store — fetch updated data via HTTP
        if (data.change_types?.includes('config')) {
          fetchSessionState(data.session_id).then(st => {
            if (st?.info) dispatch({ type: 'CONFIG_UPDATE', config: st.info });
          });
        }
        if (data.change_types?.includes('content')) {
          fetchSessionMessages(data.session_id).then(messages => {
            if (messages) dispatch({ type: 'RESTORE_MESSAGES', sessionId: data.session_id, messages });
          });
        }
        return;

      default:
        break;
    }

    // All other messages require an active session
    if (!sessionId) return;

    switch (data.type) {
      case 'token':
        handleStreamToken(data);
        break;

      case 'message_start': {
        // If there's prior streaming content (from a previous turn in the
        // same agentic loop), commit it as a completed message first so
        // the new turn gets its own streaming box.
        
        const ss = sessionId ? getStreamState(sessionId) : null;
        const hasPriorContent = ss && ss.agentOrder && ss.agentOrder.length > 0;

        if (hasPriorContent) {
          const settings = state.sessions?.[sessionId]?.settings || {};
          if (settings.separateFollowUpTurns !== false) {
            // Finalize prior sections and commit as assistant_stream message
            handleStreamEnd({});
            const priorSections = (ss.sections || []).map(s => ({
              agentId: s.agentId, content: s.content, metadata: s.metadata,
              thinkingContent: s.thinkingContent, responseContent: s.responseContent,
              responsePhase: s.responsePhase,
            }));
            if (priorSections.length > 0) {
              dispatch({
                type: 'STREAM_END', sessionId,
                finalContent: '', streamSections: priorSections, turnNumber: 1,
                keepStreamTarget: true,
              });
            }
            // Clear refs so they do not re-appear
            ss.sections = []; ss.agentContents = {}; ss.agentOrder = [];
            clearStreamingDisplay();
          }
        }
        handleStreamStart();
        break;
      }

      case 'message_end':
        handleStreamEnd(data);
        // Read finalized sections from the streaming ref (just updated by
        // handleStreamEnd), not from React state which is stale in this closure.
        {
          let currentSections = null;
          if (!data.task_id && sessionId) {
            const ss = getStreamState(sessionId);
            if (ss && ss.sections.length > 0) {
              currentSections = ss.sections.map(s => ({
                agentId: s.agentId,
                content: s.content,
                metadata: s.metadata,
                thinkingContent: s.thinkingContent,
                responseContent: s.responseContent,
                responsePhase: s.responsePhase,
              }));
            }
            // If no sections were tracked (e.g., tokens came via file tailer
            // not handleStreamToken), create a synthetic section from
            // final_content so the message renders as a foldable streaming box.
            if (!currentSections && data.final_content) {
              currentSections = [{
                agentId: 'agent',
                content: data.final_content,
                metadata: { agent_id: 'agent' },
                thinkingContent: '',
                responseContent: data.final_content,
                responsePhase: 'no_tags',
              }];
            }
            // Clear the ref's sections so they don't re-appear as
            // duplicates when switchTab reads the ref on tab switch.
            if (ss) {
              ss.sections = [];
              ss.agentContents = {};
              ss.agentOrder = [];
            }
          }
          dispatch({
            type: 'STREAM_END',
            sessionId,
            taskId: data.task_id,
            finalContent: data.final_content,
            streamSections: currentSections,
            turnNumber: data.turn_number,
          });
        }
        // Clear live streaming display (React state).
        if (!data.task_id) {
          clearStreamingDisplay();
        }
        break;

      case 'command_response': {
        // Clear streaming state (refs) regardless of response type
        handleStreamEnd(data);
        // Clear the ref's sections to prevent duplicates on tab switch
        if (sessionId) {
          const cmdSs = getStreamState(sessionId);
          if (cmdSs) {
            cmdSs.sections = [];
            cmdSs.agentContents = {};
            cmdSs.agentOrder = [];
          }
        }
        clearStreamingDisplay();

        if (data.data?.action === 'conversation') {
          // Conversation chat response → visible assistant chat bubble
          dispatch({
            type: 'CONVERSATION_RESPONSE',
            sessionId,
            content: data.content || data.data?.message || '',
            turnNumber: data.data?.turn_number || data.turn_number,
          });
        } else {
          // Slash command response → system message (existing behavior)
          dispatch({
            type: 'COMMAND_RESPONSE',
            sessionId,
            content: data.content,
          });
          // Handle config changes from command responses
          if (data.data?.config_changed && data.data?.updated_config) {
            dispatch({ type: 'CONFIG_UPDATE', config: data.data.updated_config });
          }
        }
        break;
      }

      case 'task_status': {
        const serverTaskId = data.task_id;
        if (serverTaskId) {
          if (!state.tasks[serverTaskId] && data.status === 'starting') {
            const pendingEntry = Object.entries(state.tasks).find(
              ([, t]) => t.status === 'starting' && t.parentSessionId === sessionId
            );
            if (pendingEntry) {
              const [oldId] = pendingEntry;
              remapStreamingRef(oldId, serverTaskId);
            }
          }
          dispatch({
            type: 'TASK_STATUS',
            taskId: serverTaskId,
            status: data.status,
            request: data.request,
            workspace: data.workspace,
          });
        }
        break;
      }

      case 'status':
        if (data.status === 'complete' || data.status === 'error') {
          handleStreamEnd(data);
        }
        break;

      case 'error':
        dispatch({
          type: 'STREAM_ERROR',
          sessionId,
          message: data.message,
        });
        handleStreamEnd(data);
        break;

      case 'session_sync_response':
        if (data.is_restored) {
          dispatch({ type: 'SYNC_FROM_SERVER', data });
          // Fetch full conversation history from file store
          if (sessionId) {
            fetchSessionMessages(sessionId).then(messages => {
              if (messages) {
                dispatch({ type: 'RESTORE_MESSAGES', sessionId, messages });
              }
            });
          }
        } else {
          dispatch({ type: 'CLEAR_ALL' });
        }
        break;

      case 'pending_input':
        dispatch({
          type: 'PENDING_INPUT',
          sessionId,
          content: data.content,
          inputMode: data.input_mode,
          widget: data.widget,
        });
        break;

      case 'widget_update':
        dispatch({
          type: 'WIDGET_UPDATE',
          sessionId,
          widget: data.widget,
        });
        break;

      default:
        break;
    }
  }, [state.activeSessionId, state.tasks, dispatch, handleStreamToken, handleStreamStart, handleStreamEnd, getStreamState, clearStreamingDisplay, remapStreamingRef, fetchSessionState, fetchSessionMessages]);

  const handleStatusChange = useCallback((status) => {
    // WebSocket transport status changes. "connected" is NOT set here —
    // it's set only when session_init is received from the server.
    const mapped = status === 'connecting' ? 'connecting'
      : status === 'disconnected' ? 'disconnected'
      : status === 'error' ? 'server_down'
      : status;
    dispatch({ type: 'SET_SERVER_STATUS', status: mapped });
  }, [dispatch]);

  // WebSocket connection — one per active session
  const { send, cancelRequest } = useAgentWebSocket(
    state.activeSessionId,
    handleWsMessage,
    handleStatusChange,
  );

  // Send message action
  const sendMessage = useCallback((text) => {
    if (!text.trim()) return;
    const sessionId = state.activeSessionId;
    if (!sessionId) return;

    // Detect /task command
    if (text.startsWith('/task ')) {
      const request = text.slice(6).trim();
      const taskId = `task-${Date.now().toString(36)}`;
      dispatch({
        type: 'CREATE_TASK',
        sessionId,
        taskId,
        label: request.slice(0, 40),
        request,
      });
    } else {
      dispatch({
        type: 'ADD_MESSAGE',
        sessionId,
        message: { role: 'user', content: text },
      });
    }

    // Handle /clear locally
    if (text.trim() === '/clear') {
      dispatch({ type: 'CLEAR_MESSAGES', sessionId });
    }

    send({ type: 'message', content: text });
  }, [state.activeSessionId, dispatch, send]);

  // Create new session — dispatch to reducer, then clear stale streaming state
  const createSession = useCallback((label) => {
    dispatch({ type: 'CREATE_SESSION', label });
    // CREATE_SESSION sets activeTabId in the reducer but doesn't swap the
    // displayed streaming sections (that's switchTab's job). Clear stale
    // streaming state so the old session's content doesn't bleed into the
    // new session's view.
    clearStreamingDisplay();
  }, [dispatch, clearStreamingDisplay]);

  // Clear messages for active session
  const clearMessages = useCallback(() => {
    if (state.activeSessionId) {
      dispatch({ type: 'CLEAR_MESSAGES', sessionId: state.activeSessionId });
    }
  }, [state.activeSessionId, dispatch]);

  // Send a response to a pending input prompt (text or widget)
  const sendPendingInputResponse = useCallback((response) => {
    const sessionId = state.activeSessionId;
    if (!sessionId) return;

    const msg = {
      type: 'pending_input_response',
      session_id: sessionId,
      ...response,
    };
    send(msg);
    dispatch({ type: 'CLEAR_PENDING_INPUT', sessionId });
  }, [state.activeSessionId, send, dispatch]);

  // Notify server when active session changes (session switch)
  const prevSessionIdRef = useRef(state.activeSessionId);
  useEffect(() => {
    if (
      state.activeSessionId &&
      state.activeSessionId !== prevSessionIdRef.current
    ) {
      send({ type: 'switch_session', session_id: state.activeSessionId });
      prevSessionIdRef.current = state.activeSessionId;
    }
  }, [state.activeSessionId, send]);

  // Persist session IDs to localStorage when sessions change
  useEffect(() => {
    persistSessionIds();
  }, [state.sessions, persistSessionIds]);

  // Periodic queue status polling (every 10s when connected)
  const queuePollRef = useRef(null);
  useEffect(() => {
    if (state.serverStatus === 'connected' && state.activeSessionId) {
      // Request immediately on connect
      send({ type: 'queue_status' });
      queuePollRef.current = setInterval(() => {
        send({ type: 'queue_status' });
      }, 10000);
    }
    return () => {
      if (queuePollRef.current) {
        clearInterval(queuePollRef.current);
        queuePollRef.current = null;
      }
    };
  }, [state.serverStatus, state.activeSessionId, send]);

  // Fetch welcome message from server on mount
  useEffect(() => {
    fetchWelcomeMessage().then(data => {
      if (data) {
        dispatch({
          type: 'UPDATE_GLOBAL_SETTINGS',
          settings: {
            welcomeMessage: data.content || '',
            welcomeMessageIsCustom: data.is_custom || false,
          },
        });
      }
    });
  }, [fetchWelcomeMessage, dispatch]);

  // Save welcome message to server (UI update is handled by the caller)
  const saveWelcomeMessage = useCallback(async (content) => {
    await updateWelcomeMessage(content);
  }, [updateWelcomeMessage]);

  // Reset welcome message to default
  const resetWelcomeToDefault = useCallback(async () => {
    const result = await resetWelcomeMessage();
    if (result?.success) {
      dispatch({
        type: 'UPDATE_GLOBAL_SETTINGS',
        settings: { welcomeMessage: result.content || '', welcomeMessageIsCustom: false },
      });
    }
  }, [dispatch, resetWelcomeMessage]);

  // Memoized selectors
  const activeSession = useMemo(() =>
    state.sessions[state.activeSessionId] || null,
    [state.sessions, state.activeSessionId]
  );

  const activeTask = useMemo(() =>
    state.activeTabType === 'task' ? state.tasks[state.activeTabId] || null : null,
    [state.activeTabType, state.activeTabId, state.tasks]
  );

  const sessionList = useMemo(() =>
    Object.values(state.sessions).sort((a, b) => a.createdAt - b.createdAt),
    [state.sessions]
  );

  const isConnected = state.serverStatus === 'connected';

  const value = useMemo(() => ({
    // State
    sessions: state.sessions,
    tasks: state.tasks,
    activeSession,
    activeTask,
    activeSessionId: state.activeSessionId,
    activeTabId: state.activeTabId,
    activeTabType: state.activeTabType,
    sessionList,
    config: state.config,
    taskPhase: state.taskPhase,
    serverStatus: state.serverStatus,
    serverSessionId: state.serverSessionId,
    serverId: state.serverId,
    isConnected,
    connectionStatus: state.serverStatus,
    queueStatus: state.queueStatus,
    globalSettings: state.globalSettings,

    // Streaming
    streamingSections,
    isStreaming,

    // Pending input
    pendingInput: state.pendingInput,

    // Actions
    sendMessage,
    cancelRequest,
    clearMessages,
    createSession,
    switchTab,
    sendPendingInputResponse,
    saveWelcomeMessage,
    resetWelcomeToDefault,
    fetchTurnData,
    dispatch,
  }), [
    state.sessions, state.tasks, activeSession, activeTask,
    state.activeSessionId, state.activeTabId, state.activeTabType,
    sessionList, state.config, state.taskPhase, state.serverStatus, state.serverSessionId, state.serverId,
    isConnected, state.queueStatus, state.globalSettings, state.pendingInput, streamingSections, isStreaming,
    sendMessage, cancelRequest, clearMessages, createSession, switchTab,
    sendPendingInputResponse, saveWelcomeMessage, resetWelcomeToDefault, fetchTurnData, dispatch,
  ]);

  return (
    <SessionContext.Provider value={value}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const ctx = useContext(SessionContext);
  if (!ctx) {
    throw new Error('useSession must be used within a SessionProvider');
  }
  return ctx;
}

export default SessionContext;
