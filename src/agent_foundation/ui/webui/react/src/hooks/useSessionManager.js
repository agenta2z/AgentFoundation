/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * useSessionManager — Multi-session state manager with useReducer + streaming refs.
 *
 * Reducer holds metadata (session list, task status, active tab, config).
 * Streaming content (agentContents, agentOrder) lives in mutable refs
 * (Map<targetId, StreamState>) to avoid re-renders per token.
 */

import { useReducer, useRef, useState, useCallback } from 'react';

// ── Helpers ────────────────────────────────────────────────────────

const DEFAULT_SESSION_SETTINGS = {
  showStreamingSections: true,
  streamSectionsCollapsed: false,
  separateFollowUpTurns: true,
};

const DEFAULT_GLOBAL_SETTINGS = {
  welcomeMessage: '', // Loaded from server on mount
  welcomeMessageIsCustom: false,
};

/**
 * Parse `<Response>...</Response>` tags from streamed LLM output.
 * Content before <Response> is "thinking"; content inside is the response.
 * Returns { phase, thinkingContent, responseContent }.
 */
function parseResponseTags(rawContent) {
  const responseStart = rawContent.indexOf('<Response>');
  if (responseStart === -1) {
    return { phase: 'pre_response', thinkingContent: rawContent, responseContent: '' };
  }
  const thinking = rawContent.slice(0, responseStart).trim();
  const afterTag = rawContent.slice(responseStart + '<Response>'.length);
  const responseEnd = afterTag.indexOf('</Response>');
  if (responseEnd === -1) {
    return { phase: 'in_response', thinkingContent: thinking, responseContent: afterTag };
  }
  return { phase: 'post_response', thinkingContent: thinking, responseContent: afterTag.slice(0, responseEnd) };
}

function generateId(prefix) {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
}

function loadSessionSettings(sessionId) {
  try {
    const raw = localStorage.getItem(`rankevolve_settings_${sessionId}`);
    if (raw) return { ...DEFAULT_SESSION_SETTINGS, ...JSON.parse(raw) };
  } catch { /* ignore */ }
  return { ...DEFAULT_SESSION_SETTINGS };
}

function saveSessionSettings(sessionId, settings) {
  try {
    localStorage.setItem(`rankevolve_settings_${sessionId}`, JSON.stringify(settings));
  } catch { /* ignore */ }
}

function createSession(label) {
  return {
    id: generateId('ses'),
    label,
    createdAt: Date.now(),
    workspacePath: '',
    messages: [],
    taskIds: [],
    status: 'creating', // 'creating' | 'ready'
    settings: { ...DEFAULT_SESSION_SETTINGS },
  };
}

// ── Reducer ────────────────────────────────────────────────────────

const initialState = {
  sessions: {},
  tasks: {},
  activeSessionId: null,
  activeTabId: null,
  activeTabType: 'session', // 'session' | 'task'
  currentStreamTarget: null, // { type, sessionId, taskId? }
  config: { model: '', target_path: '', provider: '' },
  taskPhase: null,
  serverStatus: 'connecting',
  queueStatus: null,
  pendingInput: null, // { content, inputMode, widget, sessionId }
  globalSettings: { ...DEFAULT_GLOBAL_SETTINGS },
};

function sessionReducer(state, action) {
  switch (action.type) {
    case 'CREATE_SESSION': {
      const session = createSession(action.label || `Session ${Object.keys(state.sessions).length + 1}`);
      return {
        ...state,
        sessions: { ...state.sessions, [session.id]: session },
        activeSessionId: session.id,
        activeTabId: session.id,
        activeTabType: 'session',
      };
    }

    case 'CONFIRM_SESSION': {
      // Server confirmed the session via session_init.
      // Remap client-generated ID to server-assigned ID and mark ready.
      const { clientSessionId, serverSessionId: srvId } = action;
      const savedSettings = loadSessionSettings(srvId);
      const session = state.sessions[clientSessionId];
      if (!session) {
        // Client ID not found — maybe already confirmed or synced.
        // If server ID already exists, just mark it ready.
        const existing = state.sessions[srvId];
        if (existing && existing.status === 'creating') {
          return {
            ...state,
            sessions: {
              ...state.sessions,
              [srvId]: { ...existing, label: srvId, status: 'ready', settings: savedSettings },
            },
          };
        }
        return state;
      }
      // Remove old client-keyed entry, add under server ID
      const { [clientSessionId]: _removed, ...restSessions } = state.sessions;
      const confirmed = { ...session, id: srvId, label: srvId, status: 'ready', settings: savedSettings };
      const newSessions = { ...restSessions, [srvId]: confirmed };
      return {
        ...state,
        sessions: newSessions,
        activeSessionId: state.activeSessionId === clientSessionId ? srvId : state.activeSessionId,
        activeTabId: state.activeTabId === clientSessionId ? srvId : state.activeTabId,
      };
    }

    case 'CREATE_TASK': {
      const { sessionId, taskId, label, request } = action;
      const task = {
        id: taskId,
        label: label || request?.slice(0, 40) || 'Task',
        parentSessionId: sessionId,
        status: 'starting',
        workspacePath: '',
        request: request || '',
      };
      const session = state.sessions[sessionId];
      if (!session) return state;
      const taskRef = { role: 'task_ref', taskId, label: task.label, status: 'starting' };
      return {
        ...state,
        tasks: { ...state.tasks, [taskId]: task },
        sessions: {
          ...state.sessions,
          [sessionId]: {
            ...session,
            taskIds: [...session.taskIds, taskId],
            messages: [...session.messages, taskRef],
          },
        },
        activeTabId: taskId,
        activeTabType: 'task',
        currentStreamTarget: { type: 'task', sessionId, taskId },
      };
    }

    case 'SWITCH_TAB': {
      const { tabId, tabType } = action;
      const updates = { activeTabId: tabId, activeTabType: tabType };
      if (tabType === 'session') {
        updates.activeSessionId = tabId;
      } else if (tabType === 'task') {
        const task = state.tasks[tabId];
        if (task) updates.activeSessionId = task.parentSessionId;
      }
      return { ...state, ...updates };
    }

    case 'ADD_MESSAGE': {
      const { sessionId, message } = action;
      const session = state.sessions[sessionId];
      if (!session) return state;
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [sessionId]: {
            ...session,
            messages: [...session.messages, message],
          },
        },
        currentStreamTarget: { type: 'session', sessionId },
      };
    }

    case 'STREAM_END': {
      const { sessionId, taskId, finalContent, streamSections, turnNumber, keepStreamTarget } = action;
      let newState = { ...state, currentStreamTarget: keepStreamTarget ? state.currentStreamTarget : null };
      if (taskId && newState.tasks[taskId]) {
        // Add hidden assistant message to session for conversation history
        const session = newState.sessions[sessionId];
        if (session) {
          newState.sessions = {
            ...newState.sessions,
            [sessionId]: {
              ...session,
              messages: [...session.messages, {
                role: 'assistant',
                content: finalContent || '',
                hidden: true,
              }],
            },
          };
        }
      } else if (sessionId) {
        // Non-task session stream — commit as assistant_stream message
        // with sections data so they render as persistent foldable boxes.
        const session = newState.sessions[sessionId];
        if (session && (finalContent || (streamSections && streamSections.length > 0))) {
          const msg = {
            role: 'assistant_stream',
            content: finalContent || '',
            sections: streamSections || [],
            turnNumber: turnNumber || null,
          };
          newState.sessions = {
            ...newState.sessions,
            [sessionId]: {
              ...session,
              messages: [...session.messages, msg],
            },
          };
        }
      }
      return newState;
    }

    case 'COMMAND_RESPONSE': {
      const { sessionId, content } = action;
      const session = state.sessions[sessionId];
      if (!session) return state;
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [sessionId]: {
            ...session,
            messages: [...session.messages, { role: 'system', content }],
          },
        },
        currentStreamTarget: null,
      };
    }

    case 'CONVERSATION_RESPONSE': {
      const { sessionId, content, turnNumber } = action;
      const session = state.sessions[sessionId];
      if (!session) return state;
      const msg = { role: 'assistant', content };
      if (turnNumber) msg.turnNumber = turnNumber;
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [sessionId]: {
            ...session,
            messages: [...session.messages, msg],
          },
        },
        currentStreamTarget: null,
      };
    }

    case 'CONFIG_UPDATE': {
      return {
        ...state,
        config: { ...state.config, ...action.config },
      };
    }

    case 'TASK_STATUS': {
      const { taskId: serverTaskId, status, request, workspace } = action;
      let task = state.tasks[serverTaskId];
      let resolvedTaskId = serverTaskId;
      let newTasks = state.tasks;

      // Task ID reconciliation: the frontend generates a temporary ID when
      // dispatching CREATE_TASK, but the server generates its own task_id.
      // When the first TASK_STATUS ("starting") arrives with the server's ID
      // and no matching task exists, find the pending task and remap its ID.
      if (!task && status === 'starting') {
        const pendingEntry = Object.entries(state.tasks).find(
          ([, t]) => t.status === 'starting' && t.parentSessionId === state.activeSessionId
        );
        if (pendingEntry) {
          const [oldId, pendingTask] = pendingEntry;
          task = { ...pendingTask, id: serverTaskId };
          resolvedTaskId = serverTaskId;
          // Remove old entry, add under server's ID
          const { [oldId]: _removed, ...rest } = state.tasks;
          newTasks = { ...rest, [serverTaskId]: task };
          // Also update taskIds array and task_ref messages in parent session
          const session = state.sessions[task.parentSessionId];
          if (session) {
            const updatedTaskIds = session.taskIds.map(id => id === oldId ? serverTaskId : id);
            const updatedMsgs = session.messages.map(m =>
              m.role === 'task_ref' && m.taskId === oldId
                ? { ...m, taskId: serverTaskId }
                : m
            );
            // Also update currentStreamTarget if it pointed to the old ID
            let newStreamTarget = state.currentStreamTarget;
            if (newStreamTarget?.taskId === oldId) {
              newStreamTarget = { ...newStreamTarget, taskId: serverTaskId };
            }
            return {
              ...state,
              tasks: newTasks,
              sessions: {
                ...state.sessions,
                [task.parentSessionId]: {
                  ...session,
                  taskIds: updatedTaskIds,
                  messages: updatedMsgs,
                },
              },
              activeTabId: state.activeTabId === oldId ? serverTaskId : state.activeTabId,
              currentStreamTarget: newStreamTarget,
            };
          }
          return { ...state, tasks: newTasks };
        }
        // No pending task found — server initiated the task directly.
        // Create the task entry and tab on the fly.
        const parentSessionId = state.activeSessionId;
        const taskLabel = (request || 'Task').slice(0, 40);
        const newTask = {
          id: serverTaskId,
          label: taskLabel,
          parentSessionId,
          status: 'starting',
          request: request || 'Task',
          workspace,
          messages: [],
          createdAt: Date.now(),
        };
        const parentSession = state.sessions[parentSessionId];
        const taskRef = {
          role: 'task_ref',
          taskId: serverTaskId,
          label: taskLabel,
          status: 'starting',
          request: request || 'Task',
        };
        return {
          ...state,
          tasks: { ...state.tasks, [serverTaskId]: newTask },
          sessions: {
            ...state.sessions,
            [parentSessionId]: {
              ...parentSession,
              taskIds: [...(parentSession?.taskIds || []), serverTaskId],
              messages: [...(parentSession?.messages || []), taskRef],
            },
          },
          // Don't auto-switch to task tab — the conversation turn is still
          // streaming. Let the user click "Open Task" to view the task panel.
          // Keep currentStreamTarget on the session so remaining conversation
          // tokens render in the session's streaming section.
        };
      }

      if (!task) return state;
      const updated = { ...task, status };
      if (request) updated.request = request;
      if (workspace) updated.workspace = workspace;
      // Update task_ref message status in parent session
      const session = state.sessions[task.parentSessionId];
      let updatedSessions = state.sessions;
      if (session) {
        const msgs = session.messages.map(m =>
          m.role === 'task_ref' && m.taskId === resolvedTaskId ? { ...m, status } : m
        );
        updatedSessions = {
          ...state.sessions,
          [task.parentSessionId]: { ...session, messages: msgs },
        };
      }
      return {
        ...state,
        tasks: { ...newTasks, [resolvedTaskId]: updated },
        sessions: updatedSessions,
        taskPhase: status === 'completed' || status === 'error' ? null : state.taskPhase,
      };
    }

    case 'UPDATE_TASK_AGENT': {
      const { taskId: agTaskId, agentId: agAgent } = action;
      const agTask = state.tasks[agTaskId];
      if (!agTask || agTask.currentAgent === agAgent) return state;
      return {
        ...state,
        tasks: { ...state.tasks, [agTaskId]: { ...agTask, currentAgent: agAgent } },
      };
    }

    case 'STREAM_ERROR': {
      const { sessionId, message } = action;
      const session = state.sessions[sessionId];
      if (!session) return state;
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [sessionId]: {
            ...session,
            messages: [...session.messages, { role: 'error', content: message }],
          },
        },
        currentStreamTarget: null,
      };
    }

    case 'CLEAR_MESSAGES': {
      const { sessionId } = action;
      const session = state.sessions[sessionId];
      if (!session) return state;
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [sessionId]: { ...session, messages: [], taskIds: [] },
        },
        tasks: Object.fromEntries(
          Object.entries(state.tasks).filter(([, t]) => t.parentSessionId !== sessionId)
        ),
        taskPhase: null,
      };
    }

    case 'SET_WORKSPACE_PATH': {
      const { sessionId, workspacePath } = action;
      const session = state.sessions[sessionId];
      if (!session) return state;
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [sessionId]: { ...session, workspacePath },
        },
      };
    }

    case 'UPDATE_SESSION_SETTINGS': {
      const { sessionId: settSid, settings } = action;
      const settSession = state.sessions[settSid];
      if (!settSession) return state;
      const merged = { ...settSession.settings, ...settings };
      saveSessionSettings(settSid, merged);
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [settSid]: {
            ...settSession,
            settings: merged,
          },
        },
      };
    }

    case 'UPDATE_GLOBAL_SETTINGS': {
      return { ...state, globalSettings: { ...state.globalSettings, ...action.settings } };
    }

    case 'SET_SERVER_STATUS':
      return { ...state, serverStatus: action.status };

    case 'SET_SERVER_SESSION_ID':
      return { ...state, serverSessionId: action.serverSessionId };

    case 'SET_SERVER_ID':
      return { ...state, serverId: action.serverId };

    case 'QUEUE_STATUS_UPDATE':
      return { ...state, queueStatus: action.queueStatus };

    case 'SYNC_FROM_SERVER': {
      // Server confirmed this session was restored (is_restored: true).
      // The server sends metadata (conversation_length, active_task_id,
      // workspace_path) but NOT actual message content — full history
      // restoration requires a future HISTORY_REQUEST/RESPONSE protocol.
      // For now: keep frontend's in-memory state, update workspace_path,
      // and add a system message noting the reconnection.
      const syncData = action.data || {};
      const sid = syncData.session_id || state.activeSessionId;
      const sess = state.sessions[sid];
      if (!sess) {
        return { ...state, serverStatus: 'connected' };
      }
      const updates = { workspacePath: syncData.workspace_path || sess.workspacePath };
      const reconnectMsg = {
        role: 'system',
        content: `Reconnected to server. Session restored (${syncData.conversation_length || 0} messages in server history).`,
      };
      return {
        ...state,
        serverStatus: 'connected',
        sessions: {
          ...state.sessions,
          [sid]: {
            ...sess,
            ...updates,
            messages: [...sess.messages, reconnectMsg],
          },
        },
      };
    }

    case 'PENDING_INPUT': {
      const { sessionId, content, inputMode, widget } = action;
      // Don't add assistant message — the streaming section already shows the
      // response text. Adding it again creates a duplicate raw-text box.
      return {
        ...state,
        pendingInput: { content, inputMode, widget, sessionId },
      };
    }

    case 'CLEAR_PENDING_INPUT':
      return { ...state, pendingInput: null };

    case 'WIDGET_UPDATE': {
      // Display-only widget — no pending input, just add as a message
      const { sessionId: wuSid, widget: wuWidget } = action;
      const wuSession = state.sessions[wuSid];
      if (!wuSession) return state;
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [wuSid]: {
            ...wuSession,
            messages: [...wuSession.messages, {
              role: 'widget',
              content: JSON.stringify(wuWidget),
              widget: wuWidget,
            }],
          },
        },
      };
    }

    case 'SYNC_SESSION_LIST': {
      // Server is authority — replace client sessions with server's list.
      // Preserve existing messages/taskIds for sessions the client already knows
      // (avoids wiping conversation on WebSocket reconnect).
      const serverSessions = action.sessions || [];
      const newSessions = {};
      let defaultId = null;
      for (const s of serverSessions) {
        if (s.status === 'closed') continue;
        const existing = state.sessions[s.session_id];
        newSessions[s.session_id] = {
          id: s.session_id,
          label: s.session_id,
          createdAt: (s.created_at || 0) * 1000,
          workspacePath: existing?.workspacePath || '',
          messages: existing?.messages || [],
          taskIds: existing?.taskIds || [],
          settings: existing?.settings || loadSessionSettings(s.session_id),
          status: 'ready',
        };
        if (!defaultId) defaultId = s.session_id;
      }
      // If no server sessions, keep current state
      if (Object.keys(newSessions).length === 0) return state;
      // Keep active session if it's still in the server list
      const activeId = newSessions[state.activeSessionId] ? state.activeSessionId : defaultId;
      return {
        ...state,
        sessions: newSessions,
        activeSessionId: activeId,
        activeTabId: newSessions[state.activeTabId] ? state.activeTabId : activeId,
        activeTabType: newSessions[state.activeTabId] ? state.activeTabType : 'session',
      };
    }

    case 'RESTORE_MESSAGES': {
      // Populate a session's messages from server file store
      const rmSession = state.sessions[action.sessionId];
      if (!rmSession) return state;
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [action.sessionId]: {
            ...rmSession,
            messages: (action.messages || []).map(m => ({
              role: m.role,
              content: m.content,
            })),
          },
        },
      };
    }

    case 'CLEAR_ALL': {
      const session = createSession('Session 1');
      return {
        ...initialState,
        sessions: { [session.id]: session },
        activeSessionId: session.id,
        activeTabId: session.id,
        activeTabType: 'session',
      };
    }

    default:
      return state;
  }
}

// ── Hook ───────────────────────────────────────────────────────────

export function useSessionManager() {
  // Create initial session
  const [initialized] = useState(() => {
    const s = createSession('Session 1');
    return s;
  });

  const [state, dispatch] = useReducer(sessionReducer, {
    ...initialState,
    sessions: { [initialized.id]: initialized },
    activeSessionId: initialized.id,
    activeTabId: initialized.id,
    activeTabType: 'session',
  });

  // Streaming refs: Map<targetId, StreamState>
  // StreamState = { agentContents: {}, agentOrder: [], sections: [], isStreaming: false }
  const streamingRefs = useRef(new Map());
  const [streamingSections, setStreamingSections] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const getStreamState = useCallback((targetId) => {
    if (!streamingRefs.current.has(targetId)) {
      streamingRefs.current.set(targetId, {
        agentContents: {},
        agentOrder: [],
        sections: [],
        isStreaming: false,
      });
    }
    return streamingRefs.current.get(targetId);
  }, []);

  // Remap a streaming ref key (used during task ID reconciliation)
  const remapStreamingRef = useCallback((oldId, newId) => {
    const existing = streamingRefs.current.get(oldId);
    if (existing) {
      streamingRefs.current.set(newId, existing);
      streamingRefs.current.delete(oldId);
    }
  }, []);

  // Get the active target ID for routing
  const getActiveTargetId = useCallback(() => {
    return state.activeTabId || state.activeSessionId;
  }, [state.activeTabId, state.activeSessionId]);

  const getStreamTargetId = useCallback(() => {
    const target = state.currentStreamTarget;
    if (!target) return null;
    return target.type === 'task' ? target.taskId : target.sessionId;
  }, [state.currentStreamTarget]);

  // Handle streaming token — updates refs, NOT reducer
  const handleStreamToken = useCallback((data) => {
    // If token has a task_id, route to the task's streaming refs
    // (background task tokens) instead of the conversation's target.
    const taskId = data.task_id;
    const targetId = taskId || getStreamTargetId();
    if (!targetId) return;

    const ss = getStreamState(targetId);
    const agentId = data.metadata?.agent_id || 'agent';

    if (!ss.agentContents[agentId]) {
      ss.agentContents[agentId] = '';
      ss.agentOrder.push(agentId);
    }
    ss.agentContents[agentId] += data.content;
    ss.isStreaming = true;

    // Update the task's currentAgent so the status chip reflects the active agent
    if (taskId) {
      dispatch({ type: 'UPDATE_TASK_AGENT', taskId, agentId });
    } else {
      const target = state.currentStreamTarget;
      if (target?.type === 'task' && target.taskId) {
        dispatch({ type: 'UPDATE_TASK_AGENT', taskId: target.taskId, agentId });
      }
    }

    ss.sections = ss.agentOrder.map(id => {
      const raw = ss.agentContents[id];
      const parsed = parseResponseTags(raw);
      return {
        agentId: id,
        content: raw,
        metadata: id === agentId ? data.metadata : { agent_id: id },
        isComplete: id !== agentId,
        thinkingContent: parsed.thinkingContent,
        responseContent: parsed.responseContent,
        responsePhase: parsed.phase,
      };
    });

    // Only update rendered sections if this target is active
    if (targetId === getActiveTargetId()) {
      setStreamingSections([...ss.sections]);
      setIsStreaming(true);
    }
  }, [getStreamTargetId, getActiveTargetId, getStreamState, state.currentStreamTarget, dispatch]);

  // Handle stream start (message_start) — resets refs for target
  const handleStreamStart = useCallback(() => {
    const targetId = getStreamTargetId();
    if (!targetId) return;

    const ss = getStreamState(targetId);
    ss.agentContents = {};
    ss.agentOrder = [];
    ss.sections = [];
    ss.isStreaming = true;

    if (targetId === getActiveTargetId()) {
      setStreamingSections([]);
      setIsStreaming(true);
    }
  }, [getStreamTargetId, getActiveTargetId, getStreamState]);

  // Handle stream end — finalize sections, mark complete
  const handleStreamEnd = useCallback((data) => {
    const targetId = getStreamTargetId() ||
      (data.task_id && state.tasks[data.task_id] ? data.task_id : state.activeSessionId);
    if (!targetId) return;

    const ss = getStreamState(targetId);
    ss.isStreaming = false;
    ss.sections = ss.agentOrder.map(id => {
      const raw = ss.agentContents[id] || '';
      const parsed = parseResponseTags(raw);
      // If stream ended without <Response>, fall back to normal display
      const phase = parsed.phase === 'pre_response' ? 'no_tags' : parsed.phase;
      return {
        agentId: id,
        content: raw,
        metadata: { agent_id: id },
        isComplete: true,
        thinkingContent: parsed.thinkingContent,
        responseContent: parsed.responseContent,
        responsePhase: phase,
      };
    });

    if (targetId === getActiveTargetId()) {
      setStreamingSections([...ss.sections]);
      setIsStreaming(false);
    }
  }, [getStreamTargetId, getActiveTargetId, getStreamState, state.tasks, state.activeSessionId]);

  // Switch active tab — swap displayed streaming sections
  const switchTab = useCallback((tabId, tabType) => {
    dispatch({ type: 'SWITCH_TAB', tabId, tabType });
    const ss = streamingRefs.current.get(tabId);
    if (ss) {
      setStreamingSections([...ss.sections]);
      setIsStreaming(ss.isStreaming);
    } else {
      setStreamingSections([]);
      setIsStreaming(false);
    }
  }, []);

  // Clear displayed streaming sections (used when creating a new session
  // so old session content doesn't bleed into the new view)
  const clearStreamingDisplay = useCallback(() => {
    setStreamingSections([]);
    setIsStreaming(false);
  }, []);

  // localStorage persistence for session IDs
  const persistSessionIds = useCallback(() => {
    const ids = Object.keys(state.sessions);
    try {
      localStorage.setItem('rankevolve_session_ids', JSON.stringify(ids));
    } catch {
      // localStorage may not be available
    }
  }, [state.sessions]);

  return {
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
  };
}

export default useSessionManager;
