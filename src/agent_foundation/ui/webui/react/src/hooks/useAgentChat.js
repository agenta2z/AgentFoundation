/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * useAgentChat — WebSocket-based hook for real agent mode.
 *
 * Manages WebSocket connection to /ws/agent, message streaming,
 * command responses, and auto-reconnect with exponential backoff.
 */

import { useState, useRef, useCallback, useEffect } from 'react';

const WS_RECONNECT_BASE_MS = 1000;
const WS_RECONNECT_MAX_MS = 30000;

function getWsUrl() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws/agent`;
}

export function useAgentChat() {
  const [messages, setMessages] = useState([]);
  const [streamingSections, setStreamingSections] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [config, setConfig] = useState({ model: '', target_path: '', provider: '' });
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [taskPhase, setTaskPhase] = useState(null);

  const wsRef = useRef(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimerRef = useRef(null);
  const connectRef = useRef(null);
  const agentContentsRef = useRef({});
  const agentOrderRef = useRef([]);

  const scheduleReconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
    }
    const delay = Math.min(
      WS_RECONNECT_BASE_MS * Math.pow(2, reconnectAttemptRef.current),
      WS_RECONNECT_MAX_MS
    );
    reconnectAttemptRef.current += 1;
    reconnectTimerRef.current = setTimeout(() => {
      if (connectRef.current) connectRef.current();
    }, delay);
  }, []);

  const handleServerMessage = useCallback((data) => {
    switch (data.type) {
      case 'token': {
        const agentId = data.metadata?.agent_id || 'agent';

        if (!agentContentsRef.current[agentId]) {
          agentContentsRef.current[agentId] = '';
          agentOrderRef.current.push(agentId);
        }
        agentContentsRef.current[agentId] += data.content;

        const sections = agentOrderRef.current.map(id => ({
          agentId: id,
          content: agentContentsRef.current[id],
          metadata: id === agentId ? data.metadata : { agent_id: id },
          isComplete: id !== agentId,
        }));

        setStreamingSections(sections);
        setIsStreaming(true);
        break;
      }

      case 'message_start':
        agentContentsRef.current = {};
        agentOrderRef.current = [];
        setStreamingSections([]);
        setIsStreaming(true);
        break;

      case 'message_end': {
        // Mark all sections as complete instead of clearing them
        const completedSections = agentOrderRef.current.map(id => ({
          agentId: id,
          content: agentContentsRef.current[id] || '',
          metadata: { agent_id: id },
          isComplete: true,
        }));
        setStreamingSections(completedSections);

        // Store combined content in messages as hidden (for conversation history)
        const finalContent = agentOrderRef.current
          .map(id => {
            const label = id === 'base' ? '🔵 Base Agent'
                        : id === 'review' ? '🟣 Review Agent'
                        : id === 'system' ? '⚙️ System'
                        : `🤖 ${id}`;
            return `### ${label}\n\n${agentContentsRef.current[id] || ''}`;
          })
          .join('\n\n---\n\n');

        setMessages(prev => [...prev, {
          role: 'assistant',
          content: data.final_content || finalContent,
          message_id: data.message_id,
          hidden: true,
        }]);
        // Keep agentContentsRef and agentOrderRef for View All drawer
        setIsStreaming(false);
        break;
      }

      case 'command_response':
        setMessages(prev => [...prev, {
          role: 'system',
          content: data.content,
        }]);
        break;

      case 'config_update':
        if (data.config) {
          setConfig(prev => ({ ...prev, ...data.config }));
        }
        break;

      case 'task_status':
        setTaskPhase({
          phase: data.phase,
          state: data.state,
        });
        if (data.state === 'complete') {
          setTimeout(() => setTaskPhase(null), 3000);
        }
        break;

      case 'status':
        if (data.status === 'complete' || data.status === 'error') {
          setIsStreaming(false);
        }
        break;

      case 'error':
        setMessages(prev => [...prev, {
          role: 'error',
          content: data.message,
        }]);
        setIsStreaming(false);
        setStreamingSections([]);
        break;

      case 'heartbeat':
      case 'pong':
        break;

      default:
        console.log('Unknown message type:', data.type);
    }
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return;
    }

    const url = getWsUrl();
    setConnectionStatus('connecting');

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionStatus('connected');
      reconnectAttemptRef.current = 0;
    };

    ws.onclose = () => {
      setConnectionStatus('disconnected');
      wsRef.current = null;
      scheduleReconnect();
    };

    ws.onerror = () => {
      setConnectionStatus('error');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleServerMessage(data);
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };
  }, [scheduleReconnect, handleServerMessage]);

  connectRef.current = connect;

  const sendMessage = useCallback((text) => {
    if (!text.trim()) return;
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.error('WebSocket not connected');
      return;
    }

    setMessages(prev => [...prev, { role: 'user', content: text }]);

    wsRef.current.send(JSON.stringify({
      type: 'message',
      content: text,
    }));
  }, []);

  const cancelRequest = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'cancel' }));
    }
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setStreamingSections([]);
    setIsStreaming(false);
    setTaskPhase(null);
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  return {
    messages,
    streamingSections,
    isStreaming,
    config,
    connectionStatus,
    taskPhase,
    sendMessage,
    cancelRequest,
    clearMessages,
    isConnected: connectionStatus === 'connected',
  };
}

export default useAgentChat;
