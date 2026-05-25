/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * useAgentWebSocket — Extracted WebSocket transport for agent communication.
 *
 * Manages WebSocket connection to /ws/agent, auto-reconnect with exponential
 * backoff, send/cancel. Supports multi-session via sessionId parameter passed
 * in the initial handshake message.
 *
 * The hook does NOT send heartbeats — the server sends them via heartbeat_loop().
 * The hook only receives and ignores inbound heartbeat/pong messages.
 */

import { useRef, useCallback, useEffect } from 'react';

const WS_RECONNECT_BASE_MS = 1000;
const WS_RECONNECT_MAX_MS = 30000;

function getWsUrl() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws/agent`;
}

/**
 * @param {string|null} sessionId - Optional session ID for multi-session support
 * @param {function} onMessage - Callback for incoming messages: (data) => void
 * @param {function} onStatusChange - Callback for connection status changes: (status) => void
 * @returns {{ send, cancelRequest, disconnect }}
 */
export function useAgentWebSocket(sessionId, onMessage, onStatusChange) {
  const wsRef = useRef(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimerRef = useRef(null);
  const connectRef = useRef(null);
  const sessionIdRef = useRef(sessionId);
  const onMessageRef = useRef(onMessage);
  const onStatusChangeRef = useRef(onStatusChange);

  // Keep refs up to date
  sessionIdRef.current = sessionId;
  onMessageRef.current = onMessage;
  onStatusChangeRef.current = onStatusChange;

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

  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return;
    }

    const url = getWsUrl();
    onStatusChangeRef.current?.('connecting');

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      // WebSocket transport is open, but server session not yet confirmed.
      // Status stays "connecting" until we receive session_init from server.
      onStatusChangeRef.current?.('connecting');
      reconnectAttemptRef.current = 0;
      // Send session_id for multi-session/resume support
      if (sessionIdRef.current) {
        ws.send(JSON.stringify({
          type: 'init',
          session_id: sessionIdRef.current,
        }));
      }
    };

    ws.onclose = () => {
      onStatusChangeRef.current?.('disconnected');
      wsRef.current = null;
      scheduleReconnect();
    };

    ws.onerror = () => {
      onStatusChangeRef.current?.('error');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // Filter out transport-level messages
        if (data.type === 'heartbeat' || data.type === 'pong') {
          return;
        }
        onMessageRef.current?.(data);
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };
  }, [scheduleReconnect]);

  connectRef.current = connect;

  const send = useCallback((data) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
      return true;
    }
    return false;
  }, []);

  const cancelRequest = useCallback(() => {
    send({ type: 'cancel' });
  }, [send]);

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return { send, cancelRequest, disconnect };
}

export default useAgentWebSocket;
