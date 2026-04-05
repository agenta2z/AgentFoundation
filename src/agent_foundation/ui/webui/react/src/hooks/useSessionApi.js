/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * useSessionApi — HTTP API hook for reading session data from the server's file store.
 *
 * The server writes session state to disk; these endpoints read it via the
 * WebUI backend's REST API. This avoids sending heavy data through the queue.
 */

import { useCallback } from 'react';

export function useSessionApi() {
  const fetchSessionState = useCallback(async (sessionId) => {
    try {
      const resp = await fetch(`/api/sessions/${sessionId}/state`);
      if (!resp.ok) return null;
      return resp.json();
    } catch {
      return null;
    }
  }, []);

  const fetchSessionList = useCallback(async () => {
    try {
      const resp = await fetch('/api/sessions');
      if (!resp.ok) return [];
      return resp.json();
    } catch {
      return [];
    }
  }, []);

  const fetchSessionMessages = useCallback(async (sessionId) => {
    try {
      const resp = await fetch(`/api/sessions/${sessionId}/messages`);
      if (!resp.ok) return null;
      const data = await resp.json();
      return data.messages || [];
    } catch {
      return null;
    }
  }, []);

  const fetchWelcomeMessage = useCallback(async () => {
    try {
      const resp = await fetch('/api/welcome-message');
      if (!resp.ok) return null;
      return resp.json();
    } catch {
      return null;
    }
  }, []);

  const updateWelcomeMessage = useCallback(async (content) => {
    try {
      const resp = await fetch('/api/welcome-message', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
      });
      if (!resp.ok) return null;
      return resp.json();
    } catch {
      return null;
    }
  }, []);

  const resetWelcomeMessage = useCallback(async () => {
    try {
      const resp = await fetch('/api/welcome-message', { method: 'DELETE' });
      if (!resp.ok) return null;
      return resp.json();
    } catch {
      return null;
    }
  }, []);

  const fetchTurnData = useCallback(async (sessionId, turnNumber) => {
    try {
      const resp = await fetch(`/api/sessions/${sessionId}/turns/${turnNumber}`);
      if (!resp.ok) {
        console.error(`fetchTurnData: ${resp.status} ${resp.statusText} for session=${sessionId} turn=${turnNumber}`);
        return null;
      }
      return resp.json();
    } catch (err) {
      console.error(`fetchTurnData: network error for session=${sessionId} turn=${turnNumber}:`, err);
      return null;
    }
  }, []);

  return {
    fetchSessionState,
    fetchSessionList,
    fetchSessionMessages,
    fetchWelcomeMessage,
    updateWelcomeMessage,
    resetWelcomeMessage,
    fetchTurnData,
  };
}

export default useSessionApi;
