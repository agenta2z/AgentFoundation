/**
 * useServerStatus — polls /api/health to track server connection status.
 *
 * Returns:
 *   status     — 'connected' | 'connecting' | 'disconnected'
 *   serverInfo — { status, mode, real_sessions, version } when connected, null otherwise
 *
 * Transitions:
 *   - Initial state: 'connecting'
 *   - On first successful poll: 'connected'
 *   - On 2+ consecutive failures: 'disconnected'
 *   - On recovery after disconnect: 'connected'
 */

import { useState, useEffect, useRef, useCallback } from 'react';

const HEALTH_ENDPOINT = '/api/health';

export function useServerStatus(intervalMs = 5000) {
  const [status, setStatus] = useState('connecting');
  const [serverInfo, setServerInfo] = useState(null);
  const failCountRef = useRef(0);
  const timerRef = useRef(null);

  const checkHealth = useCallback(async () => {
    try {
      const response = await fetch(HEALTH_ENDPOINT);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      failCountRef.current = 0;
      setStatus('connected');
      setServerInfo(data);
    } catch {
      failCountRef.current += 1;
      if (failCountRef.current >= 2) {
        setStatus('disconnected');
        setServerInfo(null);
      }
      // On first failure stay in current state (could be transient)
    }
  }, []);

  useEffect(() => {
    // Initial check immediately
    checkHealth();

    // Start polling
    timerRef.current = setInterval(checkHealth, intervalMs);

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [checkHealth, intervalMs]);

  return { status, serverInfo };
}

export default useServerStatus;
