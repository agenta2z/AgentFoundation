/**
 * (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
 *
 * useWorkspace — hook for browsing workspace files via the REST API.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

const API_BASE = '/api/workspace';

export function useWorkspace(workspacePath, isTaskRunning = false) {
  const [tree, setTree] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const intervalRef = useRef(null);

  const fetchTree = useCallback(async () => {
    if (!workspacePath) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `${API_BASE}/tree?workspace=${encodeURIComponent(workspacePath)}`
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setTree(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [workspacePath]);

  const fetchFile = useCallback(
    async (relativePath) => {
      if (!workspacePath) return null;
      try {
        const res = await fetch(
          `${API_BASE}/file?workspace=${encodeURIComponent(workspacePath)}&path=${encodeURIComponent(relativePath)}`
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [workspacePath]
  );

  const fetchOutputs = useCallback(async () => {
    if (!workspacePath) return [];
    try {
      const res = await fetch(
        `${API_BASE}/outputs?workspace=${encodeURIComponent(workspacePath)}`
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      return data.outputs || [];
    } catch (err) {
      setError(err.message);
      return [];
    }
  }, [workspacePath]);

  const fetchResults = useCallback(async () => {
    if (!workspacePath) return [];
    try {
      const res = await fetch(
        `${API_BASE}/results?workspace=${encodeURIComponent(workspacePath)}`
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      return data.results || [];
    } catch (err) {
      setError(err.message);
      return [];
    }
  }, [workspacePath]);

  // Auto-refresh tree while task is running
  useEffect(() => {
    if (workspacePath) {
      fetchTree();
    }
    if (isTaskRunning && workspacePath) {
      intervalRef.current = setInterval(fetchTree, 5000);
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [workspacePath, isTaskRunning, fetchTree]);

  return { tree, loading, error, fetchTree, fetchFile, fetchOutputs, fetchResults };
}
