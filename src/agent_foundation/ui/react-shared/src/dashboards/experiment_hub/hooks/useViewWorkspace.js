/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * useViewWorkspace — read/write view-workspace files via REST. Each view in
 * a panel has its own subdirectory for persistent artifacts.
 *
 * Ported verbatim from RankEvolve (hooks/useViewWorkspace.js); the only
 * change is the `API_BASE` import path (the shared `utils/api` module).
 */

import { useState, useCallback, useRef } from 'react';
import { API_BASE } from '../../../utils/api';

export function useViewWorkspace(panelId, viewId) {
  const [loading, setLoading] = useState(false);
  const cacheRef = useRef({});

  const readJSON = useCallback(async (filename) => {
    if (!panelId || !viewId) return null;
    const cacheKey = `${panelId}/${viewId}/${filename}`;
    if (cacheRef.current[cacheKey]) return cacheRef.current[cacheKey];

    setLoading(true);
    try {
      const response = await fetch(
        `${API_BASE}/panels/${panelId}/views/${viewId}/${filename}`
      );
      if (!response.ok) return null;
      const data = await response.json();
      cacheRef.current[cacheKey] = data;
      return data;
    } catch (err) {
      console.error(`useViewWorkspace: failed to read ${cacheKey}:`, err);
      return null;
    } finally {
      setLoading(false);
    }
  }, [panelId, viewId]);

  const writeJSON = useCallback(async (filename, data) => {
    if (!panelId || !viewId) return false;
    const cacheKey = `${panelId}/${viewId}/${filename}`;

    try {
      const response = await fetch(
        `${API_BASE}/panels/${panelId}/views/${viewId}/${filename}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        }
      );
      if (response.ok) {
        cacheRef.current[cacheKey] = data;
        return true;
      }
      return false;
    } catch (err) {
      console.error(`useViewWorkspace: failed to write ${cacheKey}:`, err);
      return false;
    }
  }, [panelId, viewId]);

  const invalidateCache = useCallback((filename) => {
    if (filename) {
      delete cacheRef.current[`${panelId}/${viewId}/${filename}`];
    } else {
      Object.keys(cacheRef.current).forEach(key => {
        if (key.startsWith(`${panelId}/${viewId}/`)) {
          delete cacheRef.current[key];
        }
      });
    }
  }, [panelId, viewId]);

  return { readJSON, writeJSON, invalidateCache, loading };
}

export default useViewWorkspace;
