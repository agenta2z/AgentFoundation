/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * useArchiveCount — fetches the count of committed accumulated-learnings
 * archives for a session. Refreshes when `bumpKey` changes (typically the
 * reducer's `learningsVersion`).
 *
 * Ported verbatim from RankEvolve (hooks/useArchiveCount.js).
 */

import { useEffect, useState } from 'react';

export default function useArchiveCount(sessionId, bumpKey) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!sessionId) {
      setCount(0);
      return undefined;
    }
    let cancelled = false;
    fetch(`/api/sessions/${encodeURIComponent(sessionId)}/learnings/archives`)
      .then((res) => (res.ok ? res.json() : { archives: [] }))
      .then((body) => {
        if (cancelled) return;
        const arr = Array.isArray(body?.archives) ? body.archives : [];
        setCount(arr.length);
      })
      .catch(() => {
        if (!cancelled) setCount(0);
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId, bumpKey]);

  return count;
}
