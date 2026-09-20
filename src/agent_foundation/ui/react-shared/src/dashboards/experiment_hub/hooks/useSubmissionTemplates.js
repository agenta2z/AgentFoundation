/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * useSubmissionTemplates — fetches /api/submission-templates once and caches
 * in module-scope memory.
 *
 * Ported verbatim from RankEvolve (hooks/useSubmissionTemplates.js).
 */

import { useEffect, useState } from 'react';

let _cache = null;
let _inflight = null;

async function fetchTemplates() {
  if (_cache) return _cache;
  if (_inflight) return _inflight;
  _inflight = fetch('/api/submission-templates')
    .then((res) => (res.ok ? res.json() : { templates: [] }))
    .then((body) => {
      _cache = Array.isArray(body?.templates) ? body.templates : [];
      return _cache;
    })
    .catch(() => {
      _cache = [];
      return _cache;
    })
    .finally(() => {
      _inflight = null;
    });
  return _inflight;
}

export function useSubmissionTemplates() {
  const [templates, setTemplates] = useState(_cache || []);
  const [loading, setLoading] = useState(_cache === null);

  useEffect(() => {
    let cancelled = false;
    if (_cache !== null) {
      setTemplates(_cache);
      setLoading(false);
      return undefined;
    }
    fetchTemplates().then((list) => {
      if (cancelled) return;
      setTemplates(list || []);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return { templates, loading };
}

export default useSubmissionTemplates;
