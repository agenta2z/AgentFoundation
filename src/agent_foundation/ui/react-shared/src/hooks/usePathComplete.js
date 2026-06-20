/**
 * usePathComplete — debounced, cancellable path-autocomplete query state.
 *
 * The host injects a provider (it owns the filesystem / network); shared-ui never
 * touches the filesystem directly. See conditional_path_inputs_plan.md §D7.
 *
 * Provider signature:
 *   (request:{prefix, partial, dirsOnly?, limit?}) => Promise<Array<{name, path, is_dir}>>
 *
 * Usage:
 *   const { suggestions, loading, error, query, setQuery } =
 *     usePathComplete({ prefix, provider, dirsOnly, limit });
 *
 * Behavior:
 *   - Debounces queries (~200ms).
 *   - Cancels in-flight requests via AbortController when the query changes/unmounts.
 *   - Never throws: provider failures map to a soft `error` and empty suggestions.
 *   - No provider or no prefix → no fetch, suggestions stay [].
 */

import { useState, useEffect, useRef } from 'react';

const DEBOUNCE_MS = 200;

export function usePathComplete({ prefix, provider, dirsOnly = false, limit = 50 } = {}) {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const abortRef = useRef(null);

  useEffect(() => {
    // Nothing to fetch against — degrade silently.
    if (typeof provider !== 'function' || !prefix) {
      setSuggestions([]);
      setLoading(false);
      setError(null);
      return undefined;
    }

    let cancelled = false;
    const handle = setTimeout(() => {
      // Cancel any prior in-flight request.
      if (abortRef.current) abortRef.current.abort();
      const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
      abortRef.current = controller;

      setLoading(true);
      setError(null);
      Promise.resolve()
        .then(() => provider({ prefix, partial: query, dirsOnly, limit, signal: controller?.signal }))
        .then((results) => {
          if (cancelled) return;
          setSuggestions(Array.isArray(results) ? results : []);
        })
        .catch((err) => {
          if (cancelled || err?.name === 'AbortError') return;
          // Soft failure: surface the error, keep suggestions empty, never throw.
          setError(err);
          setSuggestions([]);
        })
        .finally(() => {
          if (!cancelled) setLoading(false);
        });
    }, DEBOUNCE_MS);

    return () => {
      cancelled = true;
      clearTimeout(handle);
      if (abortRef.current) abortRef.current.abort();
    };
  }, [prefix, provider, dirsOnly, limit, query]);

  return { suggestions, loading, error, query, setQuery };
}

export default usePathComplete;
