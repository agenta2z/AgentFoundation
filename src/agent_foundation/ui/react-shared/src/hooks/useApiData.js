/**
 * useApiData — generic data-fetching hook.
 *
 * Usage:
 *   const { data, loading, error, refetch } = useApiData('/projects');
 *   const { data: employee } = useApiData(`/employees/${id}`);
 *   const { data } = useApiData(null); // skip fetch when endpoint is null
 */

import { useState, useEffect, useCallback } from 'react';
import { fetchJson } from '../utils/api';

export function useApiData(endpoint) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(!!endpoint);
  const [error, setError] = useState(null);

  const doFetch = useCallback(async () => {
    if (!endpoint) return;
    setLoading(true);
    setError(null);
    try {
      const result = await fetchJson(endpoint);
      setData(result);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, [endpoint]);

  useEffect(() => {
    doFetch();
  }, [doFetch]);

  return { data, loading, error, refetch: doFetch };
}

export default useApiData;
