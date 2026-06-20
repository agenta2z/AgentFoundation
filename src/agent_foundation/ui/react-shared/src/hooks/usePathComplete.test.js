import { describe, it, expect, vi, afterEach } from 'vitest';
import { renderHook, waitFor, act, cleanup } from '@testing-library/react';

import usePathComplete from './usePathComplete';

afterEach(() => cleanup());

describe('usePathComplete', () => {
  it('does not fetch when no provider is given', async () => {
    const { result } = renderHook(() => usePathComplete({ prefix: '/root' }));
    act(() => result.current.setQuery('abc'));
    // Give the debounce a chance to (not) fire.
    await new Promise((r) => setTimeout(r, 250));
    expect(result.current.suggestions).toEqual([]);
    expect(result.current.loading).toBe(false);
  });

  it('does not fetch when no prefix is given', async () => {
    const provider = vi.fn().mockResolvedValue([{ name: 'x', path: '/x', is_dir: false }]);
    const { result } = renderHook(() => usePathComplete({ provider }));
    act(() => result.current.setQuery('abc'));
    await new Promise((r) => setTimeout(r, 250));
    expect(provider).not.toHaveBeenCalled();
    expect(result.current.suggestions).toEqual([]);
  });

  it('debounces, calls the provider with prefix/partial, and returns suggestions', async () => {
    const provider = vi.fn().mockResolvedValue([
      { name: 'features', path: '/root/data/features', is_dir: true },
    ]);
    const { result } = renderHook(() => usePathComplete({ prefix: '/root', provider }));

    act(() => result.current.setQuery('data'));

    await waitFor(() => expect(provider).toHaveBeenCalled());
    const call = provider.mock.calls[0][0];
    expect(call.prefix).toBe('/root');
    expect(call.partial).toBe('data');

    await waitFor(() => expect(result.current.suggestions.length).toBe(1));
    expect(result.current.suggestions[0].name).toBe('features');
  });

  it('never throws on provider failure; maps to a soft error with empty suggestions', async () => {
    const provider = vi.fn().mockRejectedValue(new Error('boom'));
    const { result } = renderHook(() => usePathComplete({ prefix: '/root', provider }));

    act(() => result.current.setQuery('x'));

    await waitFor(() => expect(result.current.error).toBeTruthy());
    expect(result.current.suggestions).toEqual([]);
    expect(result.current.loading).toBe(false);
  });
});
