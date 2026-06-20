/**
 * Vitest global setup.
 *
 * jsdom ships a partial `localStorage` stub (no working `.clear()`), which breaks the
 * theme-persistence property tests. Install a complete in-memory Storage so DOM-dependent
 * tests have a faithful Web Storage API. Test-only — never bundled into the library.
 */

class MemoryStorage {
  constructor() {
    this._data = new Map();
  }
  get length() {
    return this._data.size;
  }
  clear() {
    this._data.clear();
  }
  getItem(key) {
    return this._data.has(String(key)) ? this._data.get(String(key)) : null;
  }
  setItem(key, value) {
    this._data.set(String(key), String(value));
  }
  removeItem(key) {
    this._data.delete(String(key));
  }
  key(index) {
    return Array.from(this._data.keys())[index] ?? null;
  }
}

function installStorage(target) {
  if (!target) return;
  const storage = new MemoryStorage();
  Object.defineProperty(target, 'localStorage', { value: storage, configurable: true, writable: true });
  Object.defineProperty(target, 'sessionStorage', { value: new MemoryStorage(), configurable: true, writable: true });
}

installStorage(globalThis);
if (typeof window !== 'undefined') installStorage(window);
