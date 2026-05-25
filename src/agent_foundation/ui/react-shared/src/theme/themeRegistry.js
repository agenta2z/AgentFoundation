/**
 * Theme Registry
 *
 * Stores all available ThemeDefinitions and provides lookup, listing,
 * registration, and deep-merge APIs. Supports theme inheritance via
 * the `extends` field with circular-reference detection.
 */

import dark from './themes/dark.js';
import atlassian from './themes/atlassian.js';
import pinterest from './themes/pinterest.js';

const registry = new Map();

/**
 * Deep-merge two plain objects. Arrays are replaced wholesale;
 * nested objects are merged recursively; all other values are
 * overwritten by the override.
 *
 * @param {Object} base
 * @param {Object} overrides
 * @returns {Object} A new merged object (neither input is mutated).
 */
export function mergeTheme(base, overrides) {
  if (overrides === null || overrides === undefined) return base;
  if (base === null || base === undefined) return overrides;

  const result = {};

  // Copy all base keys first
  for (const key of Object.keys(base)) {
    result[key] = base[key];
  }

  // Apply overrides
  for (const key of Object.keys(overrides)) {
    const baseVal = base[key];
    const overVal = overrides[key];

    if (Array.isArray(overVal)) {
      // Arrays replaced wholesale
      result[key] = [...overVal];
    } else if (
      overVal !== null &&
      typeof overVal === 'object' &&
      !Array.isArray(overVal) &&
      baseVal !== null &&
      typeof baseVal === 'object' &&
      !Array.isArray(baseVal)
    ) {
      // Both are plain objects — recurse
      result[key] = mergeTheme(baseVal, overVal);
    } else {
      result[key] = overVal;
    }
  }

  return result;
}

/**
 * Resolve a theme by ID, recursively resolving `extends` chains.
 * Maintains a visited set to detect circular inheritance.
 *
 * @param {string} id
 * @param {Set<string>} [visited] - Internal: IDs already seen in this chain.
 * @returns {Object} Fully resolved ThemeDefinition.
 */
function resolveTheme(id, visited = new Set()) {
  const definition = registry.get(id);
  if (!definition) {
    // Unknown ID — fall back to dark (but only if we aren't already resolving dark)
    if (id === 'dark') {
      throw new Error('Default theme "dark" is not registered.');
    }
    return resolveTheme('dark', visited);
  }

  if (!definition.extends) {
    return definition;
  }

  // Circular-extends detection
  if (visited.has(id)) {
    throw new Error(
      `Circular theme extends chain detected: ${[...visited, id].join(' → ')}`
    );
  }
  visited.add(id);

  const base = resolveTheme(definition.extends, visited);
  // Deep-merge base with this definition's overrides (excluding `extends`)
  const { extends: _ext, ...overrides } = definition;
  return mergeTheme(base, overrides);
}

/**
 * Get a fully-resolved ThemeDefinition by ID.
 * Falls back to "dark" if the ID is not registered.
 *
 * @param {string} id
 * @returns {Object} ThemeDefinition
 */
export function getTheme(id) {
  return resolveTheme(id);
}

/**
 * List all registered theme IDs.
 *
 * @returns {string[]}
 */
export function listThemes() {
  return [...registry.keys()];
}

/**
 * Register (or replace) a theme in the registry.
 *
 * @param {string} id
 * @param {Object} definition - ThemeDefinition object.
 */
export function registerTheme(id, definition) {
  registry.set(id, definition);
}

// ---------------------------------------------------------------------------
// Pre-register built-in themes
// ---------------------------------------------------------------------------
registerTheme(dark.id, dark);
registerTheme(atlassian.id, atlassian);
registerTheme(pinterest.id, pinterest);
