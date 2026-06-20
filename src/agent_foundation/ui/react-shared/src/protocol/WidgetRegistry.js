/**
 * Extensible widget registry — getWidget + registerWidget.
 *
 * Built-in widgets are auto-registered via registerBuiltins.js (imported by index.js).
 * Domain widgets use namespace.type naming:
 *   registerWidget('openteam.sprint_progress', SprintProgressWidget);
 */

const CANONICAL_TYPES = [
  'text_input', 'single_choice', 'multiple_choice',
  'dropdown', 'toggle', 'tool_argument_form',
  'confirmation', 'multi_input', 'grouped', 'proposal_selection',
  'path_input', 'default',
];

const _registry = new Map();

export function registerWidget(type, Component, { override = false } = {}) {
  if (!type || typeof type !== 'string') {
    throw new TypeError('registerWidget(type, Component): type must be a non-empty string');
  }
  const isBuiltin = CANONICAL_TYPES.includes(type);
  const isNamespaced = type.includes('.');
  if (!isBuiltin && !isNamespaced) {
    console.warn(`registerWidget: "${type}" is neither built-in nor namespaced; prefer "ns.type".`);
  }
  if (_registry.has(type) && !override) {
    throw new Error(`registerWidget: "${type}" already registered. Pass {override:true} to replace.`);
  }
  _registry.set(type, Component);
}

export function getWidget(type) {
  return _registry.get(type) ?? _registry.get('default');
}

export function unregisterWidget(type) {
  _registry.delete(type);
}

export function listRegisteredWidgets() {
  return Array.from(_registry.keys());
}

export default { registerWidget, getWidget, unregisterWidget, listRegisteredWidgets };
