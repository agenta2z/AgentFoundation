/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * ViewRegistry — maps a dashboard view-type string to its React component plus
 * lifecycle predicates. The generic analogue of `protocol/WidgetRegistry` (for
 * widgets), but for full-panel dashboard views.
 *
 * The TAB STRUCTURE (which views, order, labels, per-view config) is canonical
 * on the Python side — a Dashboard tool's `dashboard_config.view_manifest`,
 * shipped to the client in the `dashboard_open` payload. This registry holds
 * only the JS-only bits a view-type needs: its `component` and the optional
 * `unlockWhen(dashboardState)` / `autoActivateWhen(dashboardState)` predicate
 * functions (which cannot live in JSON).
 *
 * Concrete dashboards register their views on import, e.g.:
 *   registerView('queue_progress', {
 *     component: QueueProgressView,
 *     unlockWhen: (s) => (s.runQueue || []).some(r => r.status !== 'queued'),
 *   });
 */

const _viewRegistry = new Map();

export function registerView(type, def, { override = false } = {}) {
  if (!type || typeof type !== 'string') {
    throw new TypeError('registerView(type, def): type must be a non-empty string');
  }
  if (!def || typeof def.component !== 'function') {
    throw new TypeError(`registerView("${type}"): def.component must be a React component`);
  }
  if (_viewRegistry.has(type) && !override) {
    throw new Error(`registerView: "${type}" already registered. Pass {override:true} to replace.`);
  }
  _viewRegistry.set(type, {
    component: def.component,
    defaultLabel: def.defaultLabel || type,
    unlockWhen: def.unlockWhen || (() => true),
    autoActivateWhen: def.autoActivateWhen || (() => false),
  });
}

export function getView(type) {
  return _viewRegistry.get(type) || null;
}

export function listRegisteredViews() {
  return Array.from(_viewRegistry.keys());
}

export function unregisterView(type) {
  _viewRegistry.delete(type);
}

export default { registerView, getView, listRegisteredViews, unregisterView };
