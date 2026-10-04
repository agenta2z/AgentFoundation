/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * DashboardRegistry — registers a concrete dashboard (e.g. the Experiment Hub)
 * as `{ manifest?, reducer, initialState? }`. `DashboardPanel` looks a dashboard
 * up by id when it is not given an explicit reducer/manifest via props.
 *
 *   registerDashboard('experiment_hub', { reducer: hubReducer, initialState });
 *
 * Mirrors the `protocol/WidgetRegistry` API. The dashboard's *views* register
 * separately via `registerView` (see ViewRegistry); a dashboard's tab manifest
 * is normally delivered at runtime in the `dashboard_open` WS payload (Python is
 * canonical) — a `manifest` here is only a front-end-only default/fallback.
 */

const _dashboardRegistry = new Map();

export function registerDashboard(id, def, { override = false } = {}) {
  if (!id || typeof id !== 'string') {
    throw new TypeError('registerDashboard(id, def): id must be a non-empty string');
  }
  if (!def || typeof def.reducer !== 'function') {
    throw new TypeError(`registerDashboard("${id}"): def.reducer must be a function`);
  }
  if (_dashboardRegistry.has(id) && !override) {
    throw new Error(`registerDashboard: "${id}" already registered. Pass {override:true} to replace.`);
  }
  _dashboardRegistry.set(id, def);
}

export function getDashboard(id) {
  return _dashboardRegistry.get(id) || null;
}

export function listRegisteredDashboards() {
  return Array.from(_dashboardRegistry.keys());
}

/**
 * Wrap a domain reducer with the built-in view-state actions every dashboard
 * shares, so concrete dashboards never re-implement tab unlock/activate:
 *   SET_ACTIVE_VIEW {viewIndex} · UNLOCK_VIEW {viewIndex} · HYDRATE {state}
 * Unknown actions fall through to the (optional) domain reducer.
 */
export function createDashboardReducer(domainReducer) {
  return function dashboardReducer(state, action) {
    switch (action && action.type) {
      case 'SET_ACTIVE_VIEW':
        if (state.activeView === action.viewIndex) return state;
        return { ...state, activeView: action.viewIndex };
      case 'UNLOCK_VIEW': {
        const set = new Set(state.unlockedViewIds || [0]);
        if (set.has(action.viewIndex)) return state;
        set.add(action.viewIndex);
        return { ...state, unlockedViewIds: Array.from(set).sort((a, b) => a - b) };
      }
      case 'HYDRATE':
        return { ...state, ...(action.state || {}) };
      default:
        return domainReducer ? domainReducer(state, action) : state;
    }
  };
}

export default { registerDashboard, getDashboard, listRegisteredDashboards, createDashboardReducer };
