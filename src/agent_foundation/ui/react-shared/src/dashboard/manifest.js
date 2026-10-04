/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Dashboard manifest types + helpers.
 *
 * A DashboardManifest declares the tab structure of a dashboard:
 *   { id, label, icon, views: [ViewSpec], pipeline: [string] }
 *   ViewSpec = { type, label?, config? }
 *
 * The manifest is CANONICAL on the Python side — a Dashboard tool's
 * `dashboard_config.view_manifest`, delivered to the client in the
 * `dashboard_open` WS payload — which avoids FE/BE drift. The optional JS
 * registry below is only a convenience for front-end-only dashboards / dev
 * defaults; when a `dashboard_open` payload supplies a manifest, that wins.
 *
 * @typedef {{type: string, label?: string, config?: object}} ViewSpec
 * @typedef {{id?: string, label?: string, icon?: string, views: ViewSpec[], pipeline?: string[]}} DashboardManifest
 */

const _manifestRegistry = new Map();

export function registerManifest(id, manifest) {
  _manifestRegistry.set(id, manifest);
}

export function getManifest(id) {
  return _manifestRegistry.get(id) || null;
}

/**
 * Normalize a raw manifest (e.g. from a `dashboard_open` payload, or the JS
 * registry) into a stable `{id,label,icon,views,pipeline}` shape.
 * @param {DashboardManifest|null|undefined} manifest
 * @returns {DashboardManifest}
 */
export function normalizeManifest(manifest) {
  if (!manifest) {
    return { id: null, label: null, icon: null, views: [], pipeline: [] };
  }
  return {
    id: manifest.id || null,
    label: manifest.label || null,
    icon: manifest.icon || null,
    views: Array.isArray(manifest.views) ? manifest.views : [],
    pipeline: Array.isArray(manifest.pipeline) ? manifest.pipeline : [],
  };
}

export default { registerManifest, getManifest, normalizeManifest };
