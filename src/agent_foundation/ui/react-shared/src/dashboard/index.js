/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * @agent-foundation/shared-ui — generic Dashboard framework barrel.
 * A reusable, context-free, manifest-driven multi-view dashboard surface that
 * any AF app can host (the Experiment Hub is its first concrete instance).
 */

export { registerView, getView, listRegisteredViews, unregisterView } from './ViewRegistry';
export {
  registerDashboard, getDashboard, listRegisteredDashboards, createDashboardReducer,
} from './DashboardRegistry';
export { registerManifest, getManifest, normalizeManifest } from './manifest';
export {
  evaluateViewRules, getPipelineStages, getDashboardStatus, deriveStatusMap,
} from './lifecycle';
export { default as PipelineStatusBar } from './PipelineStatusBar';
export { default as WidgetHostView } from './WidgetHostView';
export { default as DashboardPanel } from './DashboardPanel';
