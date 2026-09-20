/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * comboPendingPredicate — pure helpers mirroring the server-side
 * `_derive_apply_state` + `combo_flag_bindings`. The UI uses
 * `combo.applyState` directly (server-derived) for chip rendering; these
 * helpers compute eligibility for Auto Mode and the Apply-changes drawer.
 *
 * Ported verbatim from RankEvolve (utils/comboPendingPredicate.js).
 */

// Strict regex for a scoped gin binding: `<configurable>.<param>`.
const SCOPED_BINDING_RE = /^[A-Za-z_][\w]*\.[A-Za-z_][\w]*$/;

/**
 * Return the list of Hks in `combo.selectedItems` that are NOT
 * implemented per the supplied `implMap`.
 */
export function comboGatingHypotheses(combo, implMap) {
  if (!combo || !Array.isArray(combo.selectedItems)) return [];
  const map = implMap || {};
  const out = [];
  for (const h of combo.selectedItems) {
    if (typeof h !== 'string' || !h) continue;
    const entry = map[h];
    if (!entry || entry.implemented !== true) {
      out.push(h);
    }
  }
  return out;
}

/**
 * Mirror of services/combo_capability.combo_flag_bindings.
 * Returns { allScoped, scopedBindings, bareOrMissing } for a combo.
 */
export function comboFlagBindings(combo, flagMap) {
  const items = Array.isArray(combo?.selectedItems) ? combo.selectedItems : [];
  const map = flagMap || {};
  const scopedBindings = [];
  const bareOrMissing = [];
  for (const h of items) {
    if (typeof h !== 'string' || !h) continue;
    const value = map[h];
    if (typeof value !== 'string' || !value || !SCOPED_BINDING_RE.test(value)) {
      bareOrMissing.push(h);
    } else {
      scopedBindings.push(value);
    }
  }
  return { allScoped: bareOrMissing.length === 0, scopedBindings, bareOrMissing };
}

/**
 * Four-state apply-eligibility mirror of server-side _derive_apply_state.
 * Returns { eligible, applyState, gatingHypotheses, configReady, ... }.
 */
export function isComboApplyEligible(combo, implMap, hubCapability) {
  const gating = comboGatingHypotheses(combo, implMap);
  const configReady = (combo?.configStatus || '') === 'ready';

  if (gating.length > 0) {
    return {
      eligible: false,
      applyState: 'pending_implementation',
      gatingHypotheses: gating,
      configReady,
    };
  }
  if (hubCapability?.flagComposes) {
    const { allScoped, scopedBindings, bareOrMissing } = comboFlagBindings(
      combo, hubCapability.flagMap || {}
    );
    if (allScoped) {
      return {
        eligible: true,
        applyState: 'ready',
        readyVia: 'flag_composition',
        resolvedBindings: scopedBindings,
        gatingHypotheses: [],
        configReady,
      };
    }
    return {
      eligible: false,
      applyState: 'pending_flag_map',
      bareBindings: bareOrMissing,
      gatingHypotheses: [],
      configReady,
    };
  }
  // Hub is Model B (legacy per-combo gin file).
  if (configReady) {
    return {
      eligible: true,
      applyState: 'ready',
      readyVia: 'gin_config',
      gatingHypotheses: [],
      configReady,
    };
  }
  return {
    eligible: false,
    applyState: 'pending_config',
    gatingHypotheses: [],
    configReady,
  };
}

/**
 * Split a list of combos into eligible vs pending buckets.
 */
export function partitionCombos(combos, implMap, hubCapability) {
  const eligible = [];
  const pending = [];
  for (const c of combos || []) {
    const { eligible: ok } = isComboApplyEligible(c, implMap, hubCapability);
    (ok ? eligible : pending).push(c);
  }
  return { eligible, pending };
}

export default isComboApplyEligible;
