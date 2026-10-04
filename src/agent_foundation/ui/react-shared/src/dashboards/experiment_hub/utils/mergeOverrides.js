/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * mergeOverrides — pure helper that applies the proposal_overrides.json
 * sidecar to the canonical proposals tree at READ time, without mutating
 * the input. Returns a NEW tree with:
 *   - per-proposal `rank` replaced with override.newRank where applicable
 *   - per-proposal `deprioritized: true` flag for ids in overrides.deprioritize
 *   - per-proposal `_overrideMeta` carrying {oldRank, newRank, rationale}
 *     for UI badges
 *   - per-phase `proposals[]` re-sorted by effective rank ascending
 *
 * Ported verbatim from RankEvolve (utils/mergeOverrides.js). The UI applies
 * overrides at render time so deleting the sidecar reverts ranks.
 */

function _byId(arr) {
  const m = new Map();
  for (const r of arr || []) {
    if (r && r.id) m.set(r.id, r);
  }
  return m;
}

export function mergeOverrides(baseProposals, overrides) {
  if (!baseProposals || typeof baseProposals !== 'object') {
    return baseProposals;
  }
  const rankMap = _byId(overrides?.rankings || []);
  const deprioMap = _byId(overrides?.deprioritize || []);
  if (rankMap.size === 0 && deprioMap.size === 0) {
    return baseProposals;
  }

  const cloneProposal = (p) => {
    const ovr = rankMap.get(p.id);
    const dep = deprioMap.get(p.id);
    const next = { ...p };
    if (ovr) {
      next.rank = ovr.newRank ?? p.rank;
      next._overrideMeta = {
        oldRank: ovr.oldRank ?? p.rank,
        newRank: ovr.newRank ?? p.rank,
        rationale: ovr.rationale || '',
        confidence: ovr.confidence || '',
      };
    }
    if (dep) {
      next.deprioritized = true;
      next._deprioritizeReason = dep.reason || '';
    }
    // Always tag combo-disguised hypotheses (i.e. ones whose `includes`
    // field carries other H-IDs) so the UI can render the
    // "🔗 Combo alias of HX,HY" chip + tooltip on every render path.
    const includes = Array.isArray(p.includes) ? p.includes.filter(Boolean) : [];
    if (includes.length > 0) {
      next._isComboHypothesis = true;
      next._componentIds = includes;
      next._comboKey = [...includes].sort().join(',');
      next._comboReasonAnnotated = Boolean(
        dep && /^combo-alias:/.test(dep.reason || '')
      );
    }
    return next;
  };

  const phases = (baseProposals.phases || []).map((phase) => {
    const updated = (phase.proposals || []).map(cloneProposal);
    // Stable sort by effective rank ascending; rank 0 / undefined go last.
    updated.sort((a, b) => {
      const ra = (a.rank > 0 ? a.rank : 9999);
      const rb = (b.rank > 0 ? b.rank : 9999);
      return ra - rb;
    });
    return { ...phase, proposals: updated };
  });

  return { ...baseProposals, phases };
}

export default mergeOverrides;
