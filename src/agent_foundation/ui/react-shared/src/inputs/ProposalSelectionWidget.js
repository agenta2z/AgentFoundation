/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * ProposalSelectionWidget — review and select from a ranked proposal set.
 *
 * MUI restore (post-audit): mirrors RankEvolve's pure-MUI reference
 * (`fbsource260327/…/rankevolve/src/webui/react/src/components/widgets/
 * ProposalSelectionWidget.js` — 64 `sx=`, 0 `className=`). Every visual
 * element is a MUI primitive (`Box`/`Chip`/`Tabs`/`Card`/`Paper`/`Collapse`/
 * `Checkbox`/`Button`/`IconButton`/`Typography`/`Alert`/`FormControlLabel`)
 * with `sx={{…}}` tokens off the rovo dark theme (`primary.main`,
 * `background.paper`, `text.secondary`, `divider`, `theme.custom.surfaces.*`).
 * NO dead CSS class-hooks — that pattern was the "wall of text" bug and is
 * eliminated here.
 *
 * UNIFIED (AF + RankEvolve dialects). Normalises `groups` / `proposal_ids` /
 * `constraints` (AF) OR `phases` / `hypothesis_ids` / `combo_constraints`
 * (hub) into one canonical hub-shape via `normalizeProposalData`, then always
 * renders the rich grouped/tabbed/markdown-aware layout via `HypothesisCard`.
 * Options-only payload synthesised into a single phase by the normaliser.
 *
 * Capabilities (unchanged from prior; only the render is MUI now):
 *   - MUI Tabs per phase (scrollable, underline highlight).
 *   - MUI Card per batch with `<Collapse>` body + count badge in header;
 *     default-collapsed when `>6 hyps && none selected` (G2).
 *   - MUI Paper per hypothesis w/ visible border, hover, selected accent;
 *     MUI Chip strip (impact / complexity / slots / impl-status) — color-coded
 *     by `impactColor()` / `complexityColor()`.
 *   - Per-hypothesis expand for PROBLEM / APPROACH / NOTES markdown
 *     (`MarkdownRenderer` in expand body only — collapsed cards stay light).
 *   - Default top-N-globally pre-select (matches backend `pre_select_top_n=5`).
 *   - Mutual-exclusion enforcement via `computeBlockedIds`; disabled Checkbox
 *     with tooltip carrying the reason.
 *   - Advisory hints via `checkComboConstraints` for `conflicts` / `requires`
 *     / `recommends` / per-proposal `dependencies` — never blocks submit.
 *   - Incremental submit: `config._submittedIds` (a Set) locks already-done
 *     hypotheses; only NEW selections submitted (`is_incremental: true`).
 *   - `config._implementationStatus` per-H status chips (queued/running/…).
 *   - `submit_label` / `open_dashboard` metadata → primary MUI Button labels
 *     "📊 Go To {label}" (the proposal-selection → Experiment Hub handoff).
 *   - G3 "Pre-selected N from M active combos" banner via `config._activeCombos`.
 *   - G11 ONE clean primary submit button; view-report affordance demoted to
 *     small secondary `<IconButton>` on the LEFT of the count (no duplicate CTA).
 *   - G12 "Implement Selected Proposals Now" checkbox (visible only when
 *     `opensDashboard` — the handoff path); state threaded via onSubmit's
 *     `auto_implement` field. Default UNCHECKED (back-compat preserved).
 *   - `readOnly` / `_submitted` / `value` (committed-replay) rendering.
 *   - A11y: expand IconButton is a sibling of the Checkbox (not nested); ME
 *     disable / hint activity announces via `aria-live="polite"`.
 *
 * Context-free: emits onSubmit({ selected_proposals, custom_queries,
 * total_available, is_incremental?, auto_implement? }). Hub-specific behavior
 * (Auto Mode, direct /implement-hypothesis routing, combo pre-narrowing) is
 * layered by the host view via WidgetHostView's enrichConfig/onWidgetSubmit —
 * NEVER in the widget itself. Hub concerns arrive via `config._X`:
 *   - `_activeCombos = {count, appliedAt, pendingCount, hypothesisIds, resetHandler}`
 *   - `_submittedIds` (Set), `_failedHypothesisIds` (Set)
 *   - `_implementationStatus` (per-H map)
 * The widget does NOT import `useSession`/hub hooks — that decoupling is a
 * deliberate design invariant (preserves testability + reuse).
 */

import React, { useEffect, useMemo, useState } from 'react';

import {
  Alert,
  Box,
  Button,
  Card,
  Checkbox,
  Chip,
  CircularProgress,
  Collapse,
  Divider,
  IconButton,
  Paper,
  Tab,
  Tabs,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DescriptionIcon from '@mui/icons-material/Description';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SettingsIcon from '@mui/icons-material/Settings';
import BarChartIcon from '@mui/icons-material/BarChart';
import CloseIcon from '@mui/icons-material/Close';

import { MarkdownRenderer } from '../common/MarkdownRenderer';
import {
  checkComboConstraints,
  computeBlockedIds,
} from '../dashboards/experiment_hub/utils/multiTaskHelpers';
import { normalizeProposalData } from './normalizeProposalData';

const MAX_VISIBLE = 4;
const RANK_BADGES = ['🥇', '🥈', '🥉'];
// MUST match the backend's `pre_select_top_n` default (executor.open_experiment_hub).
const DEFAULT_TOP_N_GLOBAL = 5;
// G2 — default-collapse a batch when it has more than this many hypotheses AND
// none of them are selected. Keeps the "wall of 46 proposals" scannable.
const BATCH_AUTOCOLLAPSE_THRESHOLD = 6;

const DETAIL_FIELDS = [
  { key: 'problem', label: 'PROBLEM' },
  { key: 'approach', label: 'APPROACH' },
  { key: 'notes', label: 'NOTES' },
];

// ── Helpers ───────────────────────────────────────────────────────────────

function getRankBadge(rank) {
  if (rank >= 1 && rank <= 3) return RANK_BADGES[rank - 1];
  return String(rank == null ? '' : rank);
}

/** Map "High/Medium/Low/…" impact strings to MUI Chip colors.
 * IMPACT semantic: high = large expected improvement (good) → error tone
 * ("hot spot", user attention), medium = warning, low = success (nothing to
 * worry about). We use the same tone map for complexity below — the color
 * meaning is "attention level", not good/bad. */
function impactColor(impact) {
  const s = String(impact || '').toLowerCase();
  if (s.includes('high')) return 'error';
  if (s.includes('med')) return 'warning';
  if (s.includes('low')) return 'success';
  return 'default';
}

/** Map "High/Medium/Low/…" complexity strings to MUI Chip colors.
 * COMPLEXITY semantic: high = hard to implement (bad) → error tone,
 * low = easy to ship (good) → success tone. */
function complexityColor(complexity) {
  const s = String(complexity || '').toLowerCase();
  if (s.includes('high')) return 'error';
  if (s.includes('med')) return 'warning';
  if (s.includes('low')) return 'success';
  return 'default';
}

/** Chip tooltip text for impact/complexity/status/slots — surfaces the domain
 * meaning of "high/low/medium" so the user isn't guessing. Called from the
 * `Tooltip` wrappers below. */
function impactTooltip(impact) {
  const raw = String(impact || '').trim();
  if (!raw) return 'Impact — expected improvement magnitude';
  const s = raw.toLowerCase();
  if (s.startsWith('high')) return `Impact: ${raw} — large expected improvement`;
  if (s.startsWith('med')) return `Impact: ${raw} — moderate expected improvement`;
  if (s.startsWith('low')) return `Impact: ${raw} — small expected improvement`;
  return `Impact: ${raw}`;
}

function complexityTooltip(complexity) {
  const raw = String(complexity || '').trim();
  if (!raw) return 'Complexity — implementation effort';
  const s = raw.toLowerCase();
  if (s.startsWith('high')) return `Complexity: ${raw} — major implementation effort`;
  if (s.startsWith('med')) return `Complexity: ${raw} — moderate implementation effort`;
  if (s.startsWith('low')) return `Complexity: ${raw} — small implementation effort (config-only, etc.)`;
  return `Complexity: ${raw}`;
}

/** Map implementation status → MUI Chip color + label. */
function implStatusMeta(implStatus) {
  if (implStatus === 'completed') return { color: 'success', label: '✅ Done' };
  if (implStatus === 'running') return { color: 'info', label: '▶ Running' };
  if (implStatus === 'error') return { color: 'error', label: '❌ Error' };
  return { color: 'default', label: '⏳ Queued' };
}

/**
 * Sort all proposals across all phases by GLOBAL rank, return top-N IDs.
 * Sort key: (rank ASC, -source_workers_len ASC, id ASC) — mirrors the backend
 * tiebreaker chain so the in-chat + Hub-direct-open paths agree byte-for-byte.
 */
function selectTopNGlobally(phases, n) {
  const all = [];
  for (const ph of (phases || [])) {
    for (const p of (ph?.proposals || [])) {
      if (p && p.id) all.push(p);
    }
  }
  all.sort((a, b) => {
    const ra = Number.isFinite(a.rank) ? a.rank : 9999;
    const rb = Number.isFinite(b.rank) ? b.rank : 9999;
    if (ra !== rb) return ra - rb;
    const wa = (a.source_workers || []).length;
    const wb = (b.source_workers || []).length;
    if (wa !== wb) return wb - wa;  // descending workers count
    return String(a.id).localeCompare(String(b.id));
  });
  const k = Math.max(1, Number.isFinite(n) ? n : DEFAULT_TOP_N_GLOBAL);
  return all.slice(0, k).map(p => p.id);
}

/**
 * Build a per-proposal advisory-hint index from the constraints array and
 * per-proposal `dependencies`/`cross_refs`. Returns a Map<pid, {items}>
 * where `items` is a list of { kind, tone, text, tooltip, relatedIds } entries
 * ready to render. Never affects submit; hints are display-only.
 */
function buildHintsIndex(allProposalsFlat, constraints, selectedSet) {
  const map = new Map(); // pid -> { items: [...] }
  const pushHint = (pid, hint) => {
    let bucket = map.get(pid);
    if (!bucket) {
      bucket = { items: [] };
      map.set(pid, bucket);
    }
    bucket.items.push(hint);
  };
  const violations = checkComboConstraints(selectedSet, constraints);
  const routeAll = (arr, toneDefault) => {
    for (const v of arr) {
      const c = v.constraint || {};
      const ids = c.hypothesis_ids || [];
      const tone = c.severity === 'warning' ? 'warning'
                 : c.severity === 'info' ? 'info'
                 : toneDefault;
      const relatedIds = v.conflicting || v.missing || ids.filter((i) => i !== undefined);
      let text;
      let tooltip;
      if (c.kind === 'conflicts') {
        text = `⚠️ conflicts with ${relatedIds.filter((i) => !isSelfHint(i)).join(', ')}`;
        tooltip = c.reason || c.label || 'Advisory conflict; resolved at execution time.';
      } else if (c.kind === 'mutually_exclusive') {
        text = `⚡ mutually exclusive with ${relatedIds.join(', ')}`;
        tooltip = c.reason || c.label || 'Only one of these can be selected.';
      } else if (c.kind === 'requires') {
        text = `→ requires ${(v.missing || []).join(', ') || 'other proposals'}`;
        tooltip = c.reason || c.label || 'Missing dependencies for this proposal.';
      } else if (c.kind === 'recommends') {
        text = `💡 recommends ${(v.constraint.requires_ids || []).join(', ')}`;
        tooltip = c.reason || c.label || 'Recommended companion proposals.';
      }
      if (!text) continue;
      for (const pid of ids) {
        pushHint(pid, { kind: c.kind, tone, text, tooltip, relatedIds });
      }
    }
  };
  routeAll(violations.errors, 'error');
  routeAll(violations.warnings, 'warning');
  routeAll(violations.infos, 'info');
  for (const p of allProposalsFlat) {
    const deps = Array.isArray(p.dependencies) ? p.dependencies.filter((d) => d && d !== p.id) : [];
    if (deps.length === 0) continue;
    pushHint(p.id, {
      kind: 'dependencies',
      tone: 'info',
      text: `→ requires ${deps.join(', ')}`,
      tooltip: 'This proposal declares these as prerequisites.',
      relatedIds: deps,
    });
  }
  return map;
}

function isSelfHint(v) {
  return v === undefined || v === null || v === '';
}

function relativeTime(iso) {
  if (!iso) return '';
  try {
    const then = new Date(iso).getTime();
    const now = Date.now();
    const diff = Math.max(0, Math.round((now - then) / 1000));
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
    return `${Math.round(diff / 86400)}d ago`;
  } catch {
    return '';
  }
}

/** Inline-markdown component overrides for the collapsed-card SUMMARY preview.
 *
 * MarkdownRenderer's default block components (`<p>`, `<ul>`, `<li>`, tables)
 * add vertical spacing + block layout that breaks a 2-line preview clamp.
 * Override every block-level element to a `<span>` (or `null` for tables) so
 * the output flows inline and CSS `-webkit-line-clamp: 2` truncates cleanly.
 *
 * Backend `summary` field quality is spotty (see task #144 follow-up — some
 * proposals get the raw "Diagnostic metrics" bullet list or a full markdown
 * TABLE in `summary` instead of a proper one-liner). This inline rendering
 * salvages the display: bold/code render nicely, tables are hidden, bullets
 * flatten to "• item • item …", and the line-clamp does the rest.
 */
const INLINE_SUMMARY_COMPONENTS = {
  p: ({ children }) => <span>{children}</span>,
  h1: ({ children }) => <span>{children}</span>,
  h2: ({ children }) => <span>{children}</span>,
  h3: ({ children }) => <span>{children}</span>,
  h4: ({ children }) => <span>{children}</span>,
  h5: ({ children }) => <span>{children}</span>,
  h6: ({ children }) => <span>{children}</span>,
  ul: ({ children }) => <span> {children} </span>,
  ol: ({ children }) => <span> {children} </span>,
  li: ({ children }) => <span>• {children} </span>,
  blockquote: ({ children }) => <span>{children}</span>,
  hr: () => <span> — </span>,
  // Hide tables entirely in preview — they never fit a 2-line clamp.
  // Users see the full table in the expanded body.
  table: () => null,
  thead: () => null,
  tbody: () => null,
  tr: () => null,
  th: () => null,
  td: () => null,
  // Suppress heavy syntax-highlighted code blocks in preview; inline code stays.
  pre: ({ children }) => <span>{children}</span>,
};

/** Derive a preview text from `one_line_summary` (if the backend emits it) or
 * the first N chars of `summary` (with tables stripped when the whole summary
 * IS a table — like P3.md's "Concrete schedule" section). Returns raw
 * markdown; render via `<MarkdownRenderer components={INLINE_SUMMARY_COMPONENTS} />`. */
function derivePreviewSummary(proposal) {
  const oneLine = String(proposal.one_line_summary || '').trim();
  if (oneLine) return oneLine;
  const summary = String(proposal.summary || '').trim();
  if (!summary) return '';
  // If the summary consists entirely of a markdown table (all lines start with
  // `|`), fall back to just the first row (renders as one line via the
  // inline overrides — table components are null-rendered but the row text
  // itself survives if we strip the pipes ourselves).
  const lines = summary.split('\n').map(l => l.trim()).filter(Boolean);
  const nonTable = lines.filter(l => !l.startsWith('|'));
  if (nonTable.length > 0) {
    // Prefer the first prose line; the rest are still available on expand.
    return nonTable.join(' ').slice(0, 400);
  }
  // All-table summary — strip pipes + separators to get a flat "col1 col2 …" preview.
  return lines
    .filter(l => !/^\|[\s\-|:]+\|?$/.test(l))  // drop `|---|---|` separator rows
    .map(l => l.replace(/^\|/, '').replace(/\|$/, '').split('|').map(c => c.trim()).filter(Boolean).join(' · '))
    .join(' — ')
    .slice(0, 400);
}

// ── Hint badges (MUI Chip variant) ─────────────────────────────────────────

function HintBadges({ hints }) {
  if (!hints || !hints.items || hints.items.length === 0) return null;
  const toneColor = { error: 'error', warning: 'warning', info: 'default' };
  return (
    <Box sx={{ display: 'inline-flex', gap: 0.5, flexWrap: 'wrap' }}>
      {hints.items.map((h, i) => (
        <Chip
          key={i}
          label={h.text}
          size="small"
          variant="outlined"
          color={toneColor[h.tone] || 'default'}
          title={h.tooltip}
          sx={{ fontSize: '0.7rem', height: 20 }}
        />
      ))}
    </Box>
  );
}

// ── Pre-selected banner (G3) ───────────────────────────────────────────────

function PreSelectedBanner({ activeCombos, selectedCount, totalCount, onReset }) {
  if (!activeCombos || !activeCombos.count) return null;
  const pending = activeCombos.pendingCount || 0;
  const appliedTxt = activeCombos.appliedAt ? ` Applied ${relativeTime(activeCombos.appliedAt)}.` : '';
  return (
    <Alert
      severity="info"
      sx={{ mt: 1.5, py: 0.5 }}
      action={onReset ? (
        <Button size="small" onClick={onReset} sx={{ fontSize: '0.75rem' }}>
          Reset to active combos
        </Button>
      ) : undefined}
    >
      Pre-selected {selectedCount} of {totalCount} from {activeCombos.count} active combo
      {activeCombos.count === 1 ? '' : 's'}
      {pending > 0 && ` + ${pending} pending`}.{appliedTxt}
    </Alert>
  );
}

// ── useProposalDoc ─────────────────────────────────────────────────────────
/**
 * Lazy-fetch the full per-proposal Markdown doc from `/api/view/<abs>`.
 *
 * Mirrors NodeDetailPanel.useNodeOutput (the shipped, working pattern for
 * `_runtime`-file rendering), with ONE required difference: an explicit
 * `enabled` gate. The widget mounts up to 46 `HypothesisCard`s at once and MUI
 * `<Collapse>` defaults to `unmountOnExit=false`, so an ungated fetch would
 * fire from every card on widget open. React always runs the hooks; it's the
 * *fetch* that must be gated. `enabled = expanded` makes it lazy.
 *
 * IMPORTANT — `useEffect` deps MUST be `[absPath, enabled]`:
 *   `useNodeOutput` uses `[outputPath]` alone (no gate); copying that verbatim
 *   while adding an `enabled` param means toggling false→true on expand won't
 *   re-run the effect and the doc never loads.
 *
 * URL scheme: raw interpolation, no `encodeURIComponent`. The route is
 * `/api/view/{file_path:path}` whose `:path` converter matches real slashes;
 * `%2F` breaks it (Starlette does not decode `%2F` in path segments). With an
 * absolute `absPath`, the URL is `/api/view//data/…` (double slash) — exactly
 * what `useNodeOutput` produces.
 *
 * No widget-level cache — docs are ~2.5KB, fetched only on expand; re-expand
 * re-fetches (acceptable; a bounded module Map could be added later if needed).
 */
function useProposalDoc(absPath, enabled) {
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(() => Boolean(enabled && absPath));
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!enabled || !absPath) {
      setContent('');
      setLoading(false);
      setError(null);
      return undefined;
    }
    let cancelled = false;
    setLoading(true);
    setContent('');
    setError(null);
    fetch(`/api/view/${absPath}`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((text) => {
        if (!cancelled) {
          setContent(text);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message || String(err));
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [absPath, enabled]);

  return { content, loading, error };
}

// ── InlineDetailFields (structured PROBLEM/APPROACH/NOTES from JSON) ───────
/**
 * Renders the pre-existing DETAIL_FIELDS + cross_refs block. Extracted so the
 * expanded-card render can compose it with the new "Full Proposal" section
 * (show-both) or fall back to it when no `.md` doc is available.
 *
 * Behavior-preserving — mirrors the prior inline render in `HypothesisCard`:
 * `hasDetails` excludes fields whose value equals `proposal.title` (dedup);
 * fallback text is `one_line_summary || summary || title`; `cross_refs`
 * caption renders below the fields when present.
 */
function InlineDetailFields({ proposal }) {
  const hasDetails = DETAIL_FIELDS.some(
    (f) => proposal[f.key] && proposal[f.key] !== proposal.title,
  );
  const fallbackText = proposal.one_line_summary || proposal.summary || proposal.title || '';
  return (
    <>
      {hasDetails ? (
        <Box>
          {DETAIL_FIELDS.map(({ key, label }) => {
            const value = proposal[key];
            if (!value || value === proposal.title) return null;
            return (
              <Box key={key} sx={{ mb: 1.5 }}>
                <Typography
                  variant="overline"
                  sx={{ fontSize: '0.65rem', fontWeight: 700, color: 'text.secondary', letterSpacing: '0.5px', display: 'block', mb: 0.25 }}
                >
                  {label}
                </Typography>
                <Box sx={{ fontSize: '0.85rem' }}>
                  <MarkdownRenderer content={String(value)} />
                </Box>
              </Box>
            );
          })}
        </Box>
      ) : (
        <Box sx={{ fontSize: '0.85rem' }}>
          <MarkdownRenderer content={String(fallbackText)} />
        </Box>
      )}
      {proposal.cross_refs && (
        <Typography variant="caption" sx={{ color: 'text.secondary', mt: 1, display: 'block' }}>
          Cross-refs: {proposal.cross_refs}
        </Typography>
      )}
    </>
  );
}

// ── HypothesisCard (MUI Paper) ─────────────────────────────────────────────

function HypothesisCard({
  proposal,
  selected,
  onToggle,
  expanded,
  onExpand,
  submitted,
  implStatus,
  submittedIds,
  hints,
  blockedReason,
}) {
  const isSubmittedLock = submittedIds && submittedIds.has(proposal.id);
  const isLocked = submitted || isSubmittedLock;
  const isMEBlocked = !!blockedReason && !selected && !isLocked;
  const isDisabled = isLocked || isMEBlocked;
  const disabledTooltip = isMEBlocked
    ? blockedReason
    : (isLocked ? 'Already submitted (locked)' : undefined);
  const statusMeta = implStatus ? implStatusMeta(implStatus) : null;

  // Lazy-fetch the per-proposal `P{N}.md` doc when the card is expanded.
  // `enabled=expanded` is load-bearing: MUI <Collapse> defaults to
  // `unmountOnExit=false`, so all ~46 cards mount at once — without the gate,
  // every card would fetch its doc on widget open (not lazy).
  const { content: docContent, loading: docLoading, error: docError } =
    useProposalDoc(proposal.proposal_file_abs, expanded);
  const hasDoc = !!proposal.proposal_file_abs;

  return (
    <Paper
      variant="outlined"
      sx={{
        mb: 0.75,
        borderColor: selected ? 'primary.main' : 'divider',
        bgcolor: selected ? 'action.selected' : 'transparent',
        opacity: isMEBlocked ? 0.55 : 1,
        transition: 'border-color 120ms ease',
        '&:hover': !isMEBlocked ? { borderColor: 'primary.light' } : undefined,
      }}
    >
      {/* A11y: expand IconButton is a sibling of the Checkbox — not nested. */}
      <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, p: 1 }}>
        <Tooltip title={disabledTooltip || ''} placement="top" disableHoverListener={!disabledTooltip}>
          <span>
            <Checkbox
              size="small"
              checked={!!(selected || isSubmittedLock)}
              disabled={isDisabled}
              onChange={(e) => { e.stopPropagation(); onToggle(proposal.id); }}
              inputProps={{
                'aria-label': `Select ${proposal.id}: ${proposal.title || ''}`,
                'aria-describedby': disabledTooltip ? `${proposal.id}-disabled-reason` : undefined,
              }}
              sx={{ p: 0.5, mt: -0.25 }}
            />
          </span>
        </Tooltip>
        <Box sx={{ minWidth: 24, textAlign: 'center', fontSize: '1.1rem', pt: 0.25, opacity: 0.75 }}>
          {getRankBadge(proposal.rank)}
        </Box>
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'baseline', gap: 0.75 }}>
            <Typography component="span" sx={{ color: 'primary.main', fontWeight: 700, fontSize: '0.9rem' }}>
              {proposal.id}
            </Typography>
            {proposal.id ? <Typography component="span" sx={{ fontSize: '0.9rem' }}>: {proposal.title}</Typography>
                        : <Typography component="span" sx={{ fontSize: '0.9rem' }}>{proposal.title}</Typography>}
            {proposal._overrideMeta && (
              <Chip
                label={`🆕 rank ${proposal._overrideMeta.newRank} (was ${proposal._overrideMeta.oldRank})`}
                size="small"
                color="info"
                variant="outlined"
                title={
                  `Suggested rank ${proposal._overrideMeta.newRank} (was ${proposal._overrideMeta.oldRank}) — `
                  + `confidence: ${proposal._overrideMeta.confidence || '?'}\n\n`
                  + (proposal._overrideMeta.rationale || '(no rationale)')
                }
                sx={{ fontSize: '0.65rem', height: 20 }}
              />
            )}
            {proposal.deprioritized && (
              <Chip
                label="↓ deprioritized"
                size="small"
                color="warning"
                variant="outlined"
                title={proposal._deprioritizeReason || '(no reason)'}
                sx={{ fontSize: '0.65rem', height: 20 }}
              />
            )}
            {proposal.theme && (
              <Typography component="span" variant="caption" sx={{ color: 'text.secondary', ml: 0.5 }}>
                {proposal.theme}
              </Typography>
            )}
          </Box>
          {(() => {
            // Issue #1 — render summary as INLINE markdown so `**bold**` and
            // `` `code` `` don't render as literal punctuation. Backend
            // summary quality is spotty (see task #144 — sometimes a table
            // or bullet list instead of a one-liner); derivePreviewSummary
            // salvages the display, and INLINE_SUMMARY_COMPONENTS forces
            // block elements to flow inline so line-clamp works.
            const previewText = derivePreviewSummary(proposal);
            if (!previewText) return null;
            return (
              <Box
                sx={{
                  fontSize: '0.82rem',
                  color: 'text.secondary',
                  mt: 0.5,
                  display: '-webkit-box',
                  WebkitLineClamp: 2,
                  WebkitBoxOrient: 'vertical',
                  overflow: 'hidden',
                  // Keep inline `<code>` from breaking the clamp: shrink its
                  // background/padding so the em-height stays close to text.
                  '& code': {
                    fontSize: '0.85em',
                    padding: '0 3px',
                  },
                  // Suppress bold-styled headers/paragraphs adding extra
                  // whitespace — they're overridden to <span> in the
                  // component map but still inherit default font weight.
                  '& strong': { fontWeight: 600 },
                }}
              >
                <MarkdownRenderer
                  content={previewText}
                  components={INLINE_SUMMARY_COMPONENTS}
                />
              </Box>
            );
          })()}
        </Box>
        {/* Chip strip — impact / complexity / slots / impl-status.
            Issue #7 — Tooltip wrappers surface domain meaning of high/low/medium
            (e.g. "Impact: high — large expected improvement"). Otherwise users
            hover the chip and get nothing. */}
        {proposal.impact && (
          <Tooltip title={impactTooltip(proposal.impact)} placement="top" arrow>
            <Chip
              label={proposal.impact}
              size="small"
              color={impactColor(proposal.impact)}
              variant="outlined"
              sx={{ fontSize: '0.7rem', height: 22, flexShrink: 0 }}
            />
          </Tooltip>
        )}
        {proposal.complexity && (
          <Tooltip title={complexityTooltip(proposal.complexity)} placement="top" arrow>
            <Chip
              label={proposal.complexity}
              size="small"
              color={complexityColor(proposal.complexity)}
              variant="outlined"
              sx={{ fontSize: '0.7rem', height: 22, flexShrink: 0 }}
            />
          </Tooltip>
        )}
        {proposal.slots && proposal.slots.length > 0 && (
          <Tooltip
            title={`Slot: ${proposal.slots.join(', ')} — mutually exclusive with other proposals in the same slot`}
            placement="top"
            arrow
          >
            <Chip
              label={`🎰 ${proposal.slots.join('+')}`}
              size="small"
              variant="outlined"
              color="default"
              sx={{ fontSize: '0.65rem', height: 22, flexShrink: 0, opacity: 0.75 }}
            />
          </Tooltip>
        )}
        {statusMeta && (
          <Tooltip
            title={`Implementation status: ${implStatus}`}
            placement="top"
            arrow
          >
            <Chip
              label={statusMeta.label}
              size="small"
              color={statusMeta.color}
              variant="outlined"
              sx={{ fontSize: '0.65rem', height: 20, fontWeight: 600, flexShrink: 0 }}
            />
          </Tooltip>
        )}
        <IconButton
          size="small"
          onClick={() => onExpand(proposal.id)}
          aria-expanded={expanded}
          aria-label={expanded ? `Collapse ${proposal.id} details` : `Expand ${proposal.id} details`}
          sx={{ p: 0.25, flexShrink: 0 }}
        >
          {expanded ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
        </IconButton>
      </Box>

      {(hints && hints.items.length > 0) && (
        <Box sx={{ pl: 5, pb: 0.75 }}>
          <HintBadges hints={hints} />
        </Box>
      )}
      {isMEBlocked && (
        <Typography
          id={`${proposal.id}-disabled-reason`}
          variant="caption"
          sx={{ pl: 5, pb: 0.75, display: 'block', color: 'warning.main' }}
        >
          🔒 {blockedReason}
        </Typography>
      )}

      <Collapse in={expanded}>
        {/* maxHeight cap is CONDITIONAL: keep the 480px inner scroll for the
         * inline-only path (older workspaces / no `.md` → no regression); drop
         * it when a full doc renders so the ~2.5KB Markdown doesn't nest-scroll
         * inside a 480px window (the widget's own container handles overflow). */}
        <Box sx={{ pl: 5, pr: 2, pb: 2, pt: 0, ...(hasDoc ? {} : { maxHeight: 480, overflowY: 'auto' }) }}>
          {proposal.source_workers && proposal.source_workers.length > 0 && (
            <Typography variant="caption" sx={{ color: 'text.secondary', mb: 1, display: 'block' }}>
              Source: {proposal.source_workers.join(', ')}
            </Typography>
          )}
          <Typography
            variant="overline"
            sx={{ fontSize: '0.7rem', fontWeight: 700, color: 'text.secondary', letterSpacing: '0.5px', mb: 0.5, display: 'block' }}
          >
            Hypothesis Details
          </Typography>
          <InlineDetailFields proposal={proposal} />
          {/* Show-both (D2): full `.md` under the inline fields when available.
           * `hasDoc` is truthy only when the backend attached `proposal_file_abs`
           * (enrichment succeeded for this proposal). Absent → block omitted →
           * zero regression for older workspaces / graceful fallback. */}
          {hasDoc && (
            <>
              <Divider sx={{ my: 1.5 }} />
              <Typography
                variant="overline"
                sx={{ fontSize: '0.7rem', fontWeight: 700, color: 'text.secondary', letterSpacing: '0.5px', mb: 0.5, display: 'block' }}
              >
                Full Proposal
              </Typography>
              {docLoading ? (
                <CircularProgress size={16} />
              ) : docError ? (
                <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block' }}>
                  (full proposal doc unavailable)
                </Typography>
              ) : docContent ? (
                <Box sx={{ fontSize: '0.85rem' }}>
                  <MarkdownRenderer content={docContent} />
                </Box>
              ) : null}
            </>
          )}
        </Box>
      </Collapse>
    </Paper>
  );
}

// ── BatchSection (MUI Card with Collapse) — G2 ─────────────────────────────

function BatchSection({
  batch,
  proposals,
  selectedIds,
  onToggle,
  expandedIds,
  onExpand,
  submitted,
  implementationStatus,
  submittedIds,
  hintsByPid,
  blockedById,
}) {
  // Defensive: accept both `hypothesis_ids` (hub canonical / normaliser output)
  // and `proposal_ids` (AF canonical) in case a raw hub payload bypasses the
  // normaliser somehow. Won't happen on the current data path, but keeps the
  // BatchSection safe if it ever does.
  const memberIds = batch.hypothesis_ids || batch.proposal_ids || [];
  const batchProposals = memberIds
    .map(id => proposals.find(p => p.id === id))
    .filter(Boolean);

  const selectedCount = batchProposals.filter(p => selectedIds.has(p.id)).length;
  const allSelected = batchProposals.length > 0 && selectedCount === batchProposals.length;
  const someSelected = selectedCount > 0 && !allSelected;

  // G2 — default-collapse when the batch is big AND no member is selected.
  // User can toggle open; state is per-batch local.
  const [collapsed, setCollapsed] = useState(
    () => batchProposals.length > BATCH_AUTOCOLLAPSE_THRESHOLD && selectedCount === 0,
  );

  if (batchProposals.length === 0) return null;

  const toggleBatch = () => {
    if (submitted) return;
    if (allSelected) {
      batchProposals.forEach(p => { if (selectedIds.has(p.id)) onToggle(p.id); });
    } else {
      batchProposals.forEach(p => {
        if (!selectedIds.has(p.id) && !blockedById?.has(p.id)) onToggle(p.id);
      });
    }
  };

  const batchLabel = batch.label
    ? `BATCH ${batch.id}: ${batch.label}${batch.timeline ? ` (${batch.timeline})` : ''}`
    : `BATCH ${batch.id}${batch.timeline ? ` (${batch.timeline})` : ''}`;

  return (
    <Card
      variant="outlined"
      sx={{ mb: 1.5, borderColor: 'divider', bgcolor: 'transparent' }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 1, pl: 1.5 }}>
        <Checkbox
          size="small"
          checked={allSelected}
          indeterminate={someSelected}
          disabled={submitted}
          onChange={toggleBatch}
          inputProps={{ 'aria-label': `Toggle all in ${batchLabel}` }}
          sx={{ p: 0.5 }}
        />
        <Typography
          sx={{
            fontSize: '0.78rem',
            fontWeight: 700,
            color: 'text.secondary',
            letterSpacing: '0.3px',
            textTransform: 'uppercase',
            flexGrow: 1,
          }}
        >
          {batchLabel}
        </Typography>
        <Chip
          label={`${selectedCount}/${batchProposals.length}`}
          size="small"
          variant="outlined"
          color={selectedCount > 0 ? 'primary' : 'default'}
          sx={{ fontSize: '0.7rem', height: 22 }}
        />
        <IconButton
          size="small"
          onClick={() => setCollapsed(c => !c)}
          aria-expanded={!collapsed}
          aria-label={collapsed ? `Expand ${batchLabel}` : `Collapse ${batchLabel}`}
          sx={{ p: 0.25 }}
        >
          {collapsed ? <ExpandMoreIcon fontSize="small" /> : <ExpandLessIcon fontSize="small" />}
        </IconButton>
      </Box>
      <Collapse in={!collapsed}>
        <Box sx={{ px: 1.5, pb: 1.5 }}>
          {batchProposals.map(p => (
            <HypothesisCard
              key={p.id}
              proposal={p}
              selected={selectedIds.has(p.id)}
              onToggle={onToggle}
              expanded={expandedIds.has(p.id)}
              onExpand={onExpand}
              submitted={submitted}
              implStatus={implementationStatus?.[p.id]}
              submittedIds={submittedIds}
              hints={hintsByPid.get(p.id)}
              blockedReason={blockedById?.get(p.id)}
            />
          ))}
        </Box>
      </Collapse>
    </Card>
  );
}

// ── Top-level widget ──────────────────────────────────────────────────────

export default function ProposalSelectionWidget({ config, onSubmit, onOpenDashboard, onView, readOnly, value }) {
  const inputMode = config?.input_mode || config || {};
  const metadata = inputMode.metadata || config?.metadata || {};
  const prompt =
    inputMode.prompt || config?.prompt || config?.title || '';
  const allowZero = metadata.allow_zero ?? false;

  // Normalise either dialect (AF groups OR hub phases OR options-only) into
  // the canonical hub shape. Stable-dep memo pattern to avoid re-normalising
  // on every render.
  const rawProposals = metadata.proposals;
  const rawOptions = useMemo(
    () => inputMode.options || config?.options || config?.choices || null,
    [inputMode.options, config?.options, config?.choices],
  );
  const norm = useMemo(
    () => normalizeProposalData(rawProposals, rawOptions),
    [rawProposals, rawOptions],
  );

  const phases = norm.phases;
  const combo_constraints = norm.combo_constraints;
  const totalCount = norm.total_count;

  const allProposalsFlat = useMemo(() => {
    const flat = [];
    for (const ph of phases) {
      for (const p of (ph.proposals || [])) flat.push(p);
    }
    return flat;
  }, [phases]);

  // Submit-label handoff: when the tool wants the submit button to open a
  // dashboard (proposal-selection --experiment-hub), relabel to "📊 Go To X".
  const submitLabel = metadata.submit_label || null;
  const opensDashboard = !!(metadata.open_dashboard || submitLabel);

  // View Full Report affordance (G11 — demoted to icon-only when opensDashboard,
  // so it doesn't compete with the primary CTA).
  const viewPath = metadata.view || null;
  const viewLabel = metadata.view_label || 'View Full Proposal Report';

  // Host-injected enrichments (WidgetHostView / SelectionView).
  const implementationStatus = config?._implementationStatus || null;
  const submittedIds = config?._submittedIds || null;
  const isDisabled = !!readOnly || !!config?._disabled || !!config?.readOnly;
  const activeCombos = config?._activeCombos || null;  // G3 banner

  // Committed-replay / pre-submitted state. `ConversationToolWidget` forwards the
  // committed answer via the top-level `readOnly` + `value` props (the convention
  // every other leaf widget — Confirmation/MultipleChoice/SingleChoice — already
  // honors). On the committed transcript path `value` is the submitted payload
  // OBJECT ({ selected_proposals, custom_queries, … }); we read those fields.
  // The `config._*` fallbacks preserve the host-injected enrichment path
  // (WidgetHostView / SelectionView) which seeds committed state via config.
  const isPreSubmitted = !!readOnly || config?._submitted || false;
  const preSelectedProposals = useMemo(() => {
    if (Array.isArray(value?.selected_proposals)) return value.selected_proposals;
    if (Array.isArray(value)) return value;
    if (Array.isArray(config?.value)) return config.value;
    return config?._selectedProposals || [];
  }, [value, config?.value, config?._selectedProposals]);
  const preCustomQueries =
    (value && Array.isArray(value.custom_queries) ? value.custom_queries : null)
    || config?._customQueries
    || [];

  const [activeTab, setActiveTab] = useState(0);
  const [expandedIds, setExpandedIds] = useState(() => new Set());
  const [showMore, setShowMore] = useState(() => {
    const initial = {};
    if (!isPreSubmitted && phases.length > 0) {
      const defaultSel = new Set(selectTopNGlobally(phases, DEFAULT_TOP_N_GLOBAL));
      phases.forEach((phase, phaseIdx) => {
        const beyond = (phase.proposals || []).slice(MAX_VISIBLE);
        if (beyond.some(p => defaultSel.has(p.id))) initial[phaseIdx] = true;
      });
    }
    return initial;
  });
  const [customQueries, setCustomQueries] = useState(preCustomQueries);
  const [customInput, setCustomInput] = useState('');
  const [showCustom, setShowCustom] = useState(false);
  const [submitted, setSubmitted] = useState(isPreSubmitted);

  // R1 — the chat-side G12 "Implement Selected Proposals Now" checkbox was
  // removed from the `opensDashboard` handoff path: clicking the primary button
  // now OPENS the hub (seeded, on the Selection tab) WITHOUT resolving the
  // pending input or auto-implementing. The "confirm & implement" decision moved
  // INTO the hub (SelectionView's "Confirm Selection & Start"). No `implementNow`
  // state remains — the handoff no longer carries `auto_implement`.

  const [selectedIds, setSelectedIds] = useState(() => {
    if (isPreSubmitted && preSelectedProposals.length > 0) {
      return new Set(preSelectedProposals.map(String));
    }
    if (Array.isArray(metadata.preselected_ids) || typeof metadata.preselected_ids === 'string') {
      const raw = metadata.preselected_ids;
      const ids = Array.isArray(raw) ? raw : String(raw).split(',');
      return new Set(ids.map((s) => String(s).trim()).filter(Boolean));
    }
    if (phases.length > 0) {
      return new Set(selectTopNGlobally(phases, DEFAULT_TOP_N_GLOBAL));
    }
    return new Set();
  });

  // Track user-manual selection changes so the G3 "Reset to active combos"
  // link is only meaningful after the user has diverged from the pre-selection.
  const userTouchedRef = React.useRef(false);
  const handleActiveCombosReset = () => {
    if (!activeCombos || !activeCombos.hypothesisIds) return;
    userTouchedRef.current = false;
    setSelectedIds(new Set(activeCombos.hypothesisIds.map(String)));
  };

  // ── Constraint enforcement (ME) + advisory hints ───────────────────────
  // Selected + Submitted both occupy slots so incremental users can't add a
  // conflicting new selection to an already-submitted batch.
  const enforcementSelectedSet = useMemo(() => {
    const s = new Set(selectedIds);
    if (submittedIds) {
      for (const id of submittedIds) s.add(id);
    }
    return s;
  }, [selectedIds, submittedIds]);

  const { blocked: blockedById } = useMemo(
    () => computeBlockedIds(enforcementSelectedSet, allProposalsFlat),
    [enforcementSelectedSet, allProposalsFlat],
  );

  const hintsByPid = useMemo(
    () => buildHintsIndex(allProposalsFlat, combo_constraints, selectedIds),
    [allProposalsFlat, combo_constraints, selectedIds],
  );

  const toggleSelect = (id) => {
    if (submitted) return;
    userTouchedRef.current = true;
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        if (blockedById.has(id)) {
          return prev;
        }
        next.add(id);
      }
      return next;
    });
  };

  const toggleExpand = (id) => {
    setExpandedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const addCustomQuery = () => {
    const text = customInput.trim();
    if (!text || submitted) return;
    setCustomQueries([...customQueries, text]);
    setCustomInput('');
  };
  const removeCustomQuery = (idx) => {
    if (submitted) return;
    setCustomQueries(customQueries.filter((_, i) => i !== idx));
  };

  // F1 guard: mirrors GroupedWidget.js:93-98 — the widget is registered as a
  // built-in and can be dispatched from `ConversationToolWidget` render paths
  // whose parents may omit onSubmit (readOnly replay, stale-closure remount).
  // Emit a devtools-visible error instead of crashing the app.
  const safeOnSubmit = (payload) => {
    if (typeof onSubmit !== 'function') {
      console.error(
        '[ProposalSelectionWidget] onSubmit not a function; payload dropped',
        payload,
      );
      return;
    }
    onSubmit(payload);
  };

  // F1 guard for the dashboard-handoff path (R1). When `opensDashboard`, the
  // primary button OPENS + SEEDS the Experiment Hub (via `onOpenDashboard`)
  // WITHOUT resolving the Phase-2b pending input — the SOP stays at 2b and the
  // in-hub confirm advances it. Degrades gracefully to the old `onSubmit` rail
  // when the host doesn't provide `onOpenDashboard` (standalone AF webui).
  const safeOnOpenDashboard = (payload) => {
    if (typeof onOpenDashboard !== 'function') {
      console.error(
        '[ProposalSelectionWidget] onOpenDashboard not a function; '
        + 'falling back to onSubmit(auto_implement:false)',
        payload,
      );
      safeOnSubmit({ ...payload, auto_implement: false });
      return;
    }
    onOpenDashboard(payload);
  };

  const handleSubmit = () => {
    // F1b (belt): on the fresh-submit path, `submitted` becomes true after the
    // first click and the button relabels to "Submitted", but `submitDisabled`
    // (below) doesn't consume `submitted`. Guard here so a rapid double-click
    // (or an event race with the disabled attribute) can't fire twice.
    if (submitted) return;
    // R1 — dashboard handoff: OPEN the hub (seed all selections) but do NOT
    // resolve the pending input / mark the widget submitted. The widget stays
    // live so the user can reopen the hub and the SOP holds at Phase 2b.
    if (opensDashboard) {
      const selected = (submittedIds && submittedIds.size > 0)
        ? [...selectedIds].filter(id => !submittedIds.has(id))
        : [...selectedIds];
      safeOnOpenDashboard({ selected_proposals: selected });
      return;
    }
    if (submittedIds && submittedIds.size > 0) {
      // Incremental mode: only submit NEW selections.
      const newSelections = [...selectedIds].filter(id => !submittedIds.has(id));
      if (newSelections.length === 0) return;
      const payload = {
        selected_proposals: newSelections,
        custom_queries: customQueries,
        total_available: totalCount,
        is_incremental: true,
      };
      safeOnSubmit(payload);
      return;
    }
    setSubmitted(true);
    const payload = {
      selected_proposals: [...selectedIds],
      custom_queries: customQueries,
      total_available: totalCount,
    };
    safeOnSubmit(payload);
  };

  const newSelectionCount = (submittedIds && submittedIds.size > 0)
    ? [...selectedIds].filter(id => !submittedIds.has(id)).length
    : selectedIds.size;
  const selectedCount = selectedIds.size + customQueries.length;
  // F1b (suspenders): include `submitted` so the button visibly disables
  // AFTER the first submit on the fresh-submit path (matches the "✅ Submitted"
  // relabel below).
  const submitDisabled = isDisabled
    || submitted
    || (!allowZero && (submittedIds && submittedIds.size > 0 ? newSelectionCount === 0 : selectedIds.size === 0));

  // Issue #6 — the backend often sets `submit_label = "Go To Experiment Hub"`
  // (already includes the "Go To" prefix). Prepending "Go To " unconditionally
  // yields the doubled "Go To Go To Experiment Hub" bug shown in image10.
  // Fix: trust the backend to send the FULL label when it sets `submit_label`.
  // Only synthesize the "Go To Experiment Hub" default when the backend
  // didn't set the label. Backends that just want the noun (e.g.
  // `submit_label = "Experiment Hub"`) should either send the full label OR
  // omit and let us default; no widget-side heuristic guessing.
  const submitButtonLabel = submitted
    ? '✅ Submitted'
    : opensDashboard
      ? (submitLabel || 'Go To Experiment Hub')
      : (submittedIds && submittedIds.size > 0)
        ? `⚙ Implement Selected (${newSelectionCount} new)`
        : `Advance ${selectedIds.size} proposal${selectedIds.size === 1 ? '' : 's'}`;

  // Aggregate active-hint / ME-block counts for the aria-live announcer.
  const meBlockedCount = blockedById.size;
  const activeHintCount = [...hintsByPid.values()].reduce((n, b) => n + b.items.length, 0);

  // ── Render: empty ─────────────────────────────────────────────────────
  if (phases.length === 0) {
    return (
      <Box>
        <Alert severity="info">{prompt || 'No proposal data available.'}</Alert>
      </Box>
    );
  }

  const hasBatches = phases.some(p => (p.batches || []).length > 0);
  const showTabs = phases.length > 1;

  // ── Render: rich grouped layout (always) ──────────────────────────────
  return (
    <Box>
      <Box sx={{ mb: 2 }}>
        {/* Copy: "hypotheses across N phases" mirrors RankEvolve reference. */}
        <Typography variant="subtitle1" sx={{ fontSize: '0.95rem', fontWeight: 700 }}>
          {totalCount} {totalCount === 1 ? 'hypothesis' : 'hypotheses'}
          {phases.length > 1 ? ` across ${phases.length} phases` : ''}
        </Typography>
        {prompt && (
          <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }}>
            {prompt}
          </Typography>
        )}
        <PreSelectedBanner
          activeCombos={activeCombos}
          selectedCount={selectedIds.size}
          totalCount={totalCount}
          onReset={handleActiveCombosReset}
        />
      </Box>

      {/* A11y: live region for ME-disable and active-hint announcements. */}
      <Box
        role="status"
        aria-live="polite"
        sx={{ position: 'absolute', width: 1, height: 1, overflow: 'hidden', clipPath: 'inset(50%)' }}
      >
        {meBlockedCount > 0 && `${meBlockedCount} option${meBlockedCount === 1 ? '' : 's'} disabled due to mutual exclusion.`}
        {activeHintCount > 0 && ` ${activeHintCount} advisory hint${activeHintCount === 1 ? '' : 's'} active.`}
      </Box>

      {showTabs && (
        <Tabs
          value={activeTab}
          onChange={(_, v) => setActiveTab(v)}
          variant="scrollable"
          scrollButtons="auto"
          sx={{ borderBottom: 1, borderColor: 'divider', mb: 2, minHeight: 36 }}
        >
          {phases.map((phase, i) => (
            <Tab
              key={i}
              label={`${phase.label || `Phase ${phase.phase ?? i + 1}`} (${(phase.proposals || []).length})`}
              sx={{ minHeight: 36, textTransform: 'none', fontSize: '0.85rem' }}
            />
          ))}
        </Tabs>
      )}

      {phases.map((phase, phaseIdx) => {
        const isActive = !showTabs || activeTab === phaseIdx;
        return (
          <Box
            key={phaseIdx}
            role={showTabs ? 'tabpanel' : undefined}
            hidden={showTabs && activeTab !== phaseIdx}
          >
            {isActive && (
              <>
                {phase.description && (
                  <Box sx={{ color: 'text.secondary', mb: 1.5, fontSize: '0.85rem' }}>
                    <MarkdownRenderer content={String(phase.description)} />
                  </Box>
                )}
                {hasBatches && (phase.batches || []).length > 0 ? (
                  (phase.batches || []).map(batch => (
                    <BatchSection
                      key={batch.id}
                      batch={batch}
                      proposals={phase.proposals || []}
                      selectedIds={selectedIds}
                      onToggle={toggleSelect}
                      expandedIds={expandedIds}
                      onExpand={toggleExpand}
                      submitted={submitted || isDisabled}
                      implementationStatus={implementationStatus}
                      submittedIds={submittedIds}
                      hintsByPid={hintsByPid}
                      blockedById={blockedById}
                    />
                  ))
                ) : (
                  <>
                    {(phase.proposals || []).slice(0, MAX_VISIBLE).map(p => (
                      <HypothesisCard
                        key={p.id}
                        proposal={p}
                        selected={selectedIds.has(p.id)}
                        onToggle={toggleSelect}
                        expanded={expandedIds.has(p.id)}
                        onExpand={toggleExpand}
                        submitted={submitted || isDisabled}
                        implStatus={implementationStatus?.[p.id]}
                        submittedIds={submittedIds}
                        hints={hintsByPid.get(p.id)}
                        blockedReason={blockedById.get(p.id)}
                      />
                    ))}
                    {(phase.proposals || []).length > MAX_VISIBLE && (
                      <>
                        <Button
                          size="small"
                          variant="text"
                          disabled={submitted}
                          endIcon={showMore[phaseIdx] ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                          onClick={() => setShowMore({ ...showMore, [phaseIdx]: !showMore[phaseIdx] })}
                          sx={{ mt: 0.5, textTransform: 'none' }}
                        >
                          {showMore[phaseIdx]
                            ? 'Hide'
                            : `Show ${(phase.proposals || []).length - MAX_VISIBLE} more`}
                        </Button>
                        {showMore[phaseIdx] && (phase.proposals || []).slice(MAX_VISIBLE).map(p => (
                          <HypothesisCard
                            key={p.id}
                            proposal={p}
                            selected={selectedIds.has(p.id)}
                            onToggle={toggleSelect}
                            expanded={expandedIds.has(p.id)}
                            onExpand={toggleExpand}
                            submitted={submitted || isDisabled}
                            implStatus={implementationStatus?.[p.id]}
                            submittedIds={submittedIds}
                            hints={hintsByPid.get(p.id)}
                            blockedReason={blockedById.get(p.id)}
                          />
                        ))}
                      </>
                    )}
                  </>
                )}
              </>
            )}
          </Box>
        );
      })}

      {/* Add Custom Hypothesis */}
      <Box sx={{ mt: 2 }}>
        <Button
          size="small"
          variant="outlined"
          disabled={submitted}
          startIcon={<AddIcon />}
          onClick={() => setShowCustom(!showCustom)}
          sx={{ textTransform: 'none' }}
        >
          {showCustom ? 'Hide' : 'Add Custom Hypothesis'}
        </Button>
        <Collapse in={showCustom}>
          <Box sx={{ mt: 1, display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <TextField
                size="small"
                fullWidth
                placeholder="Enter your own hypothesis or research direction..."
                value={customInput}
                disabled={submitted}
                onChange={(e) => setCustomInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') addCustomQuery(); }}
                inputProps={{ 'aria-label': 'Custom hypothesis text' }}
              />
              <Button
                variant="contained"
                size="small"
                onClick={addCustomQuery}
                disabled={!customInput.trim() || submitted}
                startIcon={<AddIcon />}
              >
                Add
              </Button>
            </Box>
            {customQueries.map((q, i) => (
              <Box key={i} sx={{ display: 'flex', alignItems: 'center', gap: 1, pl: 1 }}>
                <Typography variant="body2" sx={{ flex: 1, color: 'text.secondary' }}>
                  — &quot;{q}&quot;
                </Typography>
                {!submitted && (
                  <IconButton
                    size="small"
                    onClick={() => removeCustomQuery(i)}
                    aria-label={`Remove custom hypothesis: ${q}`}
                  >
                    <CloseIcon fontSize="small" />
                  </IconButton>
                )}
              </Box>
            ))}
          </Box>
        </Collapse>
      </Box>

      {/* Footer: G11 ONE clean primary button; view-report demoted to
          small IconButton on the LEFT; G12 checkbox between count and submit
          (only visible when opensDashboard). */}
      <Box
        sx={{
          mt: 2,
          pt: 1.5,
          borderTop: 1,
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          gap: 1.5,
          flexWrap: 'wrap',
        }}
      >
        {viewPath && (
          <Tooltip title={viewLabel} placement="top">
            <IconButton
              size="small"
              onClick={() => onView && onView(viewPath)}
              aria-label={viewLabel}
            >
              <DescriptionIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        )}
        <Typography
          variant="body2"
          sx={{ ml: 'auto', color: 'text.secondary', fontSize: '0.85rem' }}
        >
          {selectedCount} of {totalCount} selected
        </Typography>
        {/* R1 — the chat-side "Implement Selected Proposals Now" (G12) checkbox
            was removed here. On the `opensDashboard` path the primary button now
            OPENS the hub (seeded, Selection tab) and the user confirms + starts
            implementation INSIDE the hub. */}
        <Button
          variant="contained"
          size="medium"
          disabled={submitDisabled}
          onClick={handleSubmit}
          startIcon={opensDashboard ? <BarChartIcon /> : <SettingsIcon />}
          sx={{ textTransform: 'none' }}
        >
          {submitButtonLabel}
        </Button>
      </Box>
    </Box>
  );
}
