import React from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup, within } from '@testing-library/react';

import ProposalSelectionWidget from './ProposalSelectionWidget';

afterEach(() => cleanup());

// ── Fixtures ────────────────────────────────────────────────────────────

function groupsPayload() {
  // AF-native (ProposalIndex.to_dict) — the shape today's failing session has.
  return {
    version: '1',
    total_count: 5,
    groups: [
      {
        phase: 1,
        label: 'Attention & Core Architecture',
        description: 'Bets on attention.',
        proposals: [
          { id: 'P1', rank: 1, title: 'Diagnostics Baseline', summary: 'Instrument baselines', impact: 'high', complexity: 'low', problem: 'No baselines', approach: '**Instrument** HSTU' },
          { id: 'P2', rank: 2, title: 'Differential SiLU', summary: 'Sub-attn heads', impact: 'high', complexity: 'medium' },
          { id: 'P3', rank: 3, title: 'Routed Expert Attention', summary: 'MoE', impact: 'medium', complexity: 'high' },
        ],
      },
      {
        phase: 2,
        label: 'Sequence Compression',
        description: 'Compress the sequence dim.',
        proposals: [
          { id: 'P11', rank: 1, title: 'Temporal Density', summary: 'Density boundary', impact: 'high', complexity: 'medium' },
          { id: 'P12', rank: 2, title: 'Fourier Basis', summary: 'Learnable Fourier', impact: 'medium', complexity: 'medium' },
        ],
      },
    ],
    constraints: [
      {
        id: 'c1',
        kind: 'conflicts',
        proposal_ids: ['P11', 'P12'],
        reason: 'both edit hstu.py',
        severity: 'error',
      },
    ],
  };
}

function phasesPayload() {
  // Hub-native (ProposalSelectionData.to_dict) — already canonical.
  const p = groupsPayload();
  return {
    total_count: p.total_count,
    phases: p.groups, // structurally identical
    combo_constraints: [
      { id: 'c1', kind: 'conflicts', hypothesis_ids: ['P11', 'P12'], severity: 'error', reason: 'both edit hstu.py' },
    ],
  };
}

function baseConfig(overrides = {}) {
  return {
    input_mode: {
      prompt: 'Review and select proposals to advance.',
      metadata: { proposals: groupsPayload() },
    },
    ...overrides,
  };
}

// ── Tests ───────────────────────────────────────────────────────────────

describe('ProposalSelectionWidget — always renders rich grouped layout', () => {
  it('AF groups payload renders workstream tabs (not a flat soup)', () => {
    render(<ProposalSelectionWidget config={baseConfig()} onSubmit={vi.fn()} />);

    // Tab strip visible with the 2 workstream tabs.
    const tabs = screen.getAllByRole('tab');
    expect(tabs).toHaveLength(2);
    expect(tabs[0].textContent).toContain('Attention & Core Architecture');
    expect(tabs[1].textContent).toContain('Sequence Compression');

    // First tab active by default; proposals from group 1 visible.
    expect(screen.getByText(/P1:/)).toBeTruthy();
    expect(screen.getByText(/Diagnostics Baseline/)).toBeTruthy();
  });

  it('hub phases payload renders identically', () => {
    render(<ProposalSelectionWidget
      config={{ input_mode: { metadata: { proposals: phasesPayload() } } }}
      onSubmit={vi.fn()}
    />);
    const tabs = screen.getAllByRole('tab');
    expect(tabs).toHaveLength(2);
    expect(tabs[0].textContent).toContain('Attention & Core Architecture');
  });

  it('switching workstream tabs shows different proposals', () => {
    render(<ProposalSelectionWidget config={baseConfig()} onSubmit={vi.fn()} />);
    // Initially P1 visible, P11 hidden.
    expect(screen.getByText(/P1:/)).toBeTruthy();
    fireEvent.click(screen.getByRole('tab', { name: /Sequence Compression/ }));
    expect(screen.getByText(/P11:/)).toBeTruthy();
  });
});

describe('ProposalSelectionWidget — single group hides tab strip', () => {
  it('one-group payload renders cards directly without tabs', () => {
    const single = {
      total_count: 2,
      groups: [
        {
          phase: 1,
          label: 'Only Workstream',
          proposals: [
            { id: 'P1', rank: 1, title: 'A', summary: 'sa' },
            { id: 'P2', rank: 2, title: 'B', summary: 'sb' },
          ],
        },
      ],
    };
    render(<ProposalSelectionWidget
      config={{ input_mode: { metadata: { proposals: single } } }}
      onSubmit={vi.fn()}
    />);
    // No tab role rendered.
    expect(screen.queryAllByRole('tab')).toEqual([]);
    // Both cards visible directly.
    expect(screen.getByText(/P1:/)).toBeTruthy();
    expect(screen.getByText(/P2:/)).toBeTruthy();
  });
});

describe('ProposalSelectionWidget — markdown rendering', () => {
  it('renders proposal.approach as parsed markdown (not literal ** or |)', () => {
    render(<ProposalSelectionWidget config={baseConfig()} onSubmit={vi.fn()} />);
    // Expand P1 via the sibling expand button (a11y contract).
    const expandBtn = screen.getByRole('button', { name: /Expand P1 details/ });
    fireEvent.click(expandBtn);
    // MarkdownRenderer converts `**Instrument**` to a <strong> node — asserting
    // the literal characters `**` do NOT appear in the DOM text confirms it's
    // being parsed and not dumped raw.
    const body = screen.getByText(/Hypothesis Details/).parentElement;
    expect(body.textContent).not.toMatch(/\*\*/);
    // The word "Instrument" is rendered as a strong.
    const strong = within(body).getByText('Instrument');
    expect(strong.tagName.toLowerCase()).toBe('strong');
  });
});

describe('ProposalSelectionWidget — pre-selection', () => {
  it('pre-selects top-5 by rank across all phases on initial mount', () => {
    render(<ProposalSelectionWidget config={baseConfig()} onSubmit={vi.fn()} />);
    // Footer shows selected count (customQueries=0, only selection contributes).
    // Fixture has 5 total proposals — top-5 = all 5.
    expect(screen.getByText(/5 of 5 selected/)).toBeTruthy();
  });

  it('caps pre-selection at 5 when more than 5 proposals exist', () => {
    const many = groupsPayload();
    // Extend to 10 proposals.
    for (let i = 20; i < 25; i++) {
      many.groups[0].proposals.push({ id: `P${i}`, rank: i, title: `T${i}`, summary: '' });
    }
    many.total_count = 10;
    render(<ProposalSelectionWidget
      config={{ input_mode: { metadata: { proposals: many } } }}
      onSubmit={vi.fn()}
    />);
    // 5 of 10 pre-selected.
    expect(screen.getByText(/5 of 10 selected/)).toBeTruthy();
  });

  it('respects _submitted committed-replay: uses preSelectedProposals over top-N', () => {
    render(<ProposalSelectionWidget
      config={{
        input_mode: { metadata: { proposals: groupsPayload() } },
        _submitted: true,
        value: ['P2'],
      }}
      onSubmit={vi.fn()}
    />);
    expect(screen.getByText(/1 of 5 selected/)).toBeTruthy();
  });
});

describe('ProposalSelectionWidget — submit payload contract', () => {
  it('emits { selected_proposals, custom_queries, total_available } on submit', () => {
    const onSubmit = vi.fn();
    render(<ProposalSelectionWidget config={baseConfig()} onSubmit={onSubmit} />);
    fireEvent.click(screen.getByText(/Advance 5 proposals/));
    expect(onSubmit).toHaveBeenCalledTimes(1);
    const payload = onSubmit.mock.calls[0][0];
    expect(payload).toHaveProperty('selected_proposals');
    expect(payload.selected_proposals.sort()).toEqual(['P1', 'P11', 'P12', 'P2', 'P3']);
    expect(payload).toHaveProperty('custom_queries', []);
    expect(payload).toHaveProperty('total_available', 5);
    // Non-incremental path: no is_incremental key.
    expect(payload).not.toHaveProperty('is_incremental');
  });

  it('incremental submit: sends only NEW selections + is_incremental:true', () => {
    const onSubmit = vi.fn();
    render(<ProposalSelectionWidget
      config={{
        input_mode: { metadata: { proposals: groupsPayload() } },
        // Pretend P1 was already submitted in a prior round.
        _submittedIds: new Set(['P1']),
      }}
      onSubmit={onSubmit}
    />);
    // Widget pre-selects top-5; footer button says "Implement 4 new" (5 - 1
    // already submitted).
    fireEvent.click(screen.getByText(/Implement 4 new/));
    const payload = onSubmit.mock.calls[0][0];
    expect(payload.is_incremental).toBe(true);
    expect(payload.selected_proposals).not.toContain('P1');
    expect(payload.selected_proposals.length).toBe(4);
  });
});

describe('ProposalSelectionWidget — _submittedIds locks cards', () => {
  it('checkbox for a submitted proposal is disabled + checked', () => {
    render(<ProposalSelectionWidget
      config={{
        input_mode: { metadata: { proposals: groupsPayload() } },
        _submittedIds: new Set(['P1']),
      }}
      onSubmit={vi.fn()}
    />);
    const p1CheckBox = screen.getByRole('checkbox', { name: /Select P1:/ });
    expect(p1CheckBox.disabled).toBe(true);
    expect(p1CheckBox.checked).toBe(true);
  });
});

describe('ProposalSelectionWidget — mutual exclusion enforcement', () => {
  it('slot-conflicting checkbox is disabled + reason tooltip shown', () => {
    const payload = groupsPayload();
    // Assign the same slot to P1 and P2 so P2 becomes ME-blocked when P1 is
    // selected (which happens automatically via top-5 pre-select).
    payload.groups[0].proposals[0].slots = ['attention'];
    payload.groups[0].proposals[1].slots = ['attention'];
    render(<ProposalSelectionWidget
      config={{ input_mode: { metadata: { proposals: payload } } }}
      onSubmit={vi.fn()}
    />);
    const p2CheckBox = screen.getByRole('checkbox', { name: /Select P2:/ });
    expect(p2CheckBox.disabled).toBe(true);
    // Blocked reason is announced.
    expect(screen.getByText(/Conflicts with P1/)).toBeTruthy();
  });

  it('submitted proposals also occupy slots (incremental x ME)', () => {
    const payload = groupsPayload();
    payload.groups[0].proposals[0].slots = ['attention'];
    payload.groups[0].proposals[1].slots = ['attention'];
    render(<ProposalSelectionWidget
      config={{
        input_mode: { metadata: { proposals: payload } },
        _submittedIds: new Set(['P1']),
      }}
      onSubmit={vi.fn()}
    />);
    const p2CheckBox = screen.getByRole('checkbox', { name: /Select P2:/ });
    // P2 is not selected + not submitted, but should still be blocked because
    // P1 (submitted) occupies the 'attention' slot.
    expect(p2CheckBox.disabled).toBe(true);
  });
});

describe('ProposalSelectionWidget — advisory hints (conflicts, requires, recommends, dependencies)', () => {
  it('conflicts constraint surfaces as advisory hint chip WITHOUT disabling submit', () => {
    // Both P11 and P12 are in the conflicts constraint AND both pre-selected
    // by top-5 (all 5 items selected). Hint fires, submit stays enabled.
    render(<ProposalSelectionWidget config={baseConfig()} onSubmit={vi.fn()} />);
    // Switch to workstream 2 where P11/P12 live.
    fireEvent.click(screen.getByRole('tab', { name: /Sequence Compression/ }));
    // Both P11 and P12 have a hint chip mentioning "conflicts".
    const conflictChips = screen.getAllByText(/conflicts with/);
    expect(conflictChips.length).toBeGreaterThanOrEqual(2);
    // Submit is not disabled.
    const submit = screen.getByText(/Advance 5 proposals/);
    expect(submit.disabled).toBe(false);
  });

  it('per-proposal dependencies surface as info hint chip', () => {
    const payload = groupsPayload();
    payload.groups[0].proposals[2].dependencies = ['P1', 'P2'];
    render(<ProposalSelectionWidget
      config={{ input_mode: { metadata: { proposals: payload } } }}
      onSubmit={vi.fn()}
    />);
    expect(screen.getByText(/→ requires P1, P2/)).toBeTruthy();
  });
});

describe('ProposalSelectionWidget — A11y', () => {
  it('expand affordance is a real button (sibling of checkbox, not wrapping it)', () => {
    render(<ProposalSelectionWidget config={baseConfig()} onSubmit={vi.fn()} />);
    // A11y contract: separate <button> with aria-label "Expand P1 details".
    // Distinct from the checkbox aria-label "Select P1: ...".
    expect(screen.getByRole('button', { name: /Expand P1 details/ })).toBeTruthy();
    expect(screen.getByRole('checkbox', { name: /Select P1:/ })).toBeTruthy();
  });
});

// ── Full-Proposal lazy-fetch (proposal_file_abs → /api/view/<abs>) ──────────

describe('ProposalSelectionWidget — Full Proposal doc (lazy fetch)', () => {
  function payloadWithAbs(extra = {}) {
    // Every proposal gets a proposal_file_abs pointing at a fake runtime path.
    return {
      version: '1',
      total_count: 2,
      groups: [
        {
          phase: 1,
          label: 'Alpha',
          proposals: [
            {
              id: 'P1',
              rank: 1,
              title: 'One',
              summary: 'A short summary of P1.',
              problem: 'Real problem for P1.',
              approach: 'Real approach for P1.',
              proposal_file: 'proposals/P1.md',
              proposal_file_abs: '/data/runtime/session/outputs/proposals/P1.md',
              ...extra,
            },
            {
              id: 'P2',
              rank: 2,
              title: 'Two',
              summary: 'Summary P2.',
              problem: 'Problem P2.',
              proposal_file: 'proposals/P2.md',
              proposal_file_abs: '/data/runtime/session/outputs/proposals/P2.md',
            },
          ],
        },
      ],
    };
  }

  function configWithAbs() {
    return { input_mode: { metadata: { proposals: payloadWithAbs() } } };
  }

  function stubFetch(impl) {
    const fetchMock = vi.fn(impl);
    vi.stubGlobal('fetch', fetchMock);
    return fetchMock;
  }

  function flushMicrotasks() {
    // fetch().then callbacks run on the microtask queue; a Promise.resolve
    // await lets them settle before we assert render state.
    return new Promise((resolve) => setTimeout(resolve, 0));
  }

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('does NOT fetch on widget mount (all cards collapsed → lazy gate holds)', async () => {
    const fetchMock = stubFetch(() =>
      Promise.resolve({ ok: true, text: () => Promise.resolve('# P1\nBody.') }),
    );
    render(<ProposalSelectionWidget config={configWithAbs()} onSubmit={vi.fn()} />);
    await flushMicrotasks();
    // With MUI <Collapse unmountOnExit=false>, all 2 cards mount but the
    // useProposalDoc effect early-returns on `!enabled` → zero fetches.
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('expanding a card fires EXACTLY ONE fetch with the raw abs path', async () => {
    const fetchMock = stubFetch(() =>
      Promise.resolve({ ok: true, text: () => Promise.resolve('# P1\nBody.') }),
    );
    render(<ProposalSelectionWidget config={configWithAbs()} onSubmit={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: /Expand P1 details/ }));
    await flushMicrotasks();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const url = fetchMock.mock.calls[0][0];
    // R6: raw path, leading slash preserved → double-slash after /view/.
    // The RAW-path guard: NO %2F anywhere in the URL.
    expect(url).toMatch(/^\/api\/view\/\/data\/runtime\//);
    expect(url).not.toMatch(/%2F/);
  });

  it('collapse then re-expand fires a SECOND fetch (documented no-cache)', async () => {
    const fetchMock = stubFetch(() =>
      Promise.resolve({ ok: true, text: () => Promise.resolve('# P1\nBody.') }),
    );
    render(<ProposalSelectionWidget config={configWithAbs()} onSubmit={vi.fn()} />);
    const btn = screen.getByRole('button', { name: /Expand P1 details/ });
    fireEvent.click(btn);
    await flushMicrotasks();
    // Now the button aria-label is "Collapse P1 details".
    fireEvent.click(screen.getByRole('button', { name: /Collapse P1 details/ }));
    await flushMicrotasks();
    fireEvent.click(screen.getByRole('button', { name: /Expand P1 details/ }));
    await flushMicrotasks();
    // No widget-level cache → each expand fetches. This is DELIBERATE
    // (mirrors NodeDetailPanel.useNodeOutput; documented in the plan).
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('shows the "Full Proposal" section with rendered markdown on success', async () => {
    stubFetch(() =>
      Promise.resolve({ ok: true, text: () => Promise.resolve('# Heading\n\nBody prose.') }),
    );
    render(<ProposalSelectionWidget config={configWithAbs()} onSubmit={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: /Expand P1 details/ }));
    await flushMicrotasks();
    // "Full Proposal" section label rendered.
    expect(screen.getByText(/Full Proposal/)).toBeTruthy();
    // MarkdownRenderer produced an <h1>Heading</h1>.
    const h1 = screen.getByRole('heading', { level: 1, name: /Heading/ });
    expect(h1).toBeTruthy();
  });

  it('always renders InlineDetailFields (show-both) alongside the full doc', async () => {
    stubFetch(() =>
      Promise.resolve({ ok: true, text: () => Promise.resolve('# Full doc\nBody.') }),
    );
    render(<ProposalSelectionWidget config={configWithAbs()} onSubmit={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: /Expand P1 details/ }));
    await flushMicrotasks();
    // Inline PROBLEM label from InlineDetailFields is present.
    expect(screen.getByText(/PROBLEM/)).toBeTruthy();
    // AND the full-doc heading from the fetch response.
    expect(screen.getByRole('heading', { level: 1, name: /Full doc/ })).toBeTruthy();
  });

  it('on 404 shows the "unavailable" caption; inline fields remain', async () => {
    stubFetch(() =>
      Promise.resolve({ ok: false, status: 404, text: () => Promise.resolve('not found') }),
    );
    render(<ProposalSelectionWidget config={configWithAbs()} onSubmit={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: /Expand P1 details/ }));
    await flushMicrotasks();
    expect(screen.getByText(/full proposal doc unavailable/i)).toBeTruthy();
    // Inline fields still there — graceful, never a regression.
    expect(screen.getByText(/PROBLEM/)).toBeTruthy();
  });

  it('does NOT render "Full Proposal" section when proposal_file_abs is absent', async () => {
    const fetchMock = stubFetch(() =>
      Promise.resolve({ ok: true, text: () => Promise.resolve('should not be fetched') }),
    );
    // Payload without proposal_file_abs at all — the older-workspace path.
    const payload = payloadWithAbs();
    for (const g of payload.groups) {
      for (const p of g.proposals) {
        delete p.proposal_file_abs;
        delete p.proposal_file;
      }
    }
    render(<ProposalSelectionWidget
      config={{ input_mode: { metadata: { proposals: payload } } }}
      onSubmit={vi.fn()}
    />);
    fireEvent.click(screen.getByRole('button', { name: /Expand P1 details/ }));
    await flushMicrotasks();
    // No fetch (enabled=true but absPath falsy → early return).
    expect(fetchMock).not.toHaveBeenCalled();
    // No "Full Proposal" section.
    expect(screen.queryByText(/Full Proposal/)).toBeNull();
    // Inline fields still there.
    expect(screen.getByText(/PROBLEM/)).toBeTruthy();
  });

  it('expanding a second card fires a separate fetch for that proposal', async () => {
    const fetchMock = stubFetch(() =>
      Promise.resolve({ ok: true, text: () => Promise.resolve('body') }),
    );
    render(<ProposalSelectionWidget config={configWithAbs()} onSubmit={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: /Expand P1 details/ }));
    await flushMicrotasks();
    fireEvent.click(screen.getByRole('button', { name: /Expand P2 details/ }));
    await flushMicrotasks();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0][0]).toMatch(/\/P1\.md$/);
    expect(fetchMock.mock.calls[1][0]).toMatch(/\/P2\.md$/);
  });
});

// ── F1/F1b/F3 regression tests (plan: recursive-kindling-widget.md) ────────

describe('ProposalSelectionWidget — F1 safeOnSubmit guard', () => {
  it('does NOT throw when onSubmit is undefined (missing prop from caller)', () => {
    // Mimics ConversationToolWidget mounting the widget without an onSubmit
    // (readOnly replay, stale-closure remount, misconfigured host).
    // Before F1: the button click hit `onSubmit(payload)` unguarded and
    // crashed with "TypeError: onSubmit is not a function".
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    render(<ProposalSelectionWidget config={baseConfig()} />);
    // Click should NOT throw.
    expect(() =>
      fireEvent.click(screen.getByText(/Advance 5 proposals/)),
    ).not.toThrow();
    // Devtools signal is preserved for developers who see it in the console.
    expect(errorSpy).toHaveBeenCalledWith(
      '[ProposalSelectionWidget] onSubmit not a function; payload dropped',
      expect.objectContaining({ selected_proposals: expect.any(Array) }),
    );
    errorSpy.mockRestore();
  });

  it('does NOT throw when onSubmit is null', () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    render(<ProposalSelectionWidget config={baseConfig()} onSubmit={null} />);
    expect(() =>
      fireEvent.click(screen.getByText(/Advance 5 proposals/)),
    ).not.toThrow();
    expect(errorSpy).toHaveBeenCalled();
    errorSpy.mockRestore();
  });
});

describe('ProposalSelectionWidget — F1b dedupe protection', () => {
  it('double-click on fresh-submit only fires onSubmit ONCE', () => {
    const onSubmit = vi.fn();
    render(<ProposalSelectionWidget config={baseConfig()} onSubmit={onSubmit} />);
    const btn = screen.getByText(/Advance 5 proposals/);
    fireEvent.click(btn);
    // After first click: submitted=true, button relabels to "✅ Submitted"
    // and `submitDisabled` includes `submitted` so the button attribute
    // disables. But the `if (submitted) return;` guard at the top of
    // handleSubmit is the belt (attribute is suspenders) — invoke via
    // synthetic click and confirm no second onSubmit fires.
    fireEvent.click(btn);
    fireEvent.click(btn);
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });
});

describe('ProposalSelectionWidget — auto_implement checkbox payload (F3 widget-side)', () => {
  function configOpensDashboard() {
    return {
      input_mode: {
        metadata: {
          proposals: groupsPayload(),
          open_dashboard: 'experiment_hub',
          submit_label: '📊 Go To Experiment Hub',
        },
      },
    };
  }

  it('unchecked "Implement Now" → payload.auto_implement === false', () => {
    const onSubmit = vi.fn();
    render(<ProposalSelectionWidget config={configOpensDashboard()} onSubmit={onSubmit} />);
    // Do NOT click the checkbox — leave `implementNow` at its default `false`.
    fireEvent.click(screen.getByText(/Go To Experiment Hub/));
    const payload = onSubmit.mock.calls[0][0];
    expect(payload).toHaveProperty('auto_implement', false);
  });

  it('checked "Implement Now" → payload.auto_implement === true', () => {
    const onSubmit = vi.fn();
    render(<ProposalSelectionWidget config={configOpensDashboard()} onSubmit={onSubmit} />);
    // Toggle the "Implement Selected Proposals Now" checkbox.
    fireEvent.click(
      screen.getByRole('checkbox', { name: /Implement Selected Proposals Now/i }),
    );
    fireEvent.click(screen.getByText(/Go To Experiment Hub/));
    const payload = onSubmit.mock.calls[0][0];
    expect(payload).toHaveProperty('auto_implement', true);
  });

  it('no opensDashboard → payload omits auto_implement entirely', () => {
    const onSubmit = vi.fn();
    // baseConfig() has no `open_dashboard` / `submit_label` → opensDashboard false.
    render(<ProposalSelectionWidget config={baseConfig()} onSubmit={onSubmit} />);
    fireEvent.click(screen.getByText(/Advance 5 proposals/));
    const payload = onSubmit.mock.calls[0][0];
    expect(payload).not.toHaveProperty('auto_implement');
  });
});
