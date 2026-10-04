import { describe, it, expect } from 'vitest';
import { normalizeProposalData } from './normalizeProposalData';

// ── Fixtures (inline factories, per SingleChoiceWidget.test.js style) ───

function afGroupsFixture() {
  // Mirrors the real on-disk `proposals.json` shape emitted by
  // `ProposalIndex.to_dict()`.
  return {
    version: '1',
    total_count: 4,
    source_workspace: '/tmp/ws',
    groups: [
      {
        phase: 1,
        label: 'Attention & Core Architecture',
        description: 'Bets on attention.',
        proposals: [
          {
            id: 'P1',
            rank: 1,
            title: 'Diagnostics Baseline',
            summary: 'Instrument baselines',
            impact: 'high',
            complexity: 'low',
            problem: 'No baselines',
            approach: 'Instrument HSTU',
            dependencies: [],
            proposal_file: 'p1.md',
            tags: ['diagnostics'],
          },
          {
            id: 'P2',
            rank: 2,
            title: 'Differential SiLU',
            summary: 'Two attn heads, subtract.',
            impact: 'high',
            complexity: 'medium',
          },
        ],
        batches: [
          {
            id: 'b1',
            label: 'Diagnostics',
            proposal_ids: ['P1'],
          },
        ],
      },
      {
        phase: 2,
        label: 'Sequence Compression',
        proposals: [
          { id: 'P11', rank: 1, title: 'Temporal Density', summary: 'S' },
          { id: 'P12', rank: 2, title: 'Fourier Basis', summary: 'S' },
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
    themes: ['attention', 'compression'],
  };
}

function hubPhasesFixture() {
  return {
    phases: [
      {
        phase: 1,
        label: 'H-Phase 1',
        proposals: [
          {
            id: 'H1',
            rank: 1,
            title: 'H1 title',
            one_line_summary: 'one line',
            impact: 'high',
            complexity: 'low',
            theme: 'Attn',
            source_workers: ['w1'],
            slots: ['attention'],
            includes: [],
            probability: 0.9,
          },
        ],
        batches: [{ id: 'B1', label: 'Batch 1', hypothesis_ids: ['H1'] }],
      },
    ],
    combo_constraints: [
      {
        id: 'c1',
        kind: 'mutually_exclusive',
        hypothesis_ids: ['H1', 'H2'],
      },
    ],
    total_count: 1,
    themes: ['attn'],
  };
}

describe('normalizeProposalData — AF groups → hub phases', () => {
  it('renames groups → phases and preserves phase metadata', () => {
    const norm = normalizeProposalData(afGroupsFixture());
    expect(norm.phases).toHaveLength(2);
    expect(norm.phases[0].label).toBe('Attention & Core Architecture');
    expect(norm.phases[0].description).toBe('Bets on attention.');
    expect(norm.phases[0].phase).toBe(1);
    expect(norm.phases[1].label).toBe('Sequence Compression');
  });

  it('renames batch proposal_ids → hypothesis_ids', () => {
    const norm = normalizeProposalData(afGroupsFixture());
    const batch = norm.phases[0].batches[0];
    expect(batch.id).toBe('b1');
    expect(batch.hypothesis_ids).toEqual(['P1']);
    // proposal_ids intentionally NOT present on output (canonical is hypothesis_ids)
    expect(batch.proposal_ids).toBeUndefined();
  });

  it('renames top-level constraints → combo_constraints and per-constraint proposal_ids → hypothesis_ids', () => {
    const norm = normalizeProposalData(afGroupsFixture());
    expect(norm.combo_constraints).toHaveLength(1);
    const c = norm.combo_constraints[0];
    expect(c.id).toBe('c1');
    expect(c.kind).toBe('conflicts');
    expect(c.hypothesis_ids).toEqual(['P11', 'P12']);
    expect(c.severity).toBe('error');
    expect(c.reason).toBe('both edit hstu.py');
  });

  it('IDs are NOT rewritten (no P# → H#)', () => {
    const norm = normalizeProposalData(afGroupsFixture());
    const allIds = norm.phases.flatMap((ph) => ph.proposals.map((p) => p.id));
    expect(allIds).toEqual(['P1', 'P2', 'P11', 'P12']);
    // Batch and constraint ids also stay P#
    expect(norm.phases[0].batches[0].hypothesis_ids).toEqual(['P1']);
    expect(norm.combo_constraints[0].hypothesis_ids).toEqual(['P11', 'P12']);
  });

  it('preserves total_count and themes', () => {
    const norm = normalizeProposalData(afGroupsFixture());
    expect(norm.total_count).toBe(4);
    expect(norm.themes).toEqual(['attention', 'compression']);
  });

  it('spread preserves per-proposal fields (does not whitelist)', () => {
    const raw = afGroupsFixture();
    raw.groups[0].proposals[0]._overrideMeta = {
      newRank: 5,
      oldRank: 1,
      confidence: 'high',
    };
    raw.groups[0].proposals[0].deprioritized = true;
    const norm = normalizeProposalData(raw);
    const p1 = norm.phases[0].proposals[0];
    expect(p1._overrideMeta).toEqual({ newRank: 5, oldRank: 1, confidence: 'high' });
    expect(p1.deprioritized).toBe(true);
    expect(p1.tags).toEqual(['diagnostics']);
    expect(p1.proposal_file).toBe('p1.md');
  });

  it('fills one_line_summary from summary when missing', () => {
    const norm = normalizeProposalData(afGroupsFixture());
    const p2 = norm.phases[0].proposals[1];
    expect(p2.one_line_summary).toBe(p2.summary);
  });

  it('exposes slots from p.metadata.slots when top-level slots absent', () => {
    const raw = {
      groups: [
        {
          phase: 1,
          label: 'X',
          proposals: [{ id: 'P1', title: 'T', metadata: { slots: ['attn', 'seq'] } }],
        },
      ],
    };
    const norm = normalizeProposalData(raw);
    expect(norm.phases[0].proposals[0].slots).toEqual(['attn', 'seq']);
  });

  it('slots defaults to empty array when neither top-level nor metadata slots present', () => {
    const norm = normalizeProposalData(afGroupsFixture());
    expect(norm.phases[0].proposals[0].slots).toEqual([]);
  });
});

describe('normalizeProposalData — hub phases passthrough (idempotent)', () => {
  it('accepts an already-hub payload without translation', () => {
    const norm = normalizeProposalData(hubPhasesFixture());
    expect(norm.phases[0].label).toBe('H-Phase 1');
    expect(norm.phases[0].proposals[0].id).toBe('H1');
    expect(norm.phases[0].proposals[0].theme).toBe('Attn');
    expect(norm.phases[0].proposals[0].slots).toEqual(['attention']);
    expect(norm.phases[0].batches[0].hypothesis_ids).toEqual(['H1']);
    expect(norm.combo_constraints[0].kind).toBe('mutually_exclusive');
    expect(norm.combo_constraints[0].hypothesis_ids).toEqual(['H1', 'H2']);
  });

  it('is idempotent — double normalize yields identical structure', () => {
    const once = normalizeProposalData(hubPhasesFixture());
    const twice = normalizeProposalData(once);
    // Compare structural equality on the canonical fields.
    expect(twice.phases[0].label).toBe(once.phases[0].label);
    expect(twice.phases[0].proposals[0].id).toBe(once.phases[0].proposals[0].id);
    expect(twice.phases[0].batches[0].hypothesis_ids).toEqual(
      once.phases[0].batches[0].hypothesis_ids,
    );
    expect(twice.combo_constraints[0].hypothesis_ids).toEqual(
      once.combo_constraints[0].hypothesis_ids,
    );
    expect(twice.total_count).toBe(once.total_count);
  });

  it('preferring hub form when both aliases are present (canonical wins)', () => {
    const raw = {
      phases: [{ phase: 1, label: 'HUB', proposals: [{ id: 'H1' }] }],
      groups: [{ phase: 99, label: 'AF', proposals: [{ id: 'P99' }] }],
      combo_constraints: [{ id: 'c-hub', kind: 'mutually_exclusive', hypothesis_ids: ['H1'] }],
      constraints: [{ id: 'c-af', kind: 'conflicts', proposal_ids: ['P99'] }],
    };
    const norm = normalizeProposalData(raw);
    expect(norm.phases[0].label).toBe('HUB');
    expect(norm.combo_constraints[0].id).toBe('c-hub');
  });
});

describe('normalizeProposalData — options-only fallback', () => {
  it('synthesizes a single phase from a bare option list', () => {
    const options = [
      { value: 'a', label: 'Alpha', description: 'A option' },
      { value: 'b', label: 'Beta', description: 'B option' },
    ];
    const norm = normalizeProposalData(null, options);
    expect(norm.phases).toHaveLength(1);
    expect(norm.phases[0].proposals).toHaveLength(2);
    expect(norm.phases[0].proposals[0].id).toBe('a');
    expect(norm.phases[0].proposals[0].title).toBe('Alpha');
    expect(norm.phases[0].proposals[0].summary).toBe('A option');
    expect(norm.total_count).toBe(2);
  });

  it('options fall through when rawProposals has real data (raw wins)', () => {
    const norm = normalizeProposalData(afGroupsFixture(), [
      { value: 'x', label: 'X' },
    ]);
    // AF groups payload should be used; options are ignored.
    expect(norm.phases).toHaveLength(2);
    expect(norm.phases[0].proposals[0].id).toBe('P1');
  });
});

describe('normalizeProposalData — malformed / empty input safety', () => {
  it('null input → empty canonical shape', () => {
    const norm = normalizeProposalData(null);
    expect(norm.phases).toEqual([]);
    expect(norm.combo_constraints).toEqual([]);
    expect(norm.total_count).toBe(0);
    expect(norm.themes).toEqual([]);
  });

  it('undefined input → empty canonical shape', () => {
    const norm = normalizeProposalData(undefined);
    expect(norm.phases).toEqual([]);
  });

  it('empty object → empty canonical shape', () => {
    const norm = normalizeProposalData({});
    expect(norm.phases).toEqual([]);
    expect(norm.total_count).toBe(0);
  });

  it('non-object (string/number) → empty canonical shape', () => {
    expect(normalizeProposalData('not a dict').phases).toEqual([]);
    expect(normalizeProposalData(42).phases).toEqual([]);
    expect(normalizeProposalData([]).phases).toEqual([]);
  });

  it('constraint with kind:"requires" but no requires_ids does not throw', () => {
    const raw = {
      groups: [{ phase: 1, label: 'X', proposals: [{ id: 'P1' }] }],
      constraints: [{ id: 'c1', kind: 'requires', proposal_ids: ['P1'] }],
    };
    const norm = normalizeProposalData(raw);
    expect(norm.combo_constraints[0].kind).toBe('requires');
    expect(norm.combo_constraints[0].hypothesis_ids).toEqual(['P1']);
    expect(norm.combo_constraints[0].requires_ids).toBeUndefined();
  });

  it('phase with missing proposals array vs empty array both yield []', () => {
    const raw = {
      groups: [
        { phase: 1, label: 'A' }, // no proposals key
        { phase: 2, label: 'B', proposals: [] }, // empty array
      ],
    };
    const norm = normalizeProposalData(raw);
    expect(norm.phases).toHaveLength(2);
    expect(norm.phases[0].proposals).toEqual([]);
    expect(norm.phases[1].proposals).toEqual([]);
  });

  it('proposal without id is filtered out', () => {
    const raw = {
      groups: [
        {
          phase: 1,
          label: 'X',
          proposals: [{ title: 'no id' }, { id: 'P1', title: 'ok' }],
        },
      ],
    };
    const norm = normalizeProposalData(raw);
    expect(norm.phases[0].proposals).toHaveLength(1);
    expect(norm.phases[0].proposals[0].id).toBe('P1');
  });

  it('empty options list → empty canonical shape', () => {
    const norm = normalizeProposalData(null, []);
    expect(norm.phases).toEqual([]);
  });
});
