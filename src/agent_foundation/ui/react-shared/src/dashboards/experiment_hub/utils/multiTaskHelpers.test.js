import { describe, it, expect } from 'vitest';
import { checkComboConstraints, computeBlockedIds } from './multiTaskHelpers';

describe('checkComboConstraints — conflicts (new branch)', () => {
  const conflictsConstraint = (severity = 'error') => ({
    id: 'c1',
    kind: 'conflicts',
    hypothesis_ids: ['H1', 'H2', 'H3'],
    reason: 'all edit hstu.py',
    severity,
  });

  it('emits advisory (infos), NEVER errors, when ≥2 members are selected', () => {
    const v = checkComboConstraints(
      new Set(['H1', 'H2']),
      [conflictsConstraint('error')],
    );
    expect(v.errors).toEqual([]);
    expect(v.warnings).toEqual([]);
    expect(v.infos).toHaveLength(1);
    expect(v.infos[0].constraint.id).toBe('c1');
    expect(v.infos[0].conflicting).toEqual(['H1', 'H2']);
  });

  it('routes to infos regardless of severity (warning / info / error / undefined)', () => {
    for (const sev of ['error', 'warning', 'info', undefined]) {
      const v = checkComboConstraints(
        new Set(['H1', 'H3']),
        [conflictsConstraint(sev)],
      );
      expect(v.errors).toEqual([]);
      expect(v.warnings).toEqual([]);
      expect(v.infos).toHaveLength(1);
      // Original severity carried on the constraint for badge styling
      expect(v.infos[0].constraint.severity).toBe(sev);
    }
  });

  it('does NOT fire when only 1 member is selected', () => {
    const v = checkComboConstraints(new Set(['H1']), [conflictsConstraint()]);
    expect(v.infos).toEqual([]);
    expect(v.errors).toEqual([]);
  });

  it('does NOT fire when 0 members are selected', () => {
    const v = checkComboConstraints(new Set(['H99']), [conflictsConstraint()]);
    expect(v.infos).toEqual([]);
  });

  it('carries the reason/label as hint for display', () => {
    const c = conflictsConstraint();
    c.label = 'File conflict';
    const v = checkComboConstraints(new Set(['H1', 'H2']), [c]);
    expect(v.infos[0].hint).toBe('File conflict');
  });
});

describe('checkComboConstraints — existing kinds unchanged (regression)', () => {
  it('requires with missing deps → errors (default severity)', () => {
    const v = checkComboConstraints(
      new Set(['H1']),
      [{ id: 'c', kind: 'requires', hypothesis_ids: ['H1'], requires_ids: ['H2', 'H3'] }],
    );
    expect(v.errors).toHaveLength(1);
    expect(v.errors[0].missing).toEqual(['H2', 'H3']);
  });

  it('recommends with missing hint → infos', () => {
    const v = checkComboConstraints(
      new Set(['H1']),
      [{ id: 'c', kind: 'recommends', hypothesis_ids: ['H1'], requires_ids: ['H2'], label: 'try H2' }],
    );
    expect(v.infos).toHaveLength(1);
    expect(v.infos[0].hint).toBe('try H2');
  });

  it('mutually_exclusive with 2 selected → errors', () => {
    const v = checkComboConstraints(
      new Set(['H1', 'H2']),
      [{ id: 'c', kind: 'mutually_exclusive', hypothesis_ids: ['H1', 'H2', 'H3'] }],
    );
    expect(v.errors).toHaveLength(1);
    expect(v.errors[0].conflicting).toEqual(['H1', 'H2']);
  });

  it('handles empty / null constraints list', () => {
    expect(checkComboConstraints(new Set(['H1']), [])).toEqual({
      errors: [], warnings: [], infos: [],
    });
    expect(checkComboConstraints(new Set(['H1']), null)).toEqual({
      errors: [], warnings: [], infos: [],
    });
  });

  it('accepts an Array as selectedIds (not just Set)', () => {
    const v = checkComboConstraints(
      ['H1', 'H2'],
      [{ id: 'c', kind: 'mutually_exclusive', hypothesis_ids: ['H1', 'H2'] }],
    );
    expect(v.errors).toHaveLength(1);
  });
});

describe('computeBlockedIds — defensive on missing / empty slots (regression)', () => {
  it('proposals with no slots array → no blocks', () => {
    const { blocked } = computeBlockedIds(
      new Set(['H1']),
      [{ id: 'H1' }, { id: 'H2' }],
    );
    expect(blocked.size).toBe(0);
  });

  it('empty proposals array → no blocks / no includedBy', () => {
    const { blocked, includedBy } = computeBlockedIds(new Set(), []);
    expect(blocked.size).toBe(0);
    expect(includedBy.size).toBe(0);
  });

  it('null / undefined allProposals → no throw, empty result', () => {
    const { blocked, includedBy } = computeBlockedIds(new Set(), null);
    expect(blocked.size).toBe(0);
    expect(includedBy.size).toBe(0);
  });

  it('slot conflict blocks a non-selected proposal', () => {
    const { blocked } = computeBlockedIds(
      new Set(['H1']),
      [
        { id: 'H1', slots: ['attention'] },
        { id: 'H2', slots: ['attention'] },
        { id: 'H3', slots: ['sequence'] },
      ],
    );
    expect(blocked.has('H2')).toBe(true);
    expect(blocked.has('H3')).toBe(false);
  });

  it('submitted proposals occupy slots too (via caller passing them as selected)', () => {
    // Widget passes new Set([...selected, ...submitted]) as selectedIds so
    // submitted proposals also occupy slots — that's the incremental-submit x
    // ME contract. This test asserts the helper honors that contract.
    const { blocked } = computeBlockedIds(
      new Set(['H1' /* selected */, 'H_sub' /* submitted */]),
      [
        { id: 'H1', slots: ['a'] },
        { id: 'H_sub', slots: ['b'] },
        { id: 'H2', slots: ['b'] }, // conflicts with submitted
      ],
    );
    expect(blocked.has('H2')).toBe(true);
  });

  it('transitive includes: parent-selected → child slot occupied', () => {
    const { blocked, includedBy } = computeBlockedIds(
      new Set(['C1']),
      [
        { id: 'C1', includes: ['H1'], slots: [] },
        { id: 'H1', slots: ['x'] }, // becomes occupied because C1 includes it
        { id: 'H2', slots: ['x'] }, // conflicts
      ],
    );
    expect(includedBy.get('H1')).toBe('C1');
    expect(blocked.has('H2')).toBe(true);
  });
});
