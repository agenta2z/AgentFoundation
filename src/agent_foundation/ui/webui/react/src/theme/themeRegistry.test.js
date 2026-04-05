import { getTheme, listThemes, registerTheme, mergeTheme } from './themeRegistry';

describe('themeRegistry', () => {
  describe('listThemes', () => {
    it('returns the three built-in theme IDs', () => {
      const ids = listThemes();
      expect(ids).toContain('dark');
      expect(ids).toContain('atlassian');
      expect(ids).toContain('pinterest');
      expect(ids.length).toBeGreaterThanOrEqual(3);
    });
  });

  describe('getTheme', () => {
    it('returns the dark theme by ID', () => {
      const theme = getTheme('dark');
      expect(theme.id).toBe('dark');
      expect(theme.palette.primary.main).toBe('#4a90d9');
    });

    it('returns the atlassian theme by ID', () => {
      const theme = getTheme('atlassian');
      expect(theme.id).toBe('atlassian');
      expect(theme.palette.primary.main).toBe('#0052CC');
    });

    it('returns the pinterest theme by ID', () => {
      const theme = getTheme('pinterest');
      expect(theme.id).toBe('pinterest');
      expect(theme.palette.primary.main).toBe('#E60023');
    });

    it('falls back to dark for unknown IDs', () => {
      const theme = getTheme('nonexistent-theme');
      expect(theme.id).toBe('dark');
    });

    it('resolves extends chain via deep merge', () => {
      registerTheme('custom-dark', {
        id: 'custom-dark',
        name: 'Custom Dark',
        extends: 'dark',
        palette: {
          primary: { main: '#ff0000' },
        },
      });

      const theme = getTheme('custom-dark');
      // Overridden value
      expect(theme.palette.primary.main).toBe('#ff0000');
      // Inherited values from dark
      expect(theme.palette.background.default).toBe('#1a1a2e');
      expect(theme.palette.secondary.main).toBe('#7c4dff');
      // Sibling keys within primary should be inherited
      expect(theme.palette.primary.light).toBe('#7bb3eb');
      expect(theme.palette.primary.dark).toBe('#2d6aa8');
    });

    it('detects circular extends chains', () => {
      registerTheme('theme-a', {
        id: 'theme-a',
        name: 'Theme A',
        extends: 'theme-b',
      });
      registerTheme('theme-b', {
        id: 'theme-b',
        name: 'Theme B',
        extends: 'theme-a',
      });

      expect(() => getTheme('theme-a')).toThrow(/[Cc]ircular/);
    });

    it('resolves multi-level extends chains', () => {
      registerTheme('level-1', {
        id: 'level-1',
        name: 'Level 1',
        extends: 'dark',
        palette: {
          primary: { main: '#111111' },
        },
      });
      registerTheme('level-2', {
        id: 'level-2',
        name: 'Level 2',
        extends: 'level-1',
        palette: {
          secondary: { main: '#222222' },
        },
      });

      const theme = getTheme('level-2');
      expect(theme.palette.primary.main).toBe('#111111');    // from level-1
      expect(theme.palette.secondary.main).toBe('#222222');  // from level-2
      expect(theme.palette.background.default).toBe('#1a1a2e'); // from dark
    });
  });

  describe('registerTheme', () => {
    it('adds a new theme that can be retrieved', () => {
      registerTheme('test-new', {
        id: 'test-new',
        name: 'Test New',
        palette: { mode: 'dark' },
      });
      expect(listThemes()).toContain('test-new');
      expect(getTheme('test-new').id).toBe('test-new');
    });

    it('replaces an existing theme', () => {
      registerTheme('replaceable', {
        id: 'replaceable',
        name: 'V1',
        palette: { mode: 'dark' },
      });
      registerTheme('replaceable', {
        id: 'replaceable',
        name: 'V2',
        palette: { mode: 'light' },
      });
      expect(getTheme('replaceable').name).toBe('V2');
      expect(getTheme('replaceable').palette.mode).toBe('light');
    });
  });

  describe('mergeTheme', () => {
    it('deep-merges nested objects', () => {
      const base = { palette: { primary: { main: '#aaa', light: '#bbb' }, text: { primary: '#ccc' } } };
      const overrides = { palette: { primary: { main: '#111' } } };
      const result = mergeTheme(base, overrides);

      expect(result.palette.primary.main).toBe('#111');
      expect(result.palette.primary.light).toBe('#bbb');
      expect(result.palette.text.primary).toBe('#ccc');
    });

    it('replaces arrays wholesale', () => {
      const base = { categorical: ['#a', '#b', '#c'] };
      const overrides = { categorical: ['#x', '#y'] };
      const result = mergeTheme(base, overrides);

      expect(result.categorical).toEqual(['#x', '#y']);
    });

    it('does not mutate inputs', () => {
      const base = { palette: { primary: { main: '#aaa' } } };
      const overrides = { palette: { primary: { main: '#bbb' } } };
      mergeTheme(base, overrides);

      expect(base.palette.primary.main).toBe('#aaa');
      expect(overrides.palette.primary.main).toBe('#bbb');
    });

    it('handles null/undefined gracefully', () => {
      const obj = { a: 1 };
      expect(mergeTheme(obj, null)).toBe(obj);
      expect(mergeTheme(obj, undefined)).toBe(obj);
      expect(mergeTheme(null, obj)).toBe(obj);
      expect(mergeTheme(undefined, obj)).toBe(obj);
    });
  });
});
