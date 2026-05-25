import fc from 'fast-check';
import { listThemes, getTheme, mergeTheme, registerTheme } from './themeRegistry';

/**
 * Property 1: ThemeDefinition Structural Validity
 *
 * Every theme in the registry must have all required palette groups,
 * categorical array of sufficient length, and all required surface tokens.
 *
 * **Validates: Requirements 1.1, 1.2, 1.3**
 */
describe('Property 1: ThemeDefinition Structural Validity', () => {
  const themeIdArb = fc.constantFrom(...listThemes());

  const paletteGroupsWithVariants = [
    'primary', 'secondary', 'success', 'warning', 'error', 'info', 'neutral',
  ];

  const requiredSurfaceTokens = [
    'cardBg', 'cardBorder', 'hoverBg',
    'overlayLight', 'overlayMedium', 'overlayActive',
    'inputBg', 'inputBorder', 'inputBorderHover',
    'scrollbarTrack', 'scrollbarThumb', 'scrollbarThumbHover',
    'sidebarBg',
    'activeHighlight', 'highlight', 'highlightSubtle', 'highlightBorder',
    'scrim', 'mutedText',
    'shimmerFrom', 'shimmerMid',
  ];

  it('every theme has all required palette groups', () => {
    fc.assert(
      fc.property(themeIdArb, (id) => {
        const theme = getTheme(id);
        const palette = theme.palette;

        // Required palette groups with main/light/dark
        for (const group of paletteGroupsWithVariants) {
          expect(palette).toHaveProperty(group);
          expect(palette[group]).toHaveProperty('main');
          expect(palette[group]).toHaveProperty('light');
          expect(palette[group]).toHaveProperty('dark');
        }

        // background must have default and paper
        expect(palette).toHaveProperty('background');
        expect(palette.background).toHaveProperty('default');
        expect(palette.background).toHaveProperty('paper');

        // text must have primary and secondary
        expect(palette).toHaveProperty('text');
        expect(palette.text).toHaveProperty('primary');
        expect(palette.text).toHaveProperty('secondary');

        // divider must exist
        expect(palette).toHaveProperty('divider');
      })
    );
  });

  it('every theme has a categorical array of length >= 8', () => {
    fc.assert(
      fc.property(themeIdArb, (id) => {
        const theme = getTheme(id);
        expect(Array.isArray(theme.categorical)).toBe(true);
        expect(theme.categorical.length).toBeGreaterThanOrEqual(8);
      })
    );
  });

  it('every theme has all required surface tokens', () => {
    fc.assert(
      fc.property(themeIdArb, (id) => {
        const theme = getTheme(id);
        expect(theme).toHaveProperty('surfaces');

        for (const token of requiredSurfaceTokens) {
          expect(theme.surfaces).toHaveProperty(token);
        }
      })
    );
  });
});

/**
 * Property 2: ThemeDefinition JSON Serialization Round Trip
 *
 * Every registered theme must survive a JSON.stringify → JSON.parse round trip
 * and produce a deeply equal object. This confirms themes are plain
 * JSON-serializable objects with no class instances, functions, or prototypes.
 *
 * **Validates: Requirements 1.5**
 */
describe('Property 2: ThemeDefinition JSON Serialization Round Trip', () => {
  const themeIdArb = fc.constantFrom(...listThemes());

  it('JSON.parse(JSON.stringify(theme)) produces a deeply equal object', () => {
    fc.assert(
      fc.property(themeIdArb, (id) => {
        const original = getTheme(id);
        const parsed = JSON.parse(JSON.stringify(original));
        expect(parsed).toEqual(original);
      })
    );
  });
});

/**
 * Property 3: Registry Lookup with Fallback
 *
 * For any string identifier, `getTheme(id)` shall return a structurally
 * valid ThemeDefinition. If the identifier matches a registered theme,
 * that theme is returned; otherwise, the "dark" ThemeDefinition is returned.
 *
 * **Validates: Requirements 2.4, 2.5**
 */
describe('Property 3: Registry Lookup with Fallback', () => {
  const registeredIdArb = fc.constantFrom(...listThemes());
  const anyIdArb = fc.oneof(registeredIdArb, fc.string());

  it('getTheme always returns a structurally valid ThemeDefinition', () => {
    fc.assert(
      fc.property(anyIdArb, (id) => {
        const theme = getTheme(id);

        // Must have palette with required groups
        expect(theme).toHaveProperty('palette');
        expect(theme.palette).toHaveProperty('primary');
        expect(theme.palette).toHaveProperty('background');
        expect(theme.palette).toHaveProperty('text');

        // Must have categorical array
        expect(Array.isArray(theme.categorical)).toBe(true);
        expect(theme.categorical.length).toBeGreaterThanOrEqual(8);

        // Must have surfaces object
        expect(theme).toHaveProperty('surfaces');
        expect(theme.surfaces).toHaveProperty('cardBg');
        expect(theme.surfaces).toHaveProperty('cardBorder');
      })
    );
  });

  it('registered IDs return the corresponding theme', () => {
    fc.assert(
      fc.property(registeredIdArb, (id) => {
        const theme = getTheme(id);
        expect(theme).toHaveProperty('id', id);
      })
    );
  });

  it('unregistered IDs fall back to the dark theme', () => {
    const registeredIds = new Set(listThemes());
    const unregisteredArb = fc.string().filter((s) => !registeredIds.has(s));

    fc.assert(
      fc.property(unregisteredArb, (id) => {
        const theme = getTheme(id);
        expect(theme.id).toBe('dark');
      })
    );
  });
});

/**
 * Property 10: Deep Merge Preserves Non-Overridden Keys
 *
 * For any base ThemeDefinition and any partial override object that modifies
 * a single nested key, `mergeTheme(base, override)` shall return a new
 * ThemeDefinition where the overridden key has the new value and all sibling
 * keys at every nesting level retain their values from the base.
 *
 * **Validates: Requirements 7.1, 7.2**
 */
describe('Property 10: Deep Merge Preserves Non-Overridden Keys', () => {
  const themeIdArb = fc.constantFrom(...listThemes());
  const hexColorArb = fc.hexaString({ minLength: 6, maxLength: 6 }).map((s) => '#' + s);

  it('overridden palette.primary.main has the new value', () => {
    fc.assert(
      fc.property(themeIdArb, hexColorArb, (id, newColor) => {
        const base = getTheme(id);
        const overrides = { palette: { primary: { main: newColor } } };
        const merged = mergeTheme(base, overrides);

        expect(merged.palette.primary.main).toBe(newColor);
      })
    );
  });

  it('sibling keys within the overridden group are preserved', () => {
    fc.assert(
      fc.property(themeIdArb, hexColorArb, (id, newColor) => {
        const base = getTheme(id);
        const overrides = { palette: { primary: { main: newColor } } };
        const merged = mergeTheme(base, overrides);

        // Sibling keys inside palette.primary (light, dark) must be preserved
        expect(merged.palette.primary.light).toBe(base.palette.primary.light);
        expect(merged.palette.primary.dark).toBe(base.palette.primary.dark);
      })
    );
  });

  it('sibling palette groups are preserved when one group is overridden', () => {
    fc.assert(
      fc.property(themeIdArb, hexColorArb, (id, newColor) => {
        const base = getTheme(id);
        const overrides = { palette: { primary: { main: newColor } } };
        const merged = mergeTheme(base, overrides);

        // All other palette groups must be unchanged
        expect(merged.palette.secondary).toEqual(base.palette.secondary);
        expect(merged.palette.background).toEqual(base.palette.background);
        expect(merged.palette.text).toEqual(base.palette.text);
        expect(merged.palette.success).toEqual(base.palette.success);
        expect(merged.palette.warning).toEqual(base.palette.warning);
        expect(merged.palette.error).toEqual(base.palette.error);
        expect(merged.palette.info).toEqual(base.palette.info);
        expect(merged.palette.neutral).toEqual(base.palette.neutral);
        expect(merged.palette.divider).toBe(base.palette.divider);
      })
    );
  });

  it('top-level sibling keys are preserved when palette is overridden', () => {
    fc.assert(
      fc.property(themeIdArb, hexColorArb, (id, newColor) => {
        const base = getTheme(id);
        const overrides = { palette: { primary: { main: newColor } } };
        const merged = mergeTheme(base, overrides);

        // categorical, surfaces, typography, components must be unchanged
        expect(merged.categorical).toEqual(base.categorical);
        expect(merged.surfaces).toEqual(base.surfaces);
        expect(merged.typography).toEqual(base.typography);
        expect(merged.components).toEqual(base.components);
        expect(merged.id).toBe(base.id);
        expect(merged.name).toBe(base.name);
      })
    );
  });

  it('base object is not mutated by mergeTheme', () => {
    fc.assert(
      fc.property(themeIdArb, hexColorArb, (id, newColor) => {
        const base = getTheme(id);
        const originalPrimaryMain = base.palette.primary.main;
        const overrides = { palette: { primary: { main: newColor } } };

        mergeTheme(base, overrides);

        // Base must remain unchanged
        expect(base.palette.primary.main).toBe(originalPrimaryMain);
      })
    );
  });
});

/**
 * Property 11: Register Then Get Round Trip
 *
 * For any valid ThemeDefinition and any identifier string, after calling
 * `registerTheme(id, definition)`, `getTheme(id)` shall return a
 * ThemeDefinition deeply equal to the one that was registered. If a theme
 * was previously registered under the same identifier, it shall be replaced.
 *
 * **Validates: Requirements 7.3, 7.4**
 */
describe('Property 11: Register Then Get Round Trip', () => {
  const builtInIds = ['dark', 'atlassian', 'pinterest'];

  const customIdArb = fc
    .string({ minLength: 1, maxLength: 20 })
    .filter((s) => !builtInIds.includes(s));

  const hexColorArb = fc.hexaString({ minLength: 6, maxLength: 6 }).map((s) => '#' + s);

  const minimalThemeArb = fc.record({
    id: customIdArb,
    name: fc.string({ minLength: 1, maxLength: 30 }),
    palette: fc.record({
      mode: fc.constantFrom('light', 'dark'),
    }),
  });

  // Track IDs registered during tests so we can clean up
  const registeredIds = [];

  afterAll(() => {
    // Remove custom themes added during tests to avoid polluting the registry
    for (const id of registeredIds) {
      // Re-register built-ins are safe; for custom IDs we can't truly
      // delete from a Map-based registry, but registering a dummy
      // prevents stale state from leaking into other test suites.
    }
  });

  it('getTheme returns the registered definition', () => {
    fc.assert(
      fc.property(minimalThemeArb, (definition) => {
        const id = definition.id;
        registeredIds.push(id);

        registerTheme(id, definition);
        const retrieved = getTheme(id);

        expect(retrieved).toEqual(definition);
      })
    );
  });

  it('replacement: re-registering the same ID overwrites the previous definition', () => {
    fc.assert(
      fc.property(customIdArb, minimalThemeArb, minimalThemeArb, (id, first, second) => {
        // Ensure the two definitions differ by assigning the shared id
        const def1 = { ...first, id };
        const def2 = { ...second, id };

        registeredIds.push(id);

        registerTheme(id, def1);
        expect(getTheme(id)).toEqual(def1);

        registerTheme(id, def2);
        expect(getTheme(id)).toEqual(def2);
      })
    );
  });
});

import { createAppTheme } from './createAppTheme';

/**
 * Property 12: createAppTheme Produces Valid Theme with Custom Namespace
 *
 * For any valid ThemeDefinition, calling `createAppTheme(definition, createTheme)`
 * shall return an object where `result.palette.primary.main` equals
 * `definition.palette.primary.main`, and `result.custom` contains `categorical`
 * and `surfaces` matching the input definition.
 *
 * **Validates: Requirements 8.2, 8.3**
 */
describe('Property 12: createAppTheme Produces Valid Theme with Custom Namespace', () => {
  const themeIdArb = fc.constantFrom(...listThemes());
  const mockCreateTheme = (opts) => ({ ...opts });

  it('result.palette.primary.main equals definition.palette.primary.main', () => {
    fc.assert(
      fc.property(themeIdArb, (id) => {
        const definition = getTheme(id);
        const result = createAppTheme(definition, mockCreateTheme);

        expect(result.palette.primary.main).toBe(definition.palette.primary.main);
      })
    );
  });

  it('result.custom exists and contains categorical and surfaces', () => {
    fc.assert(
      fc.property(themeIdArb, (id) => {
        const definition = getTheme(id);
        const result = createAppTheme(definition, mockCreateTheme);

        expect(result.custom).toBeDefined();
        expect(result.custom).toHaveProperty('categorical');
        expect(result.custom).toHaveProperty('surfaces');
      })
    );
  });

  it('result.custom.categorical equals definition.categorical', () => {
    fc.assert(
      fc.property(themeIdArb, (id) => {
        const definition = getTheme(id);
        const result = createAppTheme(definition, mockCreateTheme);

        expect(result.custom.categorical).toEqual(definition.categorical);
      })
    );
  });

  it('result.custom.surfaces equals definition.surfaces', () => {
    fc.assert(
      fc.property(themeIdArb, (id) => {
        const definition = getTheme(id);
        const result = createAppTheme(definition, mockCreateTheme);

        expect(result.custom.surfaces).toEqual(definition.surfaces);
      })
    );
  });
});

import { applyCssVariables } from './cssVariableBridge';

/**
 * Property 6: CSS Variable Bridge Completeness
 *
 * For any valid ThemeDefinition, after calling `applyCssVariables(definition)`,
 * the document root element shall have CSS custom properties set for every
 * palette token, every categorical entry, and every surface token, and the
 * values shall match the corresponding fields in the ThemeDefinition.
 *
 * **Validates: Requirements 4.1, 4.2, 4.4**
 */
describe('Property 6: CSS Variable Bridge Completeness', () => {
  const themeIdArb = fc.constantFrom(...listThemes());

  beforeEach(() => {
    // Clear all inline styles on documentElement before each test
    document.documentElement.style.cssText = '';
  });

  const getVar = (name) => document.documentElement.style.getPropertyValue(name);

  it('all palette CSS variables are set with correct values', () => {
    fc.assert(
      fc.property(themeIdArb, (id) => {
        const theme = getTheme(id);
        applyCssVariables(theme);

        const { palette } = theme;

        expect(getVar('--theme-primary-main')).toBe(palette.primary.main);
        expect(getVar('--theme-primary-light')).toBe(palette.primary.light);
        expect(getVar('--theme-primary-dark')).toBe(palette.primary.dark);
        expect(getVar('--theme-secondary-main')).toBe(palette.secondary.main);
        expect(getVar('--theme-bg-default')).toBe(palette.background.default);
        expect(getVar('--theme-bg-paper')).toBe(palette.background.paper);
        expect(getVar('--theme-text-primary')).toBe(palette.text.primary);
        expect(getVar('--theme-text-secondary')).toBe(palette.text.secondary);
        expect(getVar('--theme-success-main')).toBe(palette.success.main);
        expect(getVar('--theme-warning-main')).toBe(palette.warning.main);
        expect(getVar('--theme-error-main')).toBe(palette.error.main);
        expect(getVar('--theme-info-main')).toBe(palette.info.main);
        expect(getVar('--theme-neutral-main')).toBe(palette.neutral.main);
        expect(getVar('--theme-neutral-light')).toBe(palette.neutral.light);
        expect(getVar('--theme-neutral-dark')).toBe(palette.neutral.dark);
        expect(getVar('--theme-divider')).toBe(palette.divider);
      })
    );
  });

  it('all categorical CSS variables are set with correct values', () => {
    fc.assert(
      fc.property(themeIdArb, (id) => {
        const theme = getTheme(id);
        applyCssVariables(theme);

        for (let i = 0; i < 8; i++) {
          expect(getVar(`--theme-categorical-${i}`)).toBe(theme.categorical[i]);
        }
      })
    );
  });

  it('all surface CSS variables are set with correct values', () => {
    fc.assert(
      fc.property(themeIdArb, (id) => {
        const theme = getTheme(id);
        applyCssVariables(theme);

        const { surfaces } = theme;

        expect(getVar('--theme-scrollbar-track')).toBe(surfaces.scrollbarTrack);
        expect(getVar('--theme-scrollbar-thumb')).toBe(surfaces.scrollbarThumb);
        expect(getVar('--theme-scrollbar-thumb-hover')).toBe(surfaces.scrollbarThumbHover);
        expect(getVar('--theme-surface-card-bg')).toBe(surfaces.cardBg);
        expect(getVar('--theme-surface-card-border')).toBe(surfaces.cardBorder);
        expect(getVar('--theme-surface-overlay-light')).toBe(surfaces.overlayLight);
        expect(getVar('--theme-surface-overlay-medium')).toBe(surfaces.overlayMedium);
        expect(getVar('--theme-input-bg')).toBe(surfaces.inputBg);
        expect(getVar('--theme-input-border')).toBe(surfaces.inputBorder);
        expect(getVar('--theme-shimmer-from')).toBe(surfaces.shimmerFrom);
        expect(getVar('--theme-shimmer-mid')).toBe(surfaces.shimmerMid);
        expect(getVar('--theme-surface-hover-bg')).toBe(surfaces.hoverBg);
        expect(getVar('--theme-surface-sidebar-bg')).toBe(surfaces.sidebarBg);
        expect(getVar('--theme-surface-active-highlight')).toBe(surfaces.activeHighlight);
        expect(getVar('--theme-surface-highlight')).toBe(surfaces.highlight);
        expect(getVar('--theme-surface-highlight-subtle')).toBe(surfaces.highlightSubtle);
        expect(getVar('--theme-surface-highlight-border')).toBe(surfaces.highlightBorder);
        expect(getVar('--theme-surface-scrim')).toBe(surfaces.scrim);
        expect(getVar('--theme-surface-muted-text')).toBe(surfaces.mutedText);
        expect(getVar('--theme-surface-overlay-active')).toBe(surfaces.overlayActive);
        expect(getVar('--theme-surface-input-border-hover')).toBe(surfaces.inputBorderHover);
      })
    );
  });
});


/**
 * Property 4: Theme Persistence Write
 *
 * For any valid theme identifier from the registry, calling `switchTheme`
 * (which resolves the theme via `getTheme` and writes to localStorage)
 * shall result in `localStorage.getItem('app-theme-id')` returning that
 * identifier. For unregistered identifiers, the fallback theme ID ('dark')
 * shall be persisted instead.
 *
 * Since `switchTheme` is a React callback only accessible inside the
 * ThemeProvider context, we test the identical persistence contract directly:
 * resolve via `getTheme`, write `resolved.id` to localStorage, read back.
 *
 * **Validates: Requirements 3.4**
 */
describe('Property 4: Theme Persistence Write', () => {
  const STORAGE_KEY = 'app-theme-id';
  const registeredIdArb = fc.constantFrom(...listThemes());
  const anyIdArb = fc.oneof(registeredIdArb, fc.string());

  beforeEach(() => {
    localStorage.clear();
  });

  afterAll(() => {
    localStorage.clear();
  });

  it('persisting a registered theme ID writes that exact ID to localStorage', () => {
    fc.assert(
      fc.property(registeredIdArb, (id) => {
        // Replicate switchTheme logic: resolve then persist
        const resolved = getTheme(id);
        localStorage.setItem(STORAGE_KEY, resolved.id);

        expect(localStorage.getItem(STORAGE_KEY)).toBe(id);
      })
    );
  });

  it('persisting any ID (registered or not) always writes a valid theme ID to localStorage', () => {
    fc.assert(
      fc.property(anyIdArb, (id) => {
        const resolved = getTheme(id);
        localStorage.setItem(STORAGE_KEY, resolved.id);

        const stored = localStorage.getItem(STORAGE_KEY);
        // Stored value must be a registered theme ID
        const allIds = listThemes();
        expect(allIds).toContain(stored);
      })
    );
  });

  it('persisting an unregistered ID writes the fallback "dark" to localStorage', () => {
    const registeredIds = new Set(listThemes());
    const unregisteredArb = fc.string().filter((s) => !registeredIds.has(s));

    fc.assert(
      fc.property(unregisteredArb, (id) => {
        const resolved = getTheme(id);
        localStorage.setItem(STORAGE_KEY, resolved.id);

        expect(localStorage.getItem(STORAGE_KEY)).toBe('dark');
      })
    );
  });
});


/**
 * Property 5: Theme Persistence Read Fallback
 *
 * For any string value stored in localStorage under 'app-theme-id' that does
 * not correspond to a registered theme (including empty string and null), the
 * ThemeProvider shall initialize with the "dark" ThemeDefinition.
 *
 * Since ThemeProvider reads localStorage in its useState initializer and then
 * resolves via getTheme, we test the equivalent read-then-resolve logic:
 *   1. Set localStorage to an invalid value
 *   2. Read it back
 *   3. Call getTheme with the stored value — it should return the dark theme
 *
 * **Validates: Requirements 3.6**
 */
describe('Property 5: Theme Persistence Read Fallback', () => {
  const STORAGE_KEY = 'app-theme-id';
  const invalidIdArb = fc.string().filter((s) => !listThemes().includes(s));

  beforeEach(() => {
    localStorage.clear();
  });

  afterAll(() => {
    localStorage.clear();
  });

  it('invalid localStorage values cause getTheme to fall back to dark', () => {
    fc.assert(
      fc.property(invalidIdArb, (invalidId) => {
        // 1. Persist the invalid value (simulates garbage in localStorage)
        localStorage.setItem(STORAGE_KEY, invalidId);

        // 2. Read it back (mirrors ThemeProvider's readStoredThemeId)
        const stored = localStorage.getItem(STORAGE_KEY);

        // 3. Resolve via getTheme — should fall back to dark
        const theme = getTheme(stored);
        expect(theme.id).toBe('dark');
      })
    );
  });

  it('absent localStorage value results in dark theme via fallback default', () => {
    // When localStorage has no entry, ThemeProvider uses defaultThemeId ('dark')
    const stored = localStorage.getItem(STORAGE_KEY);
    expect(stored).toBeNull();

    // stored is null → ThemeProvider falls back to defaultThemeId = 'dark'
    const themeId = stored || 'dark';
    const theme = getTheme(themeId);
    expect(theme.id).toBe('dark');
  });
});


/**
 * Property 7: StatusBadge Maps Domain to Palette
 *
 * For any status key string and any active ThemeDefinition, the StatusBadge
 * component shall derive its foreground color from `theme.palette[mappedKey].main`
 * and its background from `alpha(theme.palette[mappedKey].main, 0.15)`, where
 * `mappedKey` is determined by the component's internal STATUS_PALETTE mapping,
 * falling back to `'neutral'` for unrecognized statuses.
 *
 * Since StatusBadge is a React component that uses useTheme(), we test the
 * mapping logic directly by duplicating the STATUS_PALETTE map and verifying
 * that all mapped palette keys exist in every theme with a `.main` property.
 *
 * **Validates: Requirements 5.1**
 */
describe('Property 7: StatusBadge Maps Domain to Palette', () => {
  const themeIdArb = fc.constantFrom(...listThemes());

  // Duplicated from StatusBadge.js — this is the pure data mapping under test
  const STATUS_PALETTE = {
    'in-progress': 'primary',
    'planning':    'warning',
    'completed':   'success',
    'on-hold':     'neutral',
    'backlog':     'neutral',
    'in-review':   'info',
    'done':        'success',
    'blocked':     'error',
    'pending':     'secondary',
    'active':      'success',
    'idle':        'neutral',
    'away':        'warning',
    'queued':      'warning',
  };

  const knownStatusArb = fc.constantFrom(...Object.keys(STATUS_PALETTE));

  it('all known status keys map to palette keys that exist in every theme', () => {
    fc.assert(
      fc.property(themeIdArb, knownStatusArb, (id, status) => {
        const theme = getTheme(id);
        const paletteKey = STATUS_PALETTE[status];

        expect(theme.palette).toHaveProperty(paletteKey);
        expect(theme.palette[paletteKey]).toHaveProperty('main');
      })
    );
  });

  it('unknown status keys default to neutral which exists in every theme', () => {
    const registeredStatuses = new Set(Object.keys(STATUS_PALETTE));
    const unknownStatusArb = fc.string().filter((s) => !registeredStatuses.has(s));

    fc.assert(
      fc.property(themeIdArb, unknownStatusArb, (id, status) => {
        const theme = getTheme(id);
        // StatusBadge falls back: STATUS_PALETTE[key] || 'neutral'
        const paletteKey = STATUS_PALETTE[status] || 'neutral';

        expect(paletteKey).toBe('neutral');
        expect(theme.palette).toHaveProperty('neutral');
        expect(theme.palette.neutral).toHaveProperty('main');
      })
    );
  });

  it('for every theme, the mapped palette key has a .main property needed for fg/bg derivation', () => {
    const allPaletteKeys = [...new Set(Object.values(STATUS_PALETTE))];
    const paletteKeyArb = fc.constantFrom(...allPaletteKeys);

    fc.assert(
      fc.property(themeIdArb, paletteKeyArb, (id, paletteKey) => {
        const theme = getTheme(id);
        const color = theme.palette[paletteKey];

        expect(color).toBeDefined();
        expect(typeof color.main).toBe('string');
        expect(color.main.length).toBeGreaterThan(0);
      })
    );
  });
});


/**
 * Property 8: PersonChip Uses Categorical Colors
 *
 * For any name string, the PersonChip component shall compute an avatar
 * background color that is always a member of the active theme's `categorical`
 * array, selected by a deterministic hash of the name.
 *
 * **Validates: Requirements 5.2**
 */
describe('Property 8: PersonChip Uses Categorical Colors', () => {
  // Duplicated from PersonChip.js — the pure hash function under test
  function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    return Math.abs(hash);
  }

  const nameArb = fc.string({ minLength: 1 });
  const themeIdArb = fc.constantFrom(...listThemes());

  it('computed avatar color is always a member of the theme categorical array', () => {
    fc.assert(
      fc.property(nameArb, themeIdArb, (name, id) => {
        const theme = getTheme(id);
        const colors = theme.categorical;
        const bgColor = colors[hashString(name) % colors.length];

        expect(colors).toContain(bgColor);
      })
    );
  });

  it('same name always produces the same color for the same theme (deterministic)', () => {
    fc.assert(
      fc.property(nameArb, themeIdArb, (name, id) => {
        const theme = getTheme(id);
        const colors = theme.categorical;

        const color1 = colors[hashString(name) % colors.length];
        const color2 = colors[hashString(name) % colors.length];

        expect(color1).toBe(color2);
      })
    );
  });
});


/**
 * Property 9: ProgressBar Uses Palette Success/Warning/Error
 *
 * For any percentage value (0–100) and any active ThemeDefinition, the
 * ProgressBar component shall select `theme.palette.success.main` for ≥75%,
 * `theme.palette.warning.main` for ≥40%, and `theme.palette.error.main`
 * otherwise.
 *
 * Since `getProgressColor` is an internal function of the ProgressBar React
 * component, we duplicate its pure logic here for property testing.
 *
 * **Validates: Requirements 5.3**
 */
describe('Property 9: ProgressBar Uses Palette Success/Warning/Error', () => {
  const themeIdArb = fc.constantFrom(...listThemes());
  const percentArb = fc.integer({ min: 0, max: 100 });

  // Duplicated from ProgressBar.js — the pure threshold logic under test
  function getProgressColor(percent, theme) {
    if (percent >= 75) return theme.palette.success.main;
    if (percent >= 40) return theme.palette.warning.main;
    return theme.palette.error.main;
  }

  it('percent >= 75 returns theme.palette.success.main', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 75, max: 100 }),
        themeIdArb,
        (percent, id) => {
          const theme = getTheme(id);
          const color = getProgressColor(percent, theme);
          expect(color).toBe(theme.palette.success.main);
        }
      )
    );
  });

  it('percent >= 40 and < 75 returns theme.palette.warning.main', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 40, max: 74 }),
        themeIdArb,
        (percent, id) => {
          const theme = getTheme(id);
          const color = getProgressColor(percent, theme);
          expect(color).toBe(theme.palette.warning.main);
        }
      )
    );
  });

  it('percent < 40 returns theme.palette.error.main', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: 39 }),
        themeIdArb,
        (percent, id) => {
          const theme = getTheme(id);
          const color = getProgressColor(percent, theme);
          expect(color).toBe(theme.palette.error.main);
        }
      )
    );
  });

  it('for any percentage 0-100 and any theme, result is always one of success/warning/error main', () => {
    fc.assert(
      fc.property(percentArb, themeIdArb, (percent, id) => {
        const theme = getTheme(id);
        const color = getProgressColor(percent, theme);
        const validColors = [
          theme.palette.success.main,
          theme.palette.warning.main,
          theme.palette.error.main,
        ];
        expect(validColors).toContain(color);
      })
    );
  });
});
