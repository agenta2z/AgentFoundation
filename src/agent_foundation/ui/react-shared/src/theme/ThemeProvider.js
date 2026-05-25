/**
 * ThemeProvider — React context provider with localStorage persistence
 *
 * Wraps MUI's ThemeProvider, injects CSS custom properties via the
 * CSS Variable Bridge, and exposes switchTheme / useAppTheme for
 * runtime theme selection.
 */
import React, {
  createContext,
  useContext,
  useState,
  useLayoutEffect,
  useMemo,
  useCallback,
} from 'react';
import { ThemeProvider as MuiThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { getTheme } from './themeRegistry';
import { createAppTheme } from './createAppTheme';
import { applyCssVariables } from './cssVariableBridge';

const ThemeContext = createContext();

const STORAGE_KEY = 'app-theme-id';

/**
 * Read the persisted theme ID from localStorage.
 * Returns the stored string, or null if absent / localStorage throws.
 */
function readStoredThemeId() {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

/**
 * AppThemeProvider
 *
 * @param {Object}   props
 * @param {React.ReactNode} props.children
 * @param {Function} props.createThemeFn - MUI's createTheme (v5 or v7)
 * @param {string}   [props.defaultThemeId='dark']
 */
export function AppThemeProvider({ children, createThemeFn, defaultThemeId = 'dark' }) {
  const [themeId, setThemeId] = useState(() => {
    const stored = readStoredThemeId();
    return stored || defaultThemeId;
  });

  const definition = useMemo(() => getTheme(themeId), [themeId]);
  const muiTheme = useMemo(
    () => createAppTheme(definition, createThemeFn),
    [definition, createThemeFn],
  );

  // useLayoutEffect ensures CSS variables are set BEFORE the browser paints,
  // preventing a one-frame flash of unstyled CSS-variable-dependent elements.
  useLayoutEffect(() => {
    applyCssVariables(definition);
  }, [definition]);

  const switchTheme = useCallback((newId) => {
    // Validate against registry before persisting to avoid garbage in localStorage
    const resolved = getTheme(newId); // falls back to 'dark' if invalid
    setThemeId(resolved.id);
    try {
      localStorage.setItem(STORAGE_KEY, resolved.id);
    } catch {
      // silent — quota exceeded or private browsing
    }
  }, []);

  return (
    <ThemeContext.Provider value={{ themeId, switchTheme, definition }}>
      <MuiThemeProvider theme={muiTheme}>
        <CssBaseline />
        {children}
      </MuiThemeProvider>
    </ThemeContext.Provider>
  );
}

/**
 * useAppTheme — access the current theme context.
 *
 * @returns {{ themeId: string, switchTheme: (id: string) => void, definition: Object }}
 */
export function useAppTheme() {
  return useContext(ThemeContext);
}
