/**
 * theme/index.js — barrel export for the shared theme module.
 *
 * Public API surface for both AgentFoundation (MUI 5) and OpenStartup (MUI 7).
 */

export { AppThemeProvider, useAppTheme } from './ThemeProvider';
export { ThemeSwitcher } from './ThemeSwitcher';
export { createAppTheme } from './createAppTheme';
export { getTheme, listThemes, registerTheme, mergeTheme } from './themeRegistry';
export { applyCssVariables } from './cssVariableBridge';
