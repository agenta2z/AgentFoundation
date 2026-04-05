/**
 * createAppTheme — MUI version adapter
 *
 * Accepts a ThemeDefinition and the host app's createTheme function
 * (MUI 5 or MUI 7), producing a valid MUI theme object. Non-standard
 * tokens (categorical, surfaces, and any unrecognized keys) are placed
 * under `theme.custom` to avoid MUI console warnings.
 *
 * @param {Object} definition - A ThemeDefinition object
 * @param {Function} createThemeFn - MUI's createTheme (v5 or v7)
 * @returns {Object} MUI theme object with custom namespace
 */
export function createAppTheme(definition, createThemeFn) {
  const { palette, typography, components, categorical, surfaces, id, name, extends: _ext, ...rest } = definition;

  const muiTheme = createThemeFn({
    palette,
    typography,
    components,
  });

  // Attach non-standard tokens under custom namespace
  muiTheme.custom = {
    categorical,
    surfaces,
    ...rest,
  };

  return muiTheme;
}
