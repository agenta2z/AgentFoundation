/**
 * CSS Variable Bridge
 *
 * Injects CSS custom properties on document.documentElement for the given
 * ThemeDefinition. Called by ThemeProvider whenever the active theme changes.
 *
 * Variables injected:
 *   Palette:     --theme-primary-main, --theme-primary-light, --theme-primary-dark,
 *                --theme-secondary-main, --theme-bg-default, --theme-bg-paper,
 *                --theme-text-primary, --theme-text-secondary,
 *                --theme-success-main, --theme-warning-main, --theme-error-main,
 *                --theme-info-main, --theme-neutral-main, --theme-neutral-light,
 *                --theme-neutral-dark, --theme-divider
 *   Categorical: --theme-categorical-0 through --theme-categorical-7
 *   Surfaces:    --theme-scrollbar-track, --theme-scrollbar-thumb,
 *                --theme-scrollbar-thumb-hover, --theme-surface-card-bg,
 *                --theme-surface-card-border, --theme-surface-overlay-light,
 *                --theme-surface-overlay-medium, --theme-input-bg,
 *                --theme-input-border, --theme-shimmer-from, --theme-shimmer-mid,
 *                --theme-surface-hover-bg, --theme-surface-sidebar-bg,
 *                --theme-surface-active-highlight, --theme-surface-highlight,
 *                --theme-surface-highlight-subtle, --theme-surface-highlight-border,
 *                --theme-surface-scrim, --theme-surface-muted-text,
 *                --theme-surface-overlay-active, --theme-surface-input-border-hover
 */
export function applyCssVariables(definition) {
  if (typeof document === 'undefined') return;

  const style = document.documentElement.style;
  const { palette, categorical, surfaces } = definition;

  // Palette tokens
  style.setProperty('--theme-primary-main', palette.primary.main);
  style.setProperty('--theme-primary-light', palette.primary.light);
  style.setProperty('--theme-primary-dark', palette.primary.dark);
  style.setProperty('--theme-secondary-main', palette.secondary.main);
  style.setProperty('--theme-bg-default', palette.background.default);
  style.setProperty('--theme-bg-paper', palette.background.paper);
  style.setProperty('--theme-text-primary', palette.text.primary);
  style.setProperty('--theme-text-secondary', palette.text.secondary);
  style.setProperty('--theme-success-main', palette.success.main);
  style.setProperty('--theme-warning-main', palette.warning.main);
  style.setProperty('--theme-error-main', palette.error.main);
  style.setProperty('--theme-info-main', palette.info.main);
  style.setProperty('--theme-neutral-main', palette.neutral.main);
  style.setProperty('--theme-neutral-light', palette.neutral.light);
  style.setProperty('--theme-neutral-dark', palette.neutral.dark);
  style.setProperty('--theme-divider', palette.divider);

  // Categorical tokens
  if (categorical) {
    for (let i = 0; i < categorical.length; i++) {
      style.setProperty(`--theme-categorical-${i}`, categorical[i]);
    }
  }

  // Surface tokens
  if (surfaces) {
    style.setProperty('--theme-scrollbar-track', surfaces.scrollbarTrack);
    style.setProperty('--theme-scrollbar-thumb', surfaces.scrollbarThumb);
    style.setProperty('--theme-scrollbar-thumb-hover', surfaces.scrollbarThumbHover);
    style.setProperty('--theme-surface-card-bg', surfaces.cardBg);
    style.setProperty('--theme-surface-card-border', surfaces.cardBorder);
    style.setProperty('--theme-surface-overlay-light', surfaces.overlayLight);
    style.setProperty('--theme-surface-overlay-medium', surfaces.overlayMedium);
    style.setProperty('--theme-input-bg', surfaces.inputBg);
    style.setProperty('--theme-input-border', surfaces.inputBorder);
    style.setProperty('--theme-shimmer-from', surfaces.shimmerFrom);
    style.setProperty('--theme-shimmer-mid', surfaces.shimmerMid);
    style.setProperty('--theme-surface-hover-bg', surfaces.hoverBg);
    style.setProperty('--theme-surface-sidebar-bg', surfaces.sidebarBg);
    style.setProperty('--theme-surface-active-highlight', surfaces.activeHighlight);
    style.setProperty('--theme-surface-highlight', surfaces.highlight);
    style.setProperty('--theme-surface-highlight-subtle', surfaces.highlightSubtle);
    style.setProperty('--theme-surface-highlight-border', surfaces.highlightBorder);
    style.setProperty('--theme-surface-scrim', surfaces.scrim);
    style.setProperty('--theme-surface-muted-text', surfaces.mutedText);
    style.setProperty('--theme-surface-overlay-active', surfaces.overlayActive);
    style.setProperty('--theme-surface-input-border-hover', surfaces.inputBorderHover);
  }
}
