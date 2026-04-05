/**
 * Dark Theme Definition
 *
 * Default theme that reproduces the current AgentFoundation/OpenStartup
 * dark palette pixel-for-pixel. All values extracted from the existing
 * theme.js and hardcoded component colors.
 */

const dark = {
  id: 'dark',
  name: 'Dark',

  palette: {
    mode: 'dark',
    primary:    { main: '#4a90d9', light: '#7bb3eb', dark: '#2d6aa8' },
    secondary:  { main: '#7c4dff', light: '#b47cff', dark: '#5c35b0' },
    background: { default: '#1a1a2e', paper: '#16213e' },
    text:       { primary: '#e0e0e0', secondary: '#a0a0a0' },
    success:    { main: '#4caf50', light: '#81c784', dark: '#388e3c' },
    warning:    { main: '#ff9800', light: '#ffb74d', dark: '#f57c00' },
    error:      { main: '#f44336', light: '#ef9a9a', dark: '#d32f2f' },
    info:       { main: '#29b6f6', light: '#4fc3f7', dark: '#0288d1' },
    neutral:    { main: '#90a4ae', light: '#b0bec5', dark: '#607d8b' },
    divider:    'rgba(255, 255, 255, 0.1)',
  },

  categorical: [
    '#4a90d9', '#7c4dff', '#00bcd4', '#ff7043',
    '#4caf50', '#ff9800', '#e91e63', '#9c27b0',
  ],

  surfaces: {
    cardBg:              'rgba(255, 255, 255, 0.03)',
    cardBorder:          'rgba(255, 255, 255, 0.06)',
    hoverBg:             'rgba(255, 255, 255, 0.06)',
    overlayLight:        'rgba(255, 255, 255, 0.04)',
    overlayMedium:       'rgba(255, 255, 255, 0.08)',
    overlayActive:       'rgba(255, 255, 255, 0.1)',
    inputBg:             'rgba(255, 255, 255, 0.05)',
    inputBorder:         'rgba(255, 255, 255, 0.15)',
    inputBorderHover:    'rgba(255, 255, 255, 0.3)',
    scrollbarTrack:      'rgba(255, 255, 255, 0.05)',
    scrollbarThumb:      'rgba(255, 255, 255, 0.2)',
    scrollbarThumbHover: 'rgba(255, 255, 255, 0.3)',
    sidebarBg:           'rgba(0, 0, 0, 0.25)',
    activeHighlight:     'rgba(74, 144, 217, 0.15)',
    highlight:           'rgba(74, 144, 217, 0.1)',
    highlightSubtle:     'rgba(74, 144, 217, 0.08)',
    highlightBorder:     'rgba(74, 144, 217, 0.3)',
    scrim:               'rgba(0, 0, 0, 0.4)',
    mutedText:           'rgba(255, 255, 255, 0.5)',
    shimmerFrom:         'rgba(255, 255, 255, 0.05)',
    shimmerMid:          'rgba(255, 255, 255, 0.1)',
  },

  typography: {
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    h1: { fontSize: '2rem', fontWeight: 600 },
    h2: { fontSize: '1.5rem', fontWeight: 600 },
    h3: { fontSize: '1.25rem', fontWeight: 600 },
    body1: { fontSize: '0.95rem', lineHeight: 1.6 },
    body2: { fontSize: '0.875rem', lineHeight: 1.5 },
  },

  components: {
    MuiButton: {
      styleOverrides: {
        root: { textTransform: 'none', borderRadius: 8, fontWeight: 500 },
        contained: { boxShadow: 'none', '&:hover': { boxShadow: 'none' } },
      },
    },
    MuiPaper: { styleOverrides: { root: { backgroundImage: 'none' } } },
    MuiTextField: {
      styleOverrides: {
        root: { '& .MuiOutlinedInput-root': { borderRadius: 8 } },
      },
    },
    MuiChip: { styleOverrides: { root: { borderRadius: 6 } } },
  },
};

export default dark;
