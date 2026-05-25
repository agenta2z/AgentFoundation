/**
 * Pinterest Theme Definition
 *
 * Inspired by Pinterest's brand identity (https://business.pinterest.com/brand-guidelines/).
 * Light mode with Pinterest Red (#E60023) primary, pure white backgrounds,
 * warm neutral text, and distinctive pill-shaped components (24px button radius).
 */

const pinterest = {
  id: 'pinterest',
  name: 'Pinterest',

  palette: {
    mode: 'light',
    primary:    { main: '#E60023', light: '#FF5247', dark: '#AD081B' },
    secondary:  { main: '#5F5F5F', light: '#8E8E8E', dark: '#3B3B3B' },
    background: { default: '#FFFFFF', paper: '#FFFFFF' },
    text:       { primary: '#333333', secondary: '#767676' },
    success:    { main: '#0FA573', light: '#5EDBA8', dark: '#0A7D56' },
    warning:    { main: '#EFB118', light: '#F5CD47', dark: '#BD5B00' },
    error:      { main: '#E60023', light: '#FF5247', dark: '#CC0000' },
    info:       { main: '#0076D3', light: '#4DA3E8', dark: '#005A9E' },
    neutral:    { main: '#767676', light: '#ABABAB', dark: '#5F5F5F' },
    divider:    'rgba(0, 0, 0, 0.1)',
  },

  categorical: [
    '#0076D3', '#8046BC', '#0FA573', '#E60023',
    '#BD5B00', '#5F5F5F', '#0A7D56', '#AD081B',
  ],

  surfaces: {
    cardBg:              '#FFFFFF',
    cardBorder:          'rgba(0, 0, 0, 0.08)',
    hoverBg:             'rgba(0, 0, 0, 0.03)',
    overlayLight:        'rgba(0, 0, 0, 0.02)',
    overlayMedium:       'rgba(0, 0, 0, 0.05)',
    overlayActive:       'rgba(0, 0, 0, 0.07)',
    inputBg:             '#EFEFEF',
    inputBorder:         'rgba(0, 0, 0, 0.15)',
    inputBorderHover:    'rgba(0, 0, 0, 0.35)',
    scrollbarTrack:      'rgba(0, 0, 0, 0.04)',
    scrollbarThumb:      'rgba(0, 0, 0, 0.15)',
    scrollbarThumbHover: 'rgba(0, 0, 0, 0.25)',
    sidebarBg:           '#F7F7F7',
    activeHighlight:     'rgba(230, 0, 35, 0.08)',
    highlight:           'rgba(230, 0, 35, 0.06)',
    highlightSubtle:     'rgba(230, 0, 35, 0.03)',
    highlightBorder:     'rgba(230, 0, 35, 0.2)',
    scrim:               'rgba(0, 0, 0, 0.5)',
    mutedText:           'rgba(0, 0, 0, 0.45)',
    shimmerFrom:         'rgba(0, 0, 0, 0.03)',
    shimmerMid:          'rgba(0, 0, 0, 0.06)',
  },

  typography: {
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif",
    h1: { fontSize: '2rem', fontWeight: 700 },
    h2: { fontSize: '1.5rem', fontWeight: 600 },
    h3: { fontSize: '1.25rem', fontWeight: 600 },
    body1: { fontSize: '0.95rem', lineHeight: 1.6 },
    body2: { fontSize: '0.875rem', lineHeight: 1.5 },
  },

  components: {
    MuiButton: {
      styleOverrides: {
        root: { textTransform: 'none', borderRadius: 24, fontWeight: 600 },
        contained: { boxShadow: 'none', '&:hover': { boxShadow: 'none' } },
      },
    },
    MuiPaper: { styleOverrides: { root: { backgroundImage: 'none' } } },
    MuiTextField: {
      styleOverrides: {
        root: { '& .MuiOutlinedInput-root': { borderRadius: 16 } },
      },
    },
    MuiChip: { styleOverrides: { root: { borderRadius: 16 } } },
  },
};

export default pinterest;
