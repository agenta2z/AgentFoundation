/**
 * Atlassian Theme Definition
 *
 * Inspired by the Atlassian Design System (https://atlassian.design/).
 * Light mode with B400 Pacific Bridge (#0052CC) primary, Squid Ink
 * neutrals, sharp 3px corners, and system UI font stack.
 */

const atlassian = {
  id: 'atlassian',
  name: 'Atlassian',

  palette: {
    mode: 'light',
    primary:    { main: '#0052CC', light: '#4C9AFF', dark: '#0747A6' },
    secondary:  { main: '#6554C0', light: '#998DD9', dark: '#403294' },
    background: { default: '#FAFBFC', paper: '#FFFFFF' },
    text:       { primary: '#172B4D', secondary: '#6B778C' },
    success:    { main: '#36B37E', light: '#79F2C0', dark: '#006644' },
    warning:    { main: '#FFAB00', light: '#FFF0B3', dark: '#FF8B00' },
    error:      { main: '#FF5630', light: '#FFBDAD', dark: '#BF2600' },
    info:       { main: '#00B8D9', light: '#B3F5FF', dark: '#008DA6' },
    neutral:    { main: '#6B778C', light: '#B3BAC5', dark: '#42526E' },
    divider:    'rgba(9, 30, 66, 0.14)',
  },

  categorical: [
    '#0052CC', '#6554C0', '#00B8D9', '#FF7452',
    '#36B37E', '#FFAB00', '#FF5630', '#E56910',
  ],

  surfaces: {
    cardBg:              '#FFFFFF',
    cardBorder:          'rgba(9, 30, 66, 0.13)',
    hoverBg:             'rgba(9, 30, 66, 0.04)',
    overlayLight:        'rgba(9, 30, 66, 0.02)',
    overlayMedium:       'rgba(9, 30, 66, 0.06)',
    overlayActive:       'rgba(9, 30, 66, 0.08)',
    inputBg:             '#FAFBFC',
    inputBorder:         'rgba(9, 30, 66, 0.14)',
    inputBorderHover:    'rgba(9, 30, 66, 0.36)',
    scrollbarTrack:      'rgba(9, 30, 66, 0.08)',
    scrollbarThumb:      'rgba(9, 30, 66, 0.2)',
    scrollbarThumbHover: 'rgba(9, 30, 66, 0.36)',
    sidebarBg:           '#F4F5F7',
    activeHighlight:     'rgba(0, 82, 204, 0.1)',
    highlight:           'rgba(0, 82, 204, 0.07)',
    highlightSubtle:     'rgba(0, 82, 204, 0.04)',
    highlightBorder:     'rgba(0, 82, 204, 0.3)',
    scrim:               'rgba(9, 30, 66, 0.54)',
    mutedText:           'rgba(9, 30, 66, 0.4)',
    shimmerFrom:         'rgba(9, 30, 66, 0.04)',
    shimmerMid:          'rgba(9, 30, 66, 0.08)',
  },

  typography: {
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans', Ubuntu, 'Helvetica Neue', sans-serif",
    h1: { fontSize: '2rem', fontWeight: 600 },
    h2: { fontSize: '1.5rem', fontWeight: 600 },
    h3: { fontSize: '1.25rem', fontWeight: 500 },
    body1: { fontSize: '0.95rem', lineHeight: 1.6 },
    body2: { fontSize: '0.875rem', lineHeight: 1.5 },
  },

  components: {
    MuiButton: {
      styleOverrides: {
        root: { textTransform: 'none', borderRadius: 3, fontWeight: 500 },
        contained: { boxShadow: 'none', '&:hover': { boxShadow: 'none' } },
      },
    },
    MuiPaper: { styleOverrides: { root: { backgroundImage: 'none' } } },
    MuiTextField: {
      styleOverrides: {
        root: { '& .MuiOutlinedInput-root': { borderRadius: 3 } },
      },
    },
    MuiChip: { styleOverrides: { root: { borderRadius: 3 } } },
  },
};

export default atlassian;
