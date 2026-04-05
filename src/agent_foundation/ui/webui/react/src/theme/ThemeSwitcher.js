/**
 * ThemeSwitcher — compact theme selector, usable standalone or inside a settings panel.
 *
 * Variants:
 *   - 'select' (default): MUI Select dropdown — compact, for headers/toolbars
 *   - 'cards': Visual theme cards with color swatches — for settings panels
 *
 * Calls switchTheme(newId) on selection — theme applies instantly, no page reload.
 */
import React from 'react';
import {
  Select,
  MenuItem,
  FormControl,
  Box,
  Typography,
} from '@mui/material';
import { useAppTheme } from './ThemeProvider';
import { listThemes, getTheme } from './themeRegistry';

/**
 * @param {Object} props
 * @param {'select'|'cards'} [props.variant='select']
 */
export function ThemeSwitcher({ variant = 'select' }) {
  const { themeId, switchTheme } = useAppTheme();
  const themeIds = listThemes();

  if (variant === 'cards') {
    return (
      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        {themeIds.map((id) => {
          const def = getTheme(id);
          const isActive = id === themeId;
          return (
            <Box
              key={id}
              onClick={() => switchTheme(id)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  switchTheme(id);
                }
              }}
              aria-label={`Select ${def.name} theme`}
              aria-pressed={isActive}
              sx={{
                cursor: 'pointer',
                border: isActive ? 2 : 1,
                borderColor: isActive ? 'primary.main' : 'divider',
                borderRadius: 2,
                p: 1.5,
                minWidth: 100,
                textAlign: 'center',
                transition: 'border-color 0.2s',
                '&:hover': { borderColor: 'primary.light' },
              }}
            >
              <Box
                sx={{
                  width: 48,
                  height: 48,
                  borderRadius: 1,
                  bgcolor: def.palette.primary.main,
                  mx: 'auto',
                  mb: 1,
                }}
              />
              <Typography variant="body2" noWrap>
                {def.name}
              </Typography>
              {isActive && (
                <Typography variant="caption" color="primary">
                  ✓
                </Typography>
              )}
            </Box>
          );
        })}
      </Box>
    );
  }

  // Default: variant='select'
  return (
    <FormControl size="small" sx={{ minWidth: 120 }}>
      <Select
        value={themeId}
        onChange={(e) => switchTheme(e.target.value)}
        aria-label="Theme selector"
        sx={{ fontSize: '0.875rem' }}
      >
        {themeIds.map((id) => {
          const def = getTheme(id);
          return (
            <MenuItem key={id} value={id}>
              {def.name}
            </MenuItem>
          );
        })}
      </Select>
    </FormControl>
  );
}
