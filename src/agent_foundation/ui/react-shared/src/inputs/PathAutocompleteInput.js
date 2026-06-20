/**
 * PathAutocompleteInput — controlled single-value path field.
 *
 * Built on MUI Autocomplete (freeSolo). The `prefix` (a concrete absolute path) is
 * shown as a non-editable start adornment; the user types the relative remainder.
 * Path suggestions are fetched through a host-injected provider (see §D7) — shared-ui
 * never inspects the filesystem itself.
 *
 *   - Selecting a directory suggestion drills deeper (appends a trailing "/" and re-queries).
 *   - Selecting a file commits that relative path.
 *   - onChange always receives the relative remainder string (never the prefix).
 *   - No provider → degrades to a plain TextField (still controlled, still relative).
 *
 * Props:
 *   value        - string (relative remainder)
 *   onChange     - (relativeString) => void
 *   prefix       - string (concrete absolute path shown as adornment)
 *   provider     - PathAutocompleteProvider | undefined
 *   dirsOnly     - bool
 *   placeholder  - string
 *   autoFocus    - bool
 *   disabled     - bool
 */

import React, { useEffect } from 'react';
import { Autocomplete, Box, TextField, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { usePathComplete } from '../hooks/usePathComplete';

function relativeOf(suggestion, prefix) {
  // Prefer deriving the relative remainder from the absolute path the provider returned.
  if (suggestion?.path && prefix && suggestion.path.startsWith(prefix)) {
    let rel = suggestion.path.slice(prefix.length);
    if (rel.startsWith('/')) rel = rel.slice(1);
    return rel;
  }
  return suggestion?.name || '';
}

const inputSx = (theme) => ({
  '& .MuiOutlinedInput-root': {
    backgroundColor: theme.custom?.surfaces?.inputBg || 'rgba(0,0,0,0.2)',
    '& fieldset': { borderColor: theme.custom?.surfaces?.inputBorder || 'rgba(255,255,255,0.2)' },
    '&:hover fieldset': { borderColor: theme.custom?.surfaces?.inputBorderHover || 'rgba(255,255,255,0.35)' },
    '&.Mui-focused fieldset': { borderColor: 'primary.main' },
  },
  '& .MuiInputBase-input': { color: 'text.primary', fontSize: '0.95rem' },
});

export default function PathAutocompleteInput({
  value = '',
  onChange,
  prefix = '',
  provider,
  dirsOnly = false,
  placeholder = 'relative/path',
  autoFocus = false,
  disabled = false,
}) {
  const theme = useTheme();
  const hasProvider = typeof provider === 'function';
  const { suggestions, loading, query, setQuery } = usePathComplete({ prefix, provider, dirsOnly });

  // Keep the query in sync with the controlled value (e.g. host resets / drill-in).
  useEffect(() => {
    setQuery(value || '');
  }, [value, setQuery]);

  const emit = (next) => {
    setQuery(next);
    onChange?.(next);
  };

  // Prefix adornment: a non-editable hint of the absolute base the relative path joins.
  const prefixAdornment = prefix ? (
    <Typography
      component="span"
      variant="caption"
      sx={{ color: 'text.disabled', mr: 0.5, whiteSpace: 'nowrap', userSelect: 'none', fontFamily: 'monospace' }}
    >
      {prefix.endsWith('/') ? prefix : `${prefix}/`}
    </Typography>
  ) : null;

  // No provider → plain (still relative) text field, with the same prefix adornment.
  if (!hasProvider) {
    return (
      <TextField
        fullWidth
        size="small"
        value={value}
        disabled={disabled}
        autoFocus={autoFocus}
        placeholder={placeholder}
        onChange={(e) => onChange?.(e.target.value)}
        variant="outlined"
        InputProps={{ startAdornment: prefixAdornment }}
        sx={inputSx(theme)}
      />
    );
  }

  return (
    <Autocomplete
      freeSolo
      fullWidth
      disabled={disabled}
      loading={loading}
      options={suggestions}
      filterOptions={(opts) => opts} // provider already filtered by `partial`
      inputValue={query}
      getOptionLabel={(opt) => (typeof opt === 'string' ? opt : relativeOf(opt, prefix))}
      isOptionEqualToValue={(opt, val) => relativeOf(opt, prefix) === relativeOf(val, prefix)}
      onInputChange={(_e, next, reason) => {
        if (reason === 'input' || reason === 'clear') emit(next);
      }}
      onChange={(_e, selected) => {
        if (selected && typeof selected === 'object') {
          const rel = relativeOf(selected, prefix);
          // Drill into directories (append "/" and re-query); commit files as-is.
          emit(selected.is_dir && !rel.endsWith('/') ? `${rel}/` : rel);
        } else if (typeof selected === 'string') {
          emit(selected);
        }
      }}
      renderOption={(props, opt) => {
        const { key, ...liProps } = props;
        return (
          <Box component="li" key={key} {...liProps} sx={{ display: 'flex', gap: 1, fontSize: '0.85rem' }}>
            <Typography component="span" sx={{ color: opt.is_dir ? 'primary.main' : 'text.primary' }}>
              {opt.name}{opt.is_dir ? '/' : ''}
            </Typography>
          </Box>
        );
      }}
      renderInput={(params) => (
        <TextField
          {...params}
          size="small"
          autoFocus={autoFocus}
          placeholder={placeholder}
          variant="outlined"
          InputProps={{
            ...params.InputProps,
            startAdornment: (
              <>
                {prefixAdornment}
                {params.InputProps.startAdornment}
              </>
            ),
          }}
          sx={inputSx(theme)}
        />
      )}
    />
  );
}
